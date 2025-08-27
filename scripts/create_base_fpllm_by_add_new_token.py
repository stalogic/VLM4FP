import os
import torch
from transformers import Qwen2VLProcessor
from transformers import Qwen2TokenizerFast
from transformers import Qwen2_5_VLForConditionalGeneration
from torch.distributions.multivariate_normal import MultivariateNormal

model_path = os.getenv("QWEN_VL_MODEL_PATH")
save_path = os.getenv("RAW_VLM4FP_MODEL_PATH")
assert model_path is not None
assert save_path is not None

tokenizer = Qwen2TokenizerFast.from_pretrained(model_path)
processor = Qwen2VLProcessor.from_pretrained(model_path)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map='auto')

orientations = ['N', 'S', 'E', 'W', 'FN', 'FS', 'FE', 'FW']
orient_tokens = [f'<ORIENT_{o}>' for o in orientations]
row_tokens = []
col_tokens = []
for i in range(224):
    row_tokens.append(f'<ROW_{i}>')
    col_tokens.append(f'<COL_{i}>')

print(f"original embeded shape: {model.get_input_embeddings().weight.shape}")
print(f"original vocab size: {len(tokenizer)}")
custom_tokens = row_tokens + col_tokens + orient_tokens
print(f'{len(custom_tokens)=}')
num_added = tokenizer.add_tokens(custom_tokens, special_tokens=False)
print(f"add token num: {num_added}")
print(f"new vocab size: {len(tokenizer)}")

sentence = 'place a macro at grid (<ROW_0>, <COL_1>) with orientation <ORIENT_N>'
print(tokenizer(sentence).tokens())

processor.tokenizer = tokenizer
print(f'{model.get_input_embeddings().weight.data.shape=}')
print(f'{model.lm_head.weight.data.shape=}')
model.resize_token_embeddings(len(tokenizer))
with torch.no_grad():
    # 获取原始嵌入（排除新增token的位置）
    old_embeddings = model.get_input_embeddings().weight.data[:-num_added]  # 形状：(原始词表大小, 隐藏层维度)
    hidden_size = old_embeddings.shape[1]

    # 计算原始嵌入的均值和协方差
    mean = old_embeddings.mean(dim=0)  # 均值：(隐藏层维度,)
    centered = old_embeddings - mean.unsqueeze(0)  # 中心化嵌入（减去均值）
    covariance = (centered.T @ centered) / (old_embeddings.shape[0] - 1)  # 协方差矩阵：(隐藏层维度, 隐藏层维度)

    # 为协方差矩阵添加微小抖动（避免数值不稳定，如奇异矩阵）
    jitter = 1e-5 * torch.eye(hidden_size, device=covariance.device)
    covariance = covariance + jitter

    # 构建多元正态分布
    mvn = MultivariateNormal(loc=mean.float(), covariance_matrix=covariance.float())

    # 从分布中采样新嵌入
    new_embeddings = mvn.sample((num_added,)).bfloat16()  # 形状：(新增token数, 隐藏层维度)

    # 将新嵌入赋值给模型
    model.get_input_embeddings().weight.data[-num_added:] = new_embeddings

print(f'{model.get_input_embeddings().weight.data.shape=}')
print(f'{model.lm_head.weight.data.shape=}')


tokenizer.save_pretrained(save_path)
processor.save_pretrained(save_path)
model.save_pretrained(save_path)

print('add new token and init new token embedding done.')