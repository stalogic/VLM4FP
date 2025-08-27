import os
from transformers import (
    Qwen2VLProcessor,
    Qwen2TokenizerFast,
    Qwen2_5_VLForConditionalGeneration,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset
import torch


from splitted_embedding import SplittedEmbedding

IGNORE_INDEX = -100

DEFAULT_IM_START_TOKEN = "<|im_start|>"
DEFAULT_IM_END_TOKEN = "<|im_end|>"
DEFAULT_IMAGE_TOKEN = "<|image_pad|>"
DEFAULT_VIDEO_TOKEN = "<|video_pad|>"
LLAVA_IMAGE_TOKEN = "<image>"
LLAVA_VIDEO_TOKEN = "<video>"
VISION_START_TOKEN = "<|vision_start|>"
VISION_END_TOKEN = "<|vision_end|>"

SYSTEM_MESSAGE = "You are a helpful assistant."

MULTIMODAL_KEYWORDS = ["pixel_values", "image_grid_thw", "video_grid_thw", "pixel_values_videos", "second_per_grid_ts"]


# ----------------------
# 1. 加载原始模型和tokenizer
# ----------------------
model_path = os.getenv("RAW_VLM4FP_MODEL_PATH")
save_path = os.getenv("NEW_TOKEN_FINETUNED_VLM4FP_MODEL_PATH")
assert model_path is not None
assert save_path is not None
tokenizer = Qwen2TokenizerFast.from_pretrained(model_path)
processor = Qwen2VLProcessor.from_pretrained(model_path)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map='auto')
model.gradient_checkpointing_enable()
print(f"{model.device=}, {model.dtype=}")

# 记录原始词表大小
original_vocab_size = 151665 # vocab_size in Qwen2.5_VL-3B-Instruct
print(f"原始词表大小: {original_vocab_size}")

added_token_num = 224 * 2 + 8
new_vocab_size = len(tokenizer)
print(f"{added_token_num=}, {new_vocab_size-original_vocab_size=}")

# ----------------------
# 4. 冻结原有参数，仅解冻新增token的embedding
# ----------------------
# 冻结所有参数
for param in model.parameters():
    param.requires_grad = False

# 解冻新增token对应的embedding参数
original_input_embeddings = model.get_input_embeddings()
splited_input_embeddings = SplittedEmbedding(original_input_embeddings, added_token_num).to(model.device)
model.set_input_embeddings(splited_input_embeddings)

param_ratio = 1

# 验证：检查可训练参数数量（应等于新增token数 × embedding维度 × 2，若LM Head独立）
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"可训练参数数量: {trainable_params}, 总计参数数量: {total_params}, 可训练参数占比: {100*trainable_params/total_params:.4f}%")


# ----------------------
# 5. 准备数据和训练配置
# ----------------------
# 加载示例数据集（可替换为自定义数据）

def pad_sequence(input_ids, max_seq_length:int=128, padding_value:int=0):
    batch_size = len(input_ids)
    output = input_ids[0].new_full((batch_size, max_seq_length), padding_value)
    for i, input_id in enumerate(input_ids):
        length = len(input_id)
        output[i, :length] = input_id
        
    return output

def preprocess_function(example):
    system_message = f"{DEFAULT_IM_START_TOKEN}system\n{SYSTEM_MESSAGE}{DEFAULT_IM_END_TOKEN}\n"
    system_input_ids = processor.tokenizer(system_message, add_special_tokens=False, return_tensors="pt")['input_ids']
    system_labels = torch.full_like(system_input_ids, IGNORE_INDEX)

    inputs = example["instruction"] + example["input"]
    output = example["output"]
    user_input = f"{DEFAULT_IM_START_TOKEN}user\n{inputs}{DEFAULT_IM_END_TOKEN}\n{DEFAULT_IM_START_TOKEN}assistant\n"
    ai_output = f"{output}{DEFAULT_IM_END_TOKEN}\n"

    prompt_input_ids = processor.tokenizer(user_input, add_special_tokens=False, padding=False, return_tensors="pt")['input_ids']
    prompt_labels = torch.full_like(prompt_input_ids, IGNORE_INDEX)
    response_input_ids = processor.tokenizer(ai_output, add_special_tokens=False, padding=False, return_tensors="pt")['input_ids']

    input_ids = torch.cat([system_input_ids, prompt_input_ids, response_input_ids], dim=-1)
    labels = torch.cat([system_labels, prompt_labels, response_input_ids], dim=-1)
    input_ids = pad_sequence(input_ids, padding_value=processor.tokenizer.pad_token_id)
    labels = pad_sequence(labels, padding_value=IGNORE_INDEX)
    attention_mask = input_ids.ne(processor.tokenizer.pad_token_id).to(torch.long)
    model_inputs = {
        "input_ids": input_ids[0],
        "attention_mask": attention_mask[0],
        "labels": labels[0],
    }
    del system_input_ids, system_labels, prompt_input_ids, prompt_labels, response_input_ids
    torch.cuda.empty_cache()  # 清理GPU缓存
    return model_inputs


dataset = load_dataset('./data/')
tokenized_dataset = dataset.map(preprocess_function, batched=False, remove_columns=['instruction', 'input', 'output'])

# 数据整理器（用于语言模型训练）

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding="max_length",
)

# 训练参数配置
training_args = TrainingArguments(
    output_dir="./training_logs/new_token_training",
    learning_rate=5e-4,
    per_device_train_batch_size=25,
    num_train_epochs=10,
    logging_dir="./logs",
    logging_steps=100,
    save_steps=1000,
    fp16=False,  # 若支持GPU混合精度训练
    bf16=True,
    bf16_full_eval=True,
)

# ----------------------
# 6. 训练模型
# ----------------------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    # data_collator=data_collator,
    # optimizers=(optim, None),
)
trainer.train()

# ----------------------
# 7. 保存模型和tokenizer
# ----------------------

# 将splited_embedding参数复制到原来的embedding参数中
original_input_embeddings.weight.data[-added_token_num:,] = splited_input_embeddings.new_weight.data
model.set_input_embeddings(original_input_embeddings)

# 保存训练后的模型和tokenizer
model.save_pretrained(save_path)
processor.save_pretrained(save_path)