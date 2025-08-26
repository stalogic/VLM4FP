import os
import json
import torch
from datasets import Dataset
from transformers import (
    Qwen2VLProcessor,
    Qwen2TokenizerFast,
    Qwen2_5_VLForConditionalGeneration,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    Trainer
)

from splitted_embedding import SplittedEmbedding

# ----------------------
# 配置参数
# ----------------------
CONFIG = {
    # 模型配置
    "model_name": "/home/jiangmingming/mntspace/FPLLM/fpllm/FPQwen",
    "add_token_num": 224*2+8,  # 冻结前28层
    "device_id": 0,       # 指定使用的GPU编号（从0开始）
    
    # 数据配置
    "input_data_path": "/home/jiangmingming/mntspace/FPLLM/data/fpllm_token_identity.jsonl",  # 支持JSONL格式
    "data_format": "jsonl",               # 数据格式：json或jsonl
    "system_prompt": "你是一个智能助手，根据用户的要求提供准确有用的回答。",
    
    # 训练配置
    "output_dir": "./qwen2_freeze_sft_results",
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-5,
    "num_train_epochs": 3,
    "max_seq_length": 1024,
    "bf16": True,         # 使用BF16加速（需GPU支持）
    "logging_steps": 10,
    "save_strategy": "epoch"
}

# ----------------------
# 1. 初始化GPU环境
# ----------------------
def setup_gpu(device_id):
    """设置使用指定的GPU"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用，无法进行GPU训练")
    
    # 检查指定的GPU是否存在
    if device_id >= torch.cuda.device_count():
        raise ValueError(f"指定的GPU编号{device_id}不存在，可用GPU数量为{torch.cuda.device_count()}")
    
    # 设置可见的GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    device = torch.device(f"cuda:{device_id}")
    
    # 检查BF16支持
    if CONFIG["bf16"] and not torch.cuda.is_bf16_supported():
        raise RuntimeError("当前GPU不支持BF16格式，请将bf16设置为False")
    
    print(f"使用GPU: {torch.cuda.get_device_name(device)} (编号: {device_id})")
    print(f"BF16支持: {'已启用' if CONFIG['bf16'] else '未启用'}")
    return device

# ----------------------
# 2. 数据格式转换（支持JSONL）
# ----------------------
def convert_sft_to_qwen_format(sft_item, system_prompt):
    """将单条instruction/input/output数据转换为Qwen格式"""
    # 构建用户内容
    user_content = sft_item["instruction"]
    if sft_item.get("input") and sft_item["input"].strip():
        user_content += "\n" + sft_item["input"]
    
    # 构建消息列表
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": sft_item["output"]})
    
    return {"messages": messages}

def load_and_convert_data(input_path, data_format, system_prompt):
    """加载原始数据（支持JSON和JSONL）并转换为Qwen格式"""
    raw_data = []
    
    # 读取JSONL格式（每行一个JSON对象）
    if data_format == "jsonl":
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:  # 跳过空行
                    try:
                        item = json.loads(line)
                        raw_data.append(item)
                    except json.JSONDecodeError as e:
                        print(f"警告：跳过无效的JSON行 - {e}")
    
    # 读取标准JSON格式（JSON数组）
    elif data_format == "json":
        with open(input_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
    
    else:
        raise ValueError(f"不支持的数据格式: {data_format}，请使用'json'或'jsonl'")
    
    # 转换数据格式
    converted_data = [
        convert_sft_to_qwen_format(item, system_prompt)
        for item in raw_data
    ]
    
    # 转换为Dataset对象
    return Dataset.from_list(converted_data)

# ----------------------
# 3. 模型加载与冻结配置
# ----------------------
def load_model_and_tokenizer(model_name, device):
    """加载模型和分词器"""
    tokenizer = Qwen2TokenizerFast.from_pretrained(model_name)
    processor = Qwen2VLProcessor.from_pretrained(model_name)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map='auto')
    
    return model, tokenizer

def freeze_model_layers(model, added_token_num):
    """冻结指定数量的模型层"""
    for param in model.parameters():
        param.requires_grad = False

    # 解冻新增token对应的embedding参数
    input_embeddings = model.get_input_embeddings()
    model.set_input_embeddings(
        SplittedEmbedding(input_embeddings, added_token_num).to(model.device)
    )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"可训练参数数量: {trainable_params}, 总计参数数量: {total_params}, 可训练参数占比: {100*trainable_params/total_params:.4f}%")
    
    return model

# ----------------------
# 4. 数据预处理
# ----------------------
def preprocess_function(examples, tokenizer, max_seq_length):
    """预处理函数：仅对assistant的输出部分计算损失，忽略input部分"""
    # 1. 应用聊天模板生成文本（不tokenize，仅用于定位output位置）
    full_text = tokenizer.apply_chat_template(
        examples["messages"],
        tokenize=False,
        truncation=False
    )
    
    # 2. 单独对full_text进行tokenize，获取完整的input_ids
    inputs = tokenizer(
        full_text,
        truncation=True,
        max_length=max_seq_length,
        padding="max_length",
        return_tensors="pt"
    )
    input_ids = inputs["input_ids"].squeeze(0)  # 移除batch维度
    labels = input_ids.clone()  # 初始化labels
    
    # 3. 找到assistant消息（output）的起始位置，用于mask输入部分
    # 提取assistant的内容（最后一条消息）
    assistant_content = examples["messages"][-1]["content"]
    
    # 生成仅包含assistant内容的token（用于定位）
    assistant_tokens = tokenizer(
        assistant_content,
        add_special_tokens=False,
        return_tensors="pt"
    )["input_ids"].squeeze(0)
    assistant_len = len(assistant_tokens)
    
    # 4. 计算需要mask的长度（总长度 - output长度）
    # 注意：需预留结束符（eos_token）的位置
    if assistant_len > 0 and assistant_len <= len(input_ids):
        mask_length = len(input_ids) - assistant_len
        # 将input部分（instruction + input）的labels设为-100
        labels[:mask_length] = -100
    
    # 5. 处理超长截断的情况（确保mask_length有效）
    if mask_length < 0:
        labels[:] = -100  # 极端情况：output超长，全部mask（实际训练中应避免）
    
    inputs["labels"] = labels
    return inputs

# ----------------------
# 5. 训练主函数
# ----------------------
def main():
    # 设置GPU
    device = setup_gpu(CONFIG["device_id"])
    
    # 创建输出目录
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    
    # 加载并转换数据
    print(f"加载并转换{CONFIG['data_format']}格式数据...")
    dataset = load_and_convert_data(
        CONFIG["input_data_path"],
        CONFIG["data_format"],
        CONFIG["system_prompt"]
    )
    print(f"数据转换完成，样本数: {len(dataset)}")
    
    # 加载模型和分词器
    print(f"加载模型: {CONFIG['model_name']}")
    model, tokenizer = load_model_and_tokenizer(CONFIG["model_name"], device)
    
    # 冻结模型层
    print(f"冻结前{CONFIG['add_token_num']}层...")
    model = freeze_model_layers(model, CONFIG["add_token_num"])
    
    # 预处理数据集
    print("预处理数据集...")
    tokenized_dataset = dataset.map(
        lambda x: preprocess_function(x, tokenizer, CONFIG["max_seq_length"]),
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # 配置训练参数
    training_args = TrainingArguments(
        output_dir=CONFIG["output_dir"],
        per_device_train_batch_size=CONFIG["per_device_train_batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        learning_rate=CONFIG["learning_rate"],
        num_train_epochs=CONFIG["num_train_epochs"],
        bf16=CONFIG["bf16"],                # 使用BF16
        logging_steps=CONFIG["logging_steps"],
        save_strategy=CONFIG["save_strategy"],
        optim="adamw_torch",
        report_to="none",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        dataloader_pin_memory=True          # 启用pin_memory加速数据传输
    )
    
    # 初始化Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )
    
    # 开始训练
    print("开始训练...")
    trainer.train()
    
    # 保存最终模型
    final_checkpoint_dir = os.path.join(CONFIG["output_dir"], "final_checkpoint")
    trainer.save_model(final_checkpoint_dir)
    print(f"训练完成，模型已保存至: {final_checkpoint_dir}")

if __name__ == "__main__":
    main()
    