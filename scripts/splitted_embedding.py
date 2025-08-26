import torch
import torch.nn as nn


class SplittedEmbedding(nn.Module):
    def __init__(self, resized_embedding: nn.Embedding, num_new_tokens: int):
        super().__init__()
        # 原始embedding参数（冻结）
        self.original_weight = nn.Parameter(resized_embedding.weight.data[:-num_new_tokens].clone())
        self.original_weight.requires_grad = False  # 冻结原始参数
        
        # 新增token的embedding参数（可训练）
        self.new_weight = nn.Parameter(
            resized_embedding.weight.data[-num_new_tokens:].clone()
        )
        self.new_weight.requires_grad = True  # 仅新增参数可训练
        
        # 保持与原始embedding相同的配置
        self.embedding_dim = resized_embedding.embedding_dim
        self.padding_idx = resized_embedding.padding_idx

    @property
    def weight(self):
        # 动态拼接原始和新增embedding（通过property模拟统一的weight接口）
        return torch.cat([self.original_weight, self.new_weight], dim=0)

    def forward(self, input_ids: torch.Tensor):
        # 手动处理padding（如果需要）
        if self.padding_idx is not None:
            input_ids = input_ids.clamp(min=0)  # 避免padding_idx导致索引越界
        
        # 分离原始token和新增token的索引
        original_mask = input_ids < self.original_weight.size(0)
        new_mask = ~original_mask
        
        # 初始化输出embedding
        embeddings = torch.zeros(
            input_ids.size(0), input_ids.size(1), self.embedding_dim,
            device=input_ids.device, dtype=self.original_weight.dtype
        )
        
        # 填充原始token的embedding
        original_ids = input_ids[original_mask]
        if original_ids.numel() > 0:
            embeddings[original_mask] = torch.index_select(self.original_weight, 0, original_ids)
        
        # 填充新增token的embedding（索引偏移：减去原始词表大小）
        new_ids = input_ids[new_mask] - self.original_weight.size(0)
        if new_ids.numel() > 0:
            embeddings[new_mask] = torch.index_select(self.new_weight, 0, new_ids)
        
        return embeddings

