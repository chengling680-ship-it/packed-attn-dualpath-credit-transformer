import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt
from masking import TriangularCausalMask, ProbMask,FeatureMask


class PackedAttention(nn.Module):
    """
    真正无 Padding 的注意力层 (Flash Attention 风格)
    输入不再是 [B, L, E]，而是拼接后的 [Total_L, E]
    """
    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.1, output_attention=False):
        super(PackedAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        """
        Args:
            queries, keys, values: [Total_L, H, D] - 扁平化后的多头张量
            attn_mask: [Total_L, Total_L] 的 Block Diagonal Mask (块对角掩码)
                       或者使用 SDPA 内部支持的变长处理逻辑
        """
        # 计算缩放因子
        E = queries.shape[-1]
        scale = self.scale or 1. / sqrt(E)*0.25

        # 核心：使用 PyTorch 原生 SDPA
        # 当 attn_mask 是一个 [Total_L, Total_L] 的布尔矩阵时：
        # True 表示屏蔽(看不到)，False 表示可见
        
        # 转换维度以符合 SDPA 要求: [1, H, Total_L, D] (把总长度看作序列维度)
        # 注意：SDPA 期待的 shape 通常是 (B, H, L, D)
        q = queries.transpose(0, 1).unsqueeze(0) # [1, H, Total_L, D]
        k = keys.transpose(0, 1).unsqueeze(0)
        v = values.transpose(0, 1).unsqueeze(0)

        # 如果有输出注意力的需求，SDPA 目前不支持直接返回权重，需手动计算
        if self.output_attention:
            # 手动实现以获取权重 (非 Flash 路径，仅用于调试/分析)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            if attn_mask is not None:
                # 确保 attn_mask 形状正确
                if attn_mask.dim() == 2:
                    attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, Total_L, Total_L]
                # 确保掩码类型为布尔类型
                if not attn_mask.dtype == torch.bool:
                    attn_mask = attn_mask.bool()
                scores = scores.masked_fill(attn_mask, -float('inf'))
            # 对最后一个维度进行 softmax，确保每行和为1
            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            out = torch.matmul(attn, v)
        else:
            # 高效路径：触发 FlashAttention / Memory Efficient Attention
            # is_causal 会自动处理三角掩码，但如果是 Packed 序列，
            # 必须传入自定义的块对角掩码 attn_mask
            out = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=attn_mask, 
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False # 掩码逻辑由外部逻辑控制
            )
            attn = None

        # 恢复形状: [Total_L, H, D] 为什么这里训练batch是1，因为已经把多个batch拼接成一个了，视为一个batch
        out = out.squeeze(0).transpose(0, 1).contiguous()
        
        return out, attn  


class PackedAttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(PackedAttentionLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        """
        此时输入 queries/keys/values 形状均为 [Total_L, d_model]
        """
        H = self.n_heads

        # 1. 投影: [Total_L, d_model] -> [Total_L, n_heads * d_keys/d_values]
        q_proj = self.query_projection(queries)
        k_proj = self.key_projection(keys)
        v_proj = self.value_projection(values)

        # 2. 切分多头: [Total_L, H, d_keys]
        # 使用 -1 自动推导 Total_L，使用 H 明确头数
        # 最后一个维度使用 // H 也是安全的，因为投影层输出明确为 H * d_keys
        queries = q_proj.view(-1, H, q_proj.size(-1) // H)
        keys = k_proj.view(-1, H, k_proj.size(-1) // H)
        values = v_proj.view(-1, H, v_proj.size(-1) // H)
        

        out, attn = self.inner_attention(
            queries, keys, values,
            attn_mask, tau=tau, delta=delta
        )
        
        # 拼接多头并映射回 d_model
        out = out.view(-1, self.n_heads * (out.size(-1)))
        return self.out_projection(out), attn

