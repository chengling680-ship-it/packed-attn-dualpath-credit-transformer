import torch
import torch.nn as nn
import math
from typing import Optional
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(project_root, 'src'))
sys.path.append(os.path.join(project_root, 'src', 'train'))
sys.path.append(os.path.join(project_root, 'src', 'transformer'))
from transformer.attention import PackedAttentionLayer, PackedAttention
class SkipGRU(nn.Module):
    """处理不规则时间序列的GRU"""
    
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        
        # GRU三个门
        self.update_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.reset_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.candidate = nn.Linear(input_size + hidden_size, hidden_size)
        
        # 可学习的时间衰减参数
        self.decay_weight = nn.Parameter(torch.tensor(0.1))
    
    def forward(
        self, 
        x: torch.Tensor,         # [batch_size, input_size]
        time_gap: torch.Tensor,  # [batch_size]
        hidden: torch.Tensor     # [batch_size, hidden_size]
    ) -> torch.Tensor:
        """
        Args:
            x: 当前时刻输入
            time_gap: 时间间隔
            hidden: 上一时刻隐藏状态
        
        Returns:
            new_hidden: 新的隐藏状态
        """
        # 时间衰减
        if time_gap.dim() == 1:
            time_gap = time_gap.unsqueeze(-1)  # [batch_size, 1]
        decay = torch.exp(-torch.abs(self.decay_weight) * time_gap)
        
        hidden_decayed = hidden * decay


        # 把 padding 用 0 填充（或其他合适的填充值）
        x_filled = torch.where(x == -1, torch.zeros_like(x), x)
        # 将填充值与 mask 相乘（冗余但表达清晰）

        
        # 门控计算
        combined = torch.cat([x_filled, hidden_decayed], dim=-1)
        # 判断 NaN
        if torch.isnan(combined).any():
            print("Warning: NaN detected in  combined input")
            
        update = torch.sigmoid(self.update_gate(combined))
        
        reset = torch.sigmoid(self.reset_gate(combined))
        candidate = torch.tanh(
            self.candidate(torch.cat([x, reset * hidden_decayed], dim=-1))
        )
        
        new_hidden = (1 - update) * hidden_decayed + update * candidate # [batch_size, hidden_size]
        
        return new_hidden


class SkipGRU_Packed(nn.Module):
    """
    专门为 Packed 序列优化的 SkipGRU。
    利用 cu_seqlens 批量处理变长序列，避免 Python 显式循环导致的低效。
    """
    def __init__(self, input_size: int, hidden_size: int, d_model: int):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 标准 GRU 单元
        self.gru_cell = SkipGRU(input_size, hidden_size)
        
        # 时间衰减参数
        # 注意：SkipGRU 已经包含了 decay_weight 参数，这里不需要重复定义
        self.decay_weight = nn.Parameter(torch.tensor(0.1))
        self.projection = nn.Linear(hidden_size, d_model)

    def forward(self, x, x_mark, cu_seqlens):
        """
        Args:
            x: [Total_L, V]
            x_mark: [Total_L] (时间间隔)
            cu_seqlens: [B + 1] 记录每个 Sequence 的起始偏移
        """
        total_l = x.size(0)
        batch_size = cu_seqlens.size(0) - 1
        device = x.device
        
        # 初始化所有样本的隐藏状态
        # 注意：在 Packed 模式下，我们需要模拟序列演进
        #outputs = torch.zeros(total_l, self.hidden_size, device=device)
        #数据类型修复
        outputs = torch.zeros(total_l, self.hidden_size, device=device, dtype=x.dtype)


        # 获取最大长度用于循环遍历（这是唯一的循环，次数=Max_L）
        max_len = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
        
        # 当前 batch 的隐藏状态
        #h = torch.zeros(batch_size, self.hidden_size, device=device)
        #数据类型修复
        h = torch.zeros(batch_size, self.hidden_size, device=device, dtype=x.dtype)
        
        for t in range(int(max_len)):
            # 找到在该时间步仍有数据的 batch 索引
            # (即序列长度 > t 的样本)
            active_mask = (cu_seqlens[1:] - cu_seqlens[:-1]) > t
            active_indices = torch.where(active_mask)[0]
            
            if len(active_indices) == 0:
                break
                
            # 计算这些活跃样本在 Total_L 维度中的实际索引
            # 实际位置 = cu_seqlens[batch_idx] + t
            curr_indices = cu_seqlens[active_indices] + t
            
            x_t = x[curr_indices]
            delta_t = x_mark[curr_indices].unsqueeze(-1)
            
            # 时间衰减逻辑
            h_active = h[active_indices]
            decay = torch.exp(-torch.abs(self.decay_weight) * delta_t)
            h_decayed = h_active * decay
            
            # GRU 更新
            h_new = self.gru_cell(x_t, delta_t.squeeze(-1),h_decayed)
            
            # 写回结果
            # h[active_indices] = h_new
            # outputs[curr_indices] = h_new
            # 数据类型修复
            h[active_indices] = h_new.to(h.dtype)
            outputs[curr_indices] = h_new.to(outputs.dtype)
            
        return self.projection(outputs)

class SkipGRU_Packed_Adjust(nn.Module):
    """
    专门为 Packed 序列优化的 SkipGRU。
    利用向量化操作替代 Python 循环，提高性能。
    """
    def __init__(self, input_size: int, hidden_size: int, d_model: int):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 标准 GRU 单元
        self.gru_cell = SkipGRU(input_size, hidden_size)
        
        # 时间衰减参数
        self.decay_weight = nn.Parameter(torch.tensor(0.1))
        self.projection = nn.Linear(hidden_size, d_model)

    def forward(self, x, x_mark, cu_seqlens):
        """
        Args:
            x: [Total_L, V]
            x_mark: [Total_L] (时间间隔)
            cu_seqlens: [B + 1] 记录每个 Sequence 的起始偏移
        """
        total_l = x.size(0)
        batch_size = cu_seqlens.size(0) - 1
        device = x.device
        
        # 初始化输出和隐藏状态
        outputs = torch.zeros(total_l, self.hidden_size, device=device, dtype=x.dtype)
        h = torch.zeros(batch_size, self.hidden_size, device=device, dtype=x.dtype)
        
        # 计算每个序列的长度
        seq_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        max_len = seq_lengths.max().item()
        
        # 预计算所有序列的起始和结束索引
        seq_starts = cu_seqlens[:-1]
        seq_ends = cu_seqlens[1:]
        
        # 向量化处理：使用序列级别的操作
        # 注意：这里仍然需要循环时间步，但使用向量化操作处理每个时间步的所有序列
        for t in range(max_len):
            # 计算当前时间步在每个序列中的位置
            # 对于每个序列，检查是否已经超过长度
            valid_mask = seq_lengths > t
            if not valid_mask.any():
                break
            
            # 获取有效的序列索引
            valid_seq_indices = torch.where(valid_mask)[0]
            
            # 计算当前时间步在总序列中的索引
            curr_indices = seq_starts[valid_seq_indices] + t
            
            # 提取当前时间步的输入和时间间隔
            x_t = x[curr_indices]
            delta_t = x_mark[curr_indices].unsqueeze(-1)
            
            # 提取当前序列的隐藏状态
            h_active = h[valid_seq_indices]
            
            # 时间衰减
            decay = torch.exp(-torch.abs(self.decay_weight) * delta_t)
            h_decayed = h_active * decay
            
            # GRU 更新
            h_new = self.gru_cell(x_t, delta_t.squeeze(-1), h_decayed)
            
            # 更新隐藏状态和输出
            h[valid_seq_indices] = h_new
            outputs[curr_indices] = h_new
        
        return self.projection(outputs)
    
class FeatureSemanticEmbedding(nn.Module):
    """
    优化版本：避免创建 [Total_L, c_in, d_model] 的大中间张量
    使用线性层直接映射，内存效率更高
    """
    def __init__(self, c_in=200, d_model=128):
        super(FeatureSemanticEmbedding, self).__init__()
        
        # 1. 特征重要性门控（保留）
        self.gate = nn.Sequential(nn.Linear(c_in, c_in), nn.Sigmoid())
        
        # 2. 直接映射层：将 [Total_L, c_in] 直接映射到 [Total_L, d_model]
        # 这等价于：value_projection(x) * id_emb 然后沿特征维度聚合
        self.projection = nn.Linear(c_in, d_model)
        
        # 初始化
        nn.init.normal_(self.projection.weight, std=0.02)
        if self.projection.bias is not None:
            nn.init.zeros_(self.projection.bias)

    def forward(self, x):
        """
        x: [Total_L, c_in] (静态或动态特征)
        返回: [Total_L, d_model]
        """
        # 1. 计算特征重要性门控
        g = self.gate(x)  # [Total_L, c_in]
        x = x * g  # 应用门控权重
        
        # 2. 直接线性映射：[Total_L, c_in] -> [Total_L, d_model]
        # 避免创建 [Total_L, c_in, d_model] 的大张量
        output = self.projection(x)  # [Total_L, d_model]
        
        return output

class DataEmbedding_packed_Adjust_Attention(nn.Module):
    """
    双重路径嵌入层 (Dual-Path Embedding Layer)，适应变长序列 (Packed/Flattened)。
    输入不再是 [B, L, V]，而是 [Total_L, V]。
    """
    def __init__(self, c_in: int, d_model: int, gru_hidden: int = 64, dropout: float = 0.1):
        super(DataEmbedding_packed_Adjust_Attention, self).__init__()
        
        # 1. 静态路径：基础画像（使用特征语义嵌入）
        self.static_embedding = FeatureSemanticEmbedding(c_in=c_in, d_model=d_model)
        
        # 2. 动态路径：差分特征 + 时间间隔
        # 预定义动态特征的语义嵌入
        self.dynamic_semantic_embedding = FeatureSemanticEmbedding(c_in=c_in, d_model=d_model)
        self.dynamic_embedding = SkipGRU_Packed_Adjust(d_model, gru_hidden, d_model)

        # 添加交叉注意力层
        self.cross_attention = PackedAttentionLayer(
            PackedAttention(mask_flag=False, attention_dropout=dropout),
            d_model,
            n_heads=4  # 可以根据需要调整
        )

        # 自适应融合权重
        self.adaptive_alpha = nn.Parameter(torch.tensor(0.5))
        
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, static_x, dynamic_x, x_mark, is_single_flag, cu_seqlens):
        """
        Args:
            static_x: [Total_L, V] - 静态特征
            dynamic_x: [Total_L, V] - 动态特征（差分）
            x_mark: [Total_L] - 时间间隔
            is_single_flag: [Total_L] - 是否为单条记录
            cu_seqlens: [B+1] - 序列长度累积
        """
        # 1. Static Stream: 基础画像（使用特征语义嵌入）
        static_emb = self.static_embedding(static_x)
        
        # 2. Dynamic Stream: 处理差分特征和时间间隔
        # 先通过特征语义嵌入，再通过SkipGRU
        semantic_emb = self.dynamic_semantic_embedding(dynamic_x)
        dynamic_emb = self.dynamic_embedding(semantic_emb, x_mark, cu_seqlens)
        
        # 3. 交叉注意力融合
        # 将静态路径作为 K 和 V，动态路径作为 Q
        cross_emb, _ = self.cross_attention(
            dynamic_emb,  # Q: 来自动态路径
            static_emb,   # K: 来自静态路径
            static_emb,   # V: 来自静态路径
            attn_mask=None  # 交叉注意力通常不需要掩码
        )

        # 4. Adaptive Fusion: 自适应融合
        # 根据序列长度自动调整权重
        # 对于单条记录，减少动态路径的权重
        batch_size = cu_seqlens.size(0) - 1
        seq_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        
        # 向量化计算权重：序列长度为1时使用0.1，否则使用自适应权重
        alpha = torch.where(seq_lengths == 1, torch.tensor(0.1, device=static_emb.device), self.adaptive_alpha)
        
        # 向量化扩展权重到所有时间步
        alpha_expanded = torch.repeat_interleave(alpha, seq_lengths)    
    
        # 融合静态和动态路径
        fused_emb = static_emb + alpha_expanded.unsqueeze(1) * cross_emb
    
        # 层归一化和dropout
        fused_emb = self.layer_norm(fused_emb)
        return self.dropout(fused_emb)  
