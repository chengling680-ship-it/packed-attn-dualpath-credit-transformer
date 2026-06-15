import torch.nn as nn
import torch.nn.functional as F
import torch

class EncoderLayer_Packed(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="gelu"):
        super(EncoderLayer_Packed, self).__init__()
        d_ff = d_ff or 4 * d_model#前馈维度
        self.attention = attention
        # 在 Packed 模式下，Conv1d 实际上是逐 Token 的线性映射，kernel_size 必须为 1
        self.conv1 = nn.Linear(d_model, d_ff)
        self.conv2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x: [Total_L, d_model]
        # 注意：这里不再需要 padding_mask，因为 Packed 架构没有 Padding
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)


# 注意力机制和 FFN 的组合形成了一个强大的特征提取单元：
# 注意力：负责"看哪里"（捕捉依赖关系）
# FFN：负责"如何处理看到的信息"（特征变换）


        # Feed Forward Network (FFN)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y)))
        y = self.dropout(self.conv2(y))

        return self.norm2(x + y), attn

class Encoder_Packed(nn.Module):
    def __init__(self, encoder_layers, norm_layer=None):
        super(Encoder_Packed, self).__init__()
        self.encoder_layers = nn.ModuleList(encoder_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [Total_L, D]
        attns = []
        for encoder_layer in self.encoder_layers:
            # 每一层共享同一个 Block Diagonal Mask
            x, attn = encoder_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns