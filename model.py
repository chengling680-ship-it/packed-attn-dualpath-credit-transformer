import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(project_root, 'src'))
sys.path.append(os.path.join(project_root, 'src', 'train'))
sys.path.append(os.path.join(project_root, 'src', 'transformer'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import Encoder_Packed, EncoderLayer_Packed
from attention import PackedAttention,PackedAttentionLayer
from emded import DataEmbedding_packed_Adjust_Attention
from masking import TriangularCausalMask, ProbMask,FeatureMask
#还未适用decoder，没有加入可查询部分
#from decoder import DecoderLayer_Packed,Decoder_Packed,DecoderLayer,Decoder


#-----vertical_Adjust模型-----
class VarTransformer_Adjust(nn.Module):
    def __init__(self, c_in, d_model, n_heads, e_layers, d_ff, dropout, gru_hidden=64):
        super(VarTransformer_Adjust, self).__init__()
        
        # 1. 嵌入层 (双重路径适配变长，支持特征拆分)
        self.embedding = DataEmbedding_packed_Adjust_Attention(
            c_in=c_in,
            d_model=d_model, 
            gru_hidden=gru_hidden, 
            dropout=dropout
        )

        
        #2. 编码器 (Packed 适配层)
        self.encoder = Encoder_Packed(
            [
                EncoderLayer_Packed(
                    PackedAttentionLayer(
                        PackedAttention(mask_flag=True, attention_dropout=dropout),
                        d_model, n_heads
                    ),
                    d_model, d_ff, dropout=dropout
                ) for _ in range(e_layers)
            ]
            # ],
            # norm_layer=nn.LayerNorm(d_model)
        )
        
        # 3. 简单的分类/回归投影
        #第一板斧修改分类头
        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            #nn.BatchNorm1d(d_model // 2),
            #nn.LayerNorm(d_model // 2),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, 1),
            # 注意：这里千万不要加 LayerNorm 或 BatchNorm
        )
        # 设置负偏置，调整初始预测概率
        self.projection[-1].bias.data.fill_(-0.5)
        
                    # ],
            # norm_layer=nn.LayerNorm(d_model)
        # )
        
        # 4. 温度平滑参数
        self.temperature = 1.0  # 温度系数，用于放大logits差异

    def forward(self, static_x, dynamic_x, x_mark, is_single_flag, cu_seqlens):
        # A. 生成块对角因果掩码 [Total_L, Total_L] attn_mask最终的掩码 初始掩码和块掩码
        attn_mask = TriangularCausalMask(cu_seqlens, device=static_x.device).mask
        
        # B. 嵌入映射: [Total_L, V] -> [Total_L, d_model]
        enc_out = self.embedding(static_x, dynamic_x, x_mark, is_single_flag, cu_seqlens)
        
        # D. 编码器计算
        enc_out, attns = self.encoder(enc_out, attn_mask=attn_mask)
        
        # D. 池化逻辑：提取每个序列的最后一个有效 Token (Last-Pooling)
        # 根据 cu_seqlens 找到每个序列的末尾索引
        last_indices = cu_seqlens[1:].long() - 1
        final_repr = enc_out[last_indices] # [B, d_model]
        
        # E. 输出
        logits = self.projection(final_repr)
        
        # 温度平滑：放大logits差异
        logits = logits / self.temperature
        
        return logits
