import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from typing import List, Tuple
import pandas as pd

class VarLenDataset_Adjust(Dataset):
    """
    每个样本长度不一
    """
    def __init__(
        self, 
        data: pd.DataFrame, 
        targets: pd.DataFrame, 
        use_norm: bool = True, 
        seq_len: int = 21,

    ):
        self.seq_len = seq_len
        self.use_norm = use_norm
        self.data = data
        self.targets = targets.drop_duplicates(subset=['userid'], keep='first')

        data.sort_values(by=['userid', 'apply_time'], inplace=True)

        # 统计每个userid的序列长度
        # 注意：使用 sort=False 保持与排序后的 data 顺序一致
        user_counts = data.groupby('userid', sort=False).size()
        self.user_lengths = user_counts.values
        self.unique_userids = user_counts.index.tolist()

        # 【新增】计算每个用户在全局矩阵中的起始偏移量
        # 结果类似于: [0, len1, len1+len2, ...]
        self.offsets = np.zeros(len(self.user_lengths) + 1, dtype=np.int32)
        self.offsets[1:] = np.cumsum(self.user_lengths)

        feat = data.iloc[:, [0] + list(range(3, data.shape[1]-1))]
        time_gaps = data['time_gap'].values.astype(np.float32)


        # 1. 时间特征处理：Log1p变换
        self.time_gaps = np.log1p(time_gaps)
        
        # 2. 计算差分特征(动态特征)
        delta_feat = self._calculate_differences(feat)
        # 移除userid列，只保留特征列
        feat = feat.drop('userid', axis=1)
        
        # 3. 双重归一化
        if self.use_norm:
            # 静态路径：Z-Score归一化
            static_feat = self._z_score_normalization(feat)
            
            # 动态路径：RobustScaler归一化
            dynamic_feat = self._robust_scaler_normalization(delta_feat)
        else:
            static_feat = feat
            dynamic_feat = delta_feat
        
        # 4. 合并静态和动态特征，预转换为PyTorch张量
        self.static_fea = torch.tensor(static_feat.values.astype(np.float32))
        self.dynamic_fea = torch.tensor(dynamic_feat.values.astype(np.float32))


        
        # 5. 添加is_single_flag特征，预转换为PyTorch张量
        self.is_single_flags = torch.zeros(len(data), dtype=torch.float32)
        for i, length in enumerate(self.user_lengths):
            if length == 1:
                start = self.offsets[i]
                self.is_single_flags[start] = 1.0
        
        # 6. 时间间隔也转换为PyTorch张量
        self.time_gaps = torch.tensor(self.time_gaps, dtype=torch.float32)
        
        # 4. 标签（预先提取，避免运行时查询）
        # 1. 建立一个快速查询字典 {userid: label}
        # 确保处理了 targets 可能存在的重复项
        label_dict = dict(zip(self.targets['userid'], self.targets['label']))
        
        # 2. 按照 Dataset 实际识别出的 unique_userids 顺序提取标签
        # 这一步至关重要！它保证了 self.labels[i] 对应的是 self.unique_userids[i] 的特征
        self.labels = np.array(
            [label_dict[uid] for uid in self.unique_userids], 
            dtype=np.float32
        )
        
        # 5. 预计算last_time_index
        # 最后有效位置 = 该客户的序列长度 - 1（索引从0开始）
        self.last_indices = np.array(
            [length - 1 for length in self.user_lengths],
            dtype=np.int32
        )
        
        # print(f"Dataset: {len(self)} samples, shape={self.fea.shape}")
    def __len__(self):
        return len(self.targets) #总客户数
    def _calculate_differences(self, feat):
        """计算差分特征Δxt = xt - xt-1"""
        delta_feat = feat.copy()
        
        # 按用户分组计算差分
        # groupby 后使用 diff() 计算差分，然后填充第一个值为0
        delta_feat = feat.groupby('userid').diff().fillna(0)
    
        return delta_feat
    
    def _z_score_normalization(self, feat):
        """Z-Score归一化，带异常值处理（内存高效版）"""
        # 把 padding (-1) 视作 NaN
        feat = feat.where(feat != -1, other=np.nan)
        
        # 转换为float32减少内存使用
        feat_np = feat.values.astype(np.float32)
        
        # 计算分位数
        q1 = np.nanpercentile(feat_np, 1, axis=0)
        q99 = np.nanpercentile(feat_np, 99, axis=0)
        
        # 缩尾
        feat_np = np.clip(feat_np, q1, q99)
        
        # 计算均值和标准差
        mean = np.nanmean(feat_np, axis=0)
        std = np.nanstd(feat_np, axis=0)
        std = np.where(std == 0, 1.0, std)
        
        # 分块处理归一化，避免内存不足
        block_size = 100000  # 每块处理10万行
        normalized_feat_np = np.zeros_like(feat_np, dtype=np.float32)
        
        for i in range(0, len(feat_np), block_size):
            end = min(i + block_size, len(feat_np))
            normalized_feat_np[i:end] = (feat_np[i:end] - mean) / std
        
        # 再次缩尾到[-3, 3]
        normalized_feat_np = np.clip(normalized_feat_np, -3, 3)
        
        # 把 NaN 恢复为 -1
        normalized_feat_np = np.where(np.isnan(normalized_feat_np), -1.0, normalized_feat_np)
        
        # 转换回DataFrame
        normalized_feat = pd.DataFrame(normalized_feat_np, columns=feat.columns, index=feat.index)
        
        return normalized_feat
    
    def _robust_scaler_normalization(self, feat):
        """RobustScaler归一化（基于分位数），带异常值处理（内存高效版）"""
        # 把 padding (-1) 视作 NaN
        feat = feat.where(feat != -1, other=np.nan)
        
        # 转换为float32减少内存使用
        feat_np = feat.values.astype(np.float32)
        
        # 计算分位数
        q01 = np.nanpercentile(feat_np, 1, axis=0)
        q99 = np.nanpercentile(feat_np, 99, axis=0)
        
        # 缩尾
        feat_np = np.clip(feat_np, q01, q99)
        
        # 计算四分位数
        q1 = np.nanpercentile(feat_np, 25, axis=0)
        q3 = np.nanpercentile(feat_np, 75, axis=0)
        median = np.nanmedian(feat_np, axis=0)
        iqr = q3 - q1
        iqr = np.where(iqr == 0, 1.0, iqr)
        
        # 分块处理归一化，避免内存不足
        block_size = 100000  # 每块处理10万行
        normalized_feat_np = np.zeros_like(feat_np, dtype=np.float32)
        
        for i in range(0, len(feat_np), block_size):
            end = min(i + block_size, len(feat_np))
            normalized_feat_np[i:end] = (feat_np[i:end] - median) / iqr
        
        # 缩尾到[-3, 3]
        normalized_feat_np = np.clip(normalized_feat_np, -3, 3)
        
        # 把 NaN 恢复为 -1
        normalized_feat_np = np.where(np.isnan(normalized_feat_np), -1.0, normalized_feat_np)
        
        # 转换回DataFrame
        normalized_feat = pd.DataFrame(normalized_feat_np, columns=feat.columns, index=feat.index)
        
        return normalized_feat
    
    def __getitem__(self, idx: int):
        """
        根据索引 idx 提取该用户的变长序列
        """
        # 获取当前用户的起止位置
        start = self.offsets[idx]
        end = self.offsets[idx + 1]
        length = self.user_lengths[idx]

        # 1. 提取静态特征 [L_i, D]
        static_x = self.static_fea[start:end]
        
        # 2. 提取动态特征 [L_i, D]
        dynamic_x = self.dynamic_fea[start:end]
        
        # 3. 提取 time_gap [L_i]
        x_mark = self.time_gaps[start:end]
        
        # 4. 提取 is_single_flag [L_i]
        is_single_flag = self.is_single_flags[start:end]
        
        # 5. 构建 T (Decoder输入，取最后一行特征平铺) [L_i, D]
        T = static_x[-1:].repeat(length, 1)

        # 返回 Tensor（已经是张量，无需转换）
        return (
            static_x,      # 静态特征
            dynamic_x,      # 动态特征
            x_mark,         # 时间间隔（已Log1p变换）
            is_single_flag, # 是否为单条记录
            T,              # Decoder 输入
            torch.tensor(self.labels[idx])    # 标签
        )

# 4. 修改 collate_fn 以包含 time_gap (x_mark)
def packed_collate_fn_Adjust(batch):
    #batch是一个列表，里面包含batchsize个样本，每个样本 = 你的 __getitem__ 方法返回的那一组数据！
    # batch 结构: [(static_x, dynamic_x, x_mark, is_single_flag, T, label), ...]
    static_xs = [item[0] for item in batch]
    dynamic_xs = [item[1] for item in batch]
    x_marks = [item[2] for item in batch]
    is_single_flags = [item[3] for item in batch]
    Ts = [item[4] for item in batch]
    labels = [item[5] for item in batch]
    seqlens = [len(x) for x in static_xs]
    cu_seqlens = torch.tensor([0] + list(np.cumsum(seqlens)), dtype=torch.int32)
    
    return {
        "static_x": torch.cat(static_xs, dim=0),           # [Total_L, D]
        "dynamic_x": torch.cat(dynamic_xs, dim=0),           # [Total_L, D]
        "x_mark": torch.cat(x_marks, dim=0),                # [Total_L]
        "is_single_flag": torch.cat(is_single_flags, dim=0), # [Total_L]
        "T": torch.cat(Ts, dim=0),                          # [Total_L, D]
        "cu_seqlens": cu_seqlens,                           # [B + 1]
        "max_s": max(seqlens),
        "position_ids": torch.cat([torch.arange(l) for l in seqlens], dim=0),
        "labels": torch.stack(labels).unsqueeze(-1)          # [B, 1]
    }

def data_provider_vertical_Adjust(x, y, flag = 'train', batch_size = 128, use_feature_engineer=True):
    """
    数据提供函数，返回数据集和数据加载器
    
    Args:
        x: 输入数据，DataFrame格式
        y: 目标数据，DataFrame格式
        padding_mask: 可选的padding掩码（已废弃）
        flag: 数据集类型，'train'、'val'或'test'
        batch_size: 批量大小
        use_feature_engineer: 是否使用特征工程
    
    Returns:
        data_set: 数据集对象
        data_loader: 数据加载器对象
    """
    # 设置shuffle和drop_last参数
    if flag == 'test':
        shuffle_flag = False
        drop_last = False
    elif flag == 'val':
        shuffle_flag = False
        drop_last = False
    else:
        shuffle_flag = True  # 训练集应该shuffle
        drop_last = True     # 训练集应该drop_last
    
    # 创建变长数据集
    data_set = VarLenDataset_Adjust(x, y, use_norm=True)
    
    # 使用packed_collate_fn处理变长序列
    collate_fn = packed_collate_fn_Adjust
    
    # 创建数据加载器
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        drop_last=drop_last,
        num_workers=0,  # 使用0个工作进程，避免多进程问题
        pin_memory=True,  # 使用固定内存，加速GPU传输
        collate_fn=collate_fn  # 使用自定义的collate_fn处理变长序列
    )
    
    return data_set, data_loader
