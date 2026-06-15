# Credit Risk Assessment with Transformer

基于 Transformer 的信用风险评估模型，采用变长序列处理技术，支持混合精度训练和早停机制。

## 📋 项目简介

本项目实现了一个基于 Transformer 的信用风险评估系统，主要特点包括：

- **变长序列处理**：使用 Packed 序列格式处理不同长度的用户行为序列
- **特征语义嵌入**：为每个特征分配唯一的语义向量，解决特征置换不变性问题
- **双路径融合**：结合静态特征和动态特征，通过交叉注意力机制融合信息
- **改进的损失函数**：采用 AUCFocalLoss，包含难例挖掘和中心正则化
- **早停机制**：基于验证集 AUC 指标自动停止训练，防止过拟合

## 🛠️ 技术栈

- **Python** 3.8+
- **PyTorch** 2.0+
- **NumPy** 1.24+
- **Pandas** 2.0+
- **scikit-learn** 1.2+
- **Matplotlib** 3.7+

## 📁 项目结构
train/ │ ├── exp.py # 实验入口，数据加载和训练流程 │ ├── trainer.py # 训练器实现，包含损失函数和优化策略 │ └── config.py # 配置类，管理超参数 ├── transformer/ │ ├── model.py # Transformer 模型架构 │ ├── encoder.py # 编码器实现 │ ├── attention.py # 注意力机制（支持交叉注意力） │ ├── emded.py # 嵌入层（特征语义嵌入） │ └── utils.py # 数据处理工具函数 └── README.md # 项目说明文档

# 🚀 快速开始

### 1. 环境准备

```bash
创建虚拟环境
conda create -n credit-risk python=3.8 conda activate credit-risk

安装依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas scikit-learn matplotlib pyarrow
```
### 2. 数据准备

确保数据已准备好并保存为 Parquet 格式：
data/ ├── split/ │ ├── train_df.parquet # 训练特征 │ ├── train_label.parquet # 训练标签 │ ├── val_df.parquet # 验证特征 │ ├── val_label.parquet # 验证标签 │ ├── test_df.parquet # 测试特征 │ └── test_label.parquet # 测试标签
### 3. 训练模型

```bash
cd train
python exp.py
```
### 4. 配置参数

主要配置参数位于 `config.py`：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `c_in` | 输入特征维度 | 200 |
| `d_model` | Transformer 隐藏层维度 | 128 |
| `n_heads` | 注意力头数 | 8 |
| `e_layers` | 编码器层数 | 3 |
| `d_ff` | 前馈网络维度 | 512 |
| `dropout` | Dropout 比例 | 0.3 |
| `learning_rate` | 学习率 | 1e-4 |
| `max_epochs` | 最大训练轮数 | 50 |
| `early_patience` | 早停耐心值 | 10 |

## 🧠 模型架构

### VarTransformer_Adjust

改进的变长序列 Transformer 模型：

1. **嵌入层**（`DataEmbedding_SkipGRU_Packed_Adjust`）
   - 静态特征嵌入：使用特征语义嵌入
   - 动态特征嵌入：使用 SkipGRU 处理时序信息
   - 交叉注意力融合：融合静态和动态路径

2. **编码器**（`Encoder_Packed`）
   - 多层注意力机制
   - 支持 Packed 序列格式
   - 块对角掩码确保序列独立性

3. **分类头**（`ResidualMLP`）
   - 深度残差 MLP
   - 负偏置初始化
   - 支持 Dropout 正则化

## 📊 损失函数

### AUCFocalLoss

复合损失函数，包含以下组件：

1. **Focal Loss**：处理类别不平衡，聚焦难分类样本
2. **难负样本挖掘**：选择得分最高的负样本进行训练
3. **中心正则化**：锚定负样本均值在 -1.0 附近
4. **不确定性损失**：惩罚模糊预测，鼓励明确分类
5. **正样本拉升**：强制正样本远离负样本锚点

## 🔍 训练流程
数据加载 → 数据集初始化 → 模型创建 → 训练循环 → 早停检查 → 模型保存

### 关键特性

- **混合精度训练**：使用 `torch.amp` 加速训练，减少内存占用
- **学习率调度**：使用 `CosineAnnealingWarmRestarts` 策略
- **梯度裁剪**：防止梯度爆炸
- **模型保存**：保存每个 epoch 的检查点和最佳模型

## 📈 评估指标

- **AUC-ROC**：评估模型的排序能力
- **F1-Score**：平衡精确率和召回率
- **KS 值**：衡量正负样本分布的分离程度
- **准确率**：整体分类准确率

## ⚠️ 注意事项

1. **内存管理**：
   - 训练前删除不必要的变量
   - 使用 `gc.collect()` 强制垃圾回收
   - 训练后清理 GPU 缓存：`torch.cuda.empty_cache()`

2. **数据预处理**：
   - 确保数据已进行归一化处理
   - 处理缺失值和异常值
   - 确保特征维度与模型输入匹配

3. **类别不平衡**：
   - 使用 `scale_pos_weight` 调整正样本权重
   - 考虑使用过采样或欠采样技术

4. **模型加载**：
   - 使用 `weights_only=False` 加载完整检查点
   - 使用 `strict=False` 允许加载不完整的权重

  
⭐ 本项目代码基于 MIT License 开源，仅供学术研究、学习交流使用，禁止直接商用落地。
如需用于企业信贷业务、商业盈利场景，请联系作者获得授权。
如果本项目对你的研究有帮助，欢迎点Star收藏支持！
