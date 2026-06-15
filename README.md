# Dual-Path Transformer Credit Risk Prediction
基于PyTorch搭建面向百万级异构信贷数据的信用风险评估模型框架
## 项目背景
信贷业务中普遍存在客户行为序列长度参差不齐、静态基础画像与动态时序行为特征难以融合两大工程痛点，本框架针对性完成架构优化与算力提速。
## 核心技术实现
1. Packed-Attention算子：摒弃冗余Padding填充，搭配块对角因果掩码，显存利用率大幅提升，训练速度提升约2.8倍，推理延迟稳定0.1ms
2. 双路径嵌入分支：时间衰减Skip-GRU提取动态时序趋势，线性分支处理静态用户特征；Adaptive Alpha自适应权重融合变长特征
3. 不平衡样本优化：融合Focal Loss与Pairwise AUC Loss，搭配Adaptive Margin动态决策阈值，KS指标提升12%
## 性能指标
百万级信贷样本测试集：AUC 0.81
## 技术栈
Python 3.9 / PyTorch 2.0 / SDPA / NumPy / Pandas
## 运行方式
1. 安装环境：pip install -r requirements.txt
2. 调用内置虚拟数据集快速启动训练：python train/exp.py
