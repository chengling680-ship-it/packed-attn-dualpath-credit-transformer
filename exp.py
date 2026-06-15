import pandas as pd
import torch
from torch.optim import AdamW
from torch.nn import BCEWithLogitsLoss
import numpy as np
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(project_root, 'src'))
sys.path.append(os.path.join(project_root, 'src', 'train'))
sys.path.append(os.path.join(project_root, 'src', 'transformer'))
import gc

from transformer.model import VarTransformer,CreditModel_FWS,WholeModel,HybridCreditFusion,VarTransformer_Adjust
from transformer.utils import data_provider_vertical, data_provider_horizontal,data_provider_fusion,data_provider_vertical_Adjust
from trainer import VarLenCreditTrainer,FWSTrainer,WholeModelTrainer,HybridCreditTrainer,VarLenCreditTrainer_Adjust

from sklearn.model_selection import train_test_split
from config import OptimalConfig
import matplotlib.pyplot as plt
import seaborn as sns

def train_val_test_split(
    df: pd.DataFrame, 
    label_df: pd.DataFrame, 
    test_ratio: float = 0.6
) :
    """
    极致优化版本：使用索引操作
    
    适用场景：
    - df已经按userid排序
    - 或者可以先排序（排序一次，后续操作都变快）
    """
    
    assert isinstance(df, pd.DataFrame), '输入不是DataFrame'
    
    print("="*80)
    print(f"初始坏样本比例: {label_df['label'].mean():.4f}")
    print("="*80)
    
    # ========== 步骤1: 分层采样（同前）==========
    train_val_label, test_label = train_test_split(
        label_df, test_size=test_ratio, stratify=label_df['label'], random_state=666
    )
    train_label, _ = train_test_split(
        train_val_label, test_size=test_ratio, stratify=train_val_label['label'], random_state=666
    )
    test_label, val_label= train_test_split(
        test_label, test_size=test_ratio, stratify=test_label['label'], random_state=666
    )
    del train_val_label
    
    print(f"训练集坏样本比例: {train_label['label'].mean():.4f} (样本数: {len(train_label)})")
    print(f"验证集坏样本比例: {val_label['label'].mean():.4f} (样本数: {len(val_label)})")
    print(f"测试集坏样本比例: {test_label['label'].mean():.4f} (样本数: {len(test_label)})")
    print("="*80)
    
    # ========== 步骤2: 使用索引加速（关键优化）==========
    print("构建索引...")
    
    # 如果df有userid作为索引，直接使用
    if df.index.name == 'userid':
        df_indexed = df
    else:
        # 设置userid为索引（排序一次，后续查询都很快）
        df_indexed = df.set_index('userid', drop=False)
    
    print("正在构建数据集（使用索引）...")
    
    # 使用.loc索引查询（比merge更快）
    train_df = df_indexed.loc[train_label['userid']].reset_index(drop=True)
    val_df = df_indexed.loc[val_label['userid']].reset_index(drop=True)
    test_df = df_indexed.loc[test_label['userid']].reset_index(drop=True)
    
    
    return train_df, train_label, val_df, val_label, test_df, test_label

def main():
    # 1. 加载配置
    config = OptimalConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. 加载数据 (假设数据已经由工程脚本处理好)
    print("正在加载数据...")
    # 读取数据
    # data = pd.read_parquet(r'C:\Users\28761\Desktop\variable_transformer\code\ver_var\data\feature\df_gain_data_666.parquet')
    # targets = pd.read_csv(r'C:\Users\28761\Desktop\variable_transformer\code\ver_var\data\ver_var\label\label_gain_data_666.csv')

    # # 类型优化
    # data = data.astype({col: 'float32' for col in data.select_dtypes('float64').columns})
    # targets = targets.astype({col: 'float32' for col in targets.select_dtypes('float64').columns})

    # # 数据分割
    # train_df, train_label, val_df, val_label, test_df, test_label = train_val_test_split(data, targets, test_ratio=0.2)
    
    #     #     # 保存分割后的数据集（保护机制）
    # import os
    # save_dir = r'C:\Users\28761\Desktop\variable_transformer\code\ver_var\data\split'
    # os.makedirs(save_dir, exist_ok=True)
    
    # print(f"💾 保存分割后的数据集到: {save_dir}")
    # train_df.to_parquet(os.path.join(save_dir, 'train_df.parquet'))
    # train_label.to_parquet(os.path.join(save_dir, 'train_label.parquet'))
    # val_df.to_parquet(os.path.join(save_dir, 'val_df.parquet'))
    # val_label.to_parquet(os.path.join(save_dir, 'val_label.parquet'))
    # test_df.to_parquet(os.path.join(save_dir, 'test_df.parquet'))
    # test_label.to_parquet(os.path.join(save_dir, 'test_label.parquet'))
    # print("✅ 分割后的数据集保存完成")


    save_dir = r'C:\Users\28761\Desktop\variable_transformer\code\ver_var\data\split'
    train_df = pd.read_parquet(os.path.join(save_dir, 'train_df.parquet'))
    train_label = pd.read_parquet(os.path.join(save_dir, 'train_label.parquet'))
    val_df = pd.read_parquet(os.path.join(save_dir, 'val_df.parquet'))
    val_label = pd.read_parquet(os.path.join(save_dir, 'val_label.parquet'))
    test_df = pd.read_parquet(os.path.join(save_dir, 'test_df.parquet'))
    test_label = pd.read_parquet(os.path.join(save_dir, 'test_label.parquet'))
    

    # 4. 初始化变长模型
    # 选择模型类型：'horizontal' 或 'vertical' 或 'whole'
    model_type = 'vertical_Adjust'  

    if model_type == 'vertical_Adjust':
        train_dataset_vertical_Adjust, train_loader_vertical_Adjust = data_provider_vertical_Adjust(
        train_df, train_label, 
        flag='train', 
        batch_size=config.batch_size
        )
        val_dataset_vertical_Adjust, val_loader_vertical_Adjust = data_provider_vertical_Adjust(
        val_df, val_label, 
        flag='val', 
        batch_size=config.batch_size
        )

        test_dataset_vertical_Adjust, test_loader_vertical_Adjust = data_provider_vertical_Adjust(
        test_df, test_label, 
        flag='test', 
        batch_size=config.batch_size
        )

        print("="*80)
        print("模型创建model_vertical_Adjust".center(80))
        print("="*80)
        seed=666
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)

        model_vertical_Adjust = VarTransformer_Adjust(
            c_in=config.c_in,
            d_model=config.d_model,
            n_heads=config.n_heads,
            e_layers=config.e_layers,
            d_ff=config.d_ff,
            dropout=config.dropout,
            gru_hidden=config.gru_hidden_dim
        ).to(device)

    del train_df, val_df, test_df, train_label, val_label, test_label  #可选data, targets, 
    gc.collect()

    # del val_df, test_df, val_label, test_label  #可选data, targets, 
    # gc.collect()


    # ==================== 5. 启动训练 ====================
    if model_type == 'vertical_Adjust':
    # 将 config 转为字典格式传给训练器
        trainer_vertical_config = {
            'lr': config.learning_rate,
            'weight_decay': config.weight_decay,
            'epochs': config.max_epochs,
            'patience': config.early_patience,
            'grad_clip': 5.0
        }

        trainer = VarLenCreditTrainer_Adjust(
            model=model_vertical_Adjust,
            train_loader=train_loader_vertical_Adjust,
            val_loader=val_loader_vertical_Adjust,
            test_loader=test_loader_vertical_Adjust,
            config=trainer_vertical_config,
            experiment_name="vertical_Adjust_Exp"
        )
        # 加载已有checkpoint并继续训练
        checkpoint_path = r"C:\Users\28761\Desktop\variable_transformer\code\wholemodel\experiments\vertical_Adjust_Exp_0525_1246\checkpoints\checkpoint_epoch_8.pth"
        if os.path.exists(checkpoint_path):
            print(f"📁 加载已有checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            # 使用strict=False允许加载不完整的权重（例如新添加的interaction层）
            model_vertical_Adjust.load_state_dict(checkpoint['model_state_dict'], strict=False)
            # 可以选择加载优化器状态
            #trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"✅ 成功加载第 {checkpoint['epoch']} 轮的模型权重")
            print("⚠️ 注意：新添加的层（如interaction层）将使用随机初始化的权重")
        else:
            print("⚠️ 未找到指定的checkpoint文件，将从头开始训练")
        print("🚀 开始变长序列 Transformer——vertical_Adjust 训练任务...")
        trainer.fit()  
        
        # ==================== 清理内存 ====================
        print("🧹 清理本次训练循环的内存...")
        
        # 1. 删除模型（如果在GPU上，先移到CPU再删除）
        if model_vertical_Adjust is not None:
            model_vertical_Adjust = model_vertical_Adjust.cpu()
            del model_vertical_Adjust
        
        # 2. 删除训练器
        if 'trainer' in locals():
            del trainer
        
        # 3. 删除数据集和数据加载器
        if 'train_dataset_vertical_Adjust' in locals():
            del train_dataset_vertical_Adjust
        if 'val_dataset_vertical_Adjust' in locals():
            del val_dataset_vertical_Adjust
        if 'test_dataset_vertical_Adjust' in locals():
            del test_dataset_vertical_Adjust
        if 'train_loader_vertical_Adjust' in locals():
            del train_loader_vertical_Adjust
        if 'val_loader_vertical_Adjust' in locals():
            del val_loader_vertical_Adjust
        if 'test_loader_vertical_Adjust' in locals():
            del test_loader_vertical_Adjust
        
        # 4. 清理 GPU 缓存（如果使用GPU）
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 5. 强制垃圾回收
        gc.collect()
        
        print("✅ 内存清理完成")
    
if __name__ == "__main__":
    main()