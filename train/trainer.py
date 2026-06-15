# trainer.py
import time
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, 
    recall_score, f1_score, confusion_matrix,roc_curve
)
import numpy as np
import numpy as np
from tabulate import tabulate
from tqdm import tqdm
import math

class SimpleGradientChecker:
    """极简梯度检查器 - 专注于找出NaN根源"""
    
    def __init__(self, model, log_every=10, log_dir=None):
        self.model = model
        self.log_every = log_every
        self.batch_count = 0
        self.log_dir = log_dir
        
    def check(self, batch_idx, loss=None, outputs=None, inputs=None):
        """
        一次性检查所有问题
        
        Args:
            batch_idx: 当前batch
            loss: 损失值
            outputs: 模型输出
            inputs: 输入数据字典 {'x_t': tensor, 'x_T': tensor, 'y': tensor}
        
        Returns:
            问题类型: None/'nan'/'vanishing'/'exploding'
        """
        if batch_idx % self.log_every != 0:
            return None
        
        # 准备日志内容
        log_lines = []
        log_lines.append(f"{'='*60}")
        log_lines.append(f"梯度检查 - Batch {batch_idx}")
        log_lines.append(f"{'='*60}")
        
        # 1️⃣ 检查输入
        if inputs:
            for name, tensor in inputs.items():
                if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                    log_lines.append(f"❌ 输入 {name} 存在NaN/Inf")
                    stats = self._get_stats(tensor)
                    log_lines.extend(stats)
                    self._write_log(log_lines)
                    return 'input_nan'
        
        # 2️⃣ 检查输出
        if outputs is not None:
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                log_lines.append(f"❌ 模型输出存在NaN/Inf")
                stats = self._get_stats(outputs, "outputs")
                log_lines.extend(stats)
                self._write_log(log_lines)
                return 'output_nan'
            
            # 检查输出范围
            out_max = outputs.abs().max().item()
            if out_max > 100:
                log_lines.append(f"⚠️  输出过大: max={out_max:.2e}")
        
        # 3️⃣ 检查损失
        if loss is not None:
            if torch.isnan(loss) or torch.isinf(loss):
                log_lines.append(f"❌ 损失为NaN/Inf: {loss.item()}")
                self._write_log(log_lines)
                return 'loss_nan'
            
            if loss.item() > 100:
                log_lines.append(f"⚠️  损失过大: {loss.item():.2e}")
        
        # 4️⃣ 检查梯度（关键！）
        issue, grad_logs = self._check_gradients()
        log_lines.extend(grad_logs)
        
        log_lines.append(f"{'='*60}\n")
        self._write_log(log_lines)
        return issue
    
    def _write_log(self, log_lines):
        """将日志写入文件"""
        if not self.log_dir:
            # 如果没有指定日志目录，回退到打印到终端
            for line in log_lines:
                print(line)
            return
        
        log_file = self.log_dir / "gradient_check.log"
        with open(log_file, 'a', encoding='utf-8') as f:
            for line in log_lines:
                f.write(line + '\n')
    
    def _get_stats(self, tensor, name="tensor"):
        """获取张量的统计信息"""
        stats = []
        stats.append(f"{name} 统计:")
        stats.append(f"  均值: {tensor.mean().item():.4f}")
        stats.append(f"  标准差: {tensor.std().item():.4f}")
        stats.append(f"  最小值: {tensor.min().item():.4f}")
        stats.append(f"  最大值: {tensor.max().item():.4f}")
        return stats
    
    def _check_gradients(self):
        """检查梯度 - 核心逻辑"""
        grad_logs = []
        grad_logs.append(f"梯度统计:")
        grad_logs.append(f"{'层名':<40} {'范数':>12} {'均值':>12} {'最大值':>12}")
        grad_logs.append("-" * 80)
        
        has_nan = False
        has_vanishing = False
        has_exploding = False
        
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            
            grad = param.grad
            
            # 检查NaN
            if torch.isnan(grad).any():
                grad_logs.append(f"❌ {name:<40} NaN detected!")
                has_nan = True
                continue
            
            if torch.isinf(grad).any():
                grad_logs.append(f"❌ {name:<40} Inf detected!")
                has_nan = True
                continue
            
            # 计算统计量
            grad_norm = grad.norm().item()
            grad_mean = grad.abs().mean().item()
            grad_max = grad.abs().max().item()
            
            # 判断问题
            status = "✓"
            if grad_norm < 1e-7:
                status = "⚠️  消失"
                has_vanishing = True
            elif grad_norm > 10:
                status = "⚠️  爆炸"
                has_exploding = True
            
            # 添加到日志（只记录关键层）
            if 'weight' in name or status != "✓":
                grad_logs.append(f"{status} {name:<38} {grad_norm:>12.2e} {grad_mean:>12.2e} {grad_max:>12.2e}")
        
        # 返回问题类型和日志
        issue = None
        if has_nan:
            issue = 'nan'
        elif has_exploding:
            issue = 'exploding'
        elif has_vanishing:
            issue = 'vanishing'
        return issue, grad_logs
    
    def _print_stats(self, tensor, name):
        """打印张量统计信息"""
        print(f"{name} 统计:")
        print(f"  形状: {tensor.shape}")
        print(f"  NaN数: {torch.isnan(tensor).sum().item()}")
        print(f"  Inf数: {torch.isinf(tensor).sum().item()}")
        valid = tensor[~torch.isnan(tensor) & ~torch.isinf(tensor)]
        if len(valid) > 0:
            print(f"  Min: {valid.min().item():.4f}")
            print(f"  Max: {valid.max().item():.4f}")
            print(f"  Mean: {valid.mean().item():.4f}")

class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=7, verbose=False, delta=0, mode='min'):
        """
        Args:
            patience: 容忍多少个epoch没有改善
            verbose: 是否打印信息
            delta: 最小改善量
            mode: 'min' 或 'max'，监控指标是越小越好还是越大越好
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.mode = mode
        
        if mode == 'min':
            self.monitor_op = np.less
            self.best_score = np.Inf
        else:
            self.monitor_op = np.greater
            self.best_score = -np.Inf
    
    def __call__(self, score):
        """
        Args:
            score: 当前监控指标的值
        Returns:
            是否应该早停
        """
        #对象的属性，存在对象里，永久保存，不会重置
        if self.mode == 'min':
            improved = score < self.best_score - self.delta
        else:
            improved = score > self.best_score + self.delta
        
        if improved:
            self.best_score = score
            self.counter = 0
            if self.verbose:
                print(f'✓ 指标改善: {score:.6f}')
            return False
        else:
            self.counter += 1
            if self.verbose:
                print(f'✗ EarlyStopping counter: {self.counter}/{self.patience}')
            
            if self.counter >= self.patience:
                self.early_stop = True
                return True
            return False

class AUCFocalLoss(nn.Module):
    """
    改进版: 结合 AUC 优化的 Focal Loss
    - 支持负样本采样
    - 自适应 margin
    """

    def __init__(self,
                 focal_alpha=0.75,
                 focal_gamma=2.0,
                 pos_weight=10.0,
                 neg_sample_ratio=0.7):
        """
        Args:
            focal_alpha: Focal Loss 中调整正负样本的 α
            focal_gamma: Focal Loss 中的 γ（用于困难样本权重）
            pos_weight: BCE 中给正样本的额外权重
            auc_weight: AUC Loss 与 Focal Loss 的权重比例
            neg_sample_ratio: 计算 AUC Loss 时负样本采样比例（0~1）
        """
        super().__init__()
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.pos_weight = pos_weight
        self.neg_sample_ratio = neg_sample_ratio

    def forward(self, preds, targets):
        preds = preds.view(-1)
        targets = targets.view(-1).float()
        if torch.isnan(preds).any():
            print("⚠️  Warning: NaN detected in loss input")
        # === 1. Focal Loss ===
        bce_loss = F.binary_cross_entropy_with_logits(
            preds, targets, reduction='none',
            pos_weight=torch.tensor([self.pos_weight], device=preds.device)
        )
        probs = torch.sigmoid(preds)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1.0 - pt).pow(self.focal_gamma)
        alpha_weight = torch.where(targets == 1, self.focal_alpha, 1 - self.focal_alpha)
        focal_loss = (alpha_weight * focal_weight * bce_loss).mean()

        # === 2. AUC Loss (改进版 Pairwise Ranking) ===
        pos_mask = targets == 1
        neg_mask = targets == 0

        if pos_mask.sum() > 0 and neg_mask.sum() > 0:
            pos_preds = preds[pos_mask]  # (n_pos,)
            neg_preds = preds[neg_mask]  # (n_neg,)

            # ---- 2.1 负样本采样 ----
            if self.neg_sample_ratio < 1.0:
                n_neg = neg_preds.size(0)
                sample_size = max(1, int(n_neg * self.neg_sample_ratio))
                idx = torch.randperm(n_neg, device=preds.device)[:sample_size]
                neg_preds = neg_preds[idx]

            # ---- 2.2 自适应 margin ----
            # 思路: margin = 正样本均值 - 负样本均值的一半
            mean_pos = pos_preds.mean()
            mean_neg = neg_preds.mean()
            # 确保 margin > 0
            margin = torch.clamp((mean_pos - mean_neg) * 0.5, min=0.1, max=2.0)

            # ---- 2.3 Pairwise Hinge Loss ----
            pos_preds_expanded = pos_preds.unsqueeze(1)  # (n_pos, 1)
            neg_preds_expanded = neg_preds.unsqueeze(0)  # (1, n_neg)
            pairwise_loss = torch.relu(margin - (pos_preds_expanded - neg_preds_expanded))
            auc_loss = pairwise_loss.mean()
        else:
            auc_loss = torch.tensor(0.0, device=preds.device)

        # === 3. Combine ===
        total_loss =  focal_loss + auc_loss
        return total_loss

class MetricsCalculator:
    """指标计算器"""
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                         y_pred_proba: np.ndarray) -> Dict[str, float]:
        """
        计算所有评估指标
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签（二分类）
            y_pred_proba: 预测概率
        
        Returns:
            包含所有指标的字典
        """
        metrics = {}
        
        # AUC
        try:
            metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
        except ValueError:
            metrics['auc'] = float('nan')
        
        # 基础分类指标
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
        
        # 混淆矩阵
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            metrics['tn'] = int(tn)
            metrics['fp'] = int(fp)
            metrics['fn'] = int(fn)
            metrics['tp'] = int(tp)
            
            # 计算特异度和敏感度
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        except ValueError:
            metrics['tn'] = metrics['fp'] = metrics['fn'] = metrics['tp'] = 0
            metrics['specificity'] = metrics['sensitivity'] = 0
        
        # 计算KS指标
        try:
            fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
            ks = max(tpr - fpr)
            metrics['ks'] = ks
        except Exception:
            metrics['ks'] = float('nan')
        return metrics
    
    @staticmethod
    def print_confusion_matrix(metrics: Dict[str, float]):
        """打印混淆矩阵"""
        table_data = [
            ["", "Predicted Negative", "Predicted Positive"],
            ["Actual Negative", metrics['tn'], metrics['fp']],
            ["Actual Positive", metrics['fn'], metrics['tp']]
        ]
        print("\n" + "="*50)
        print("Confusion Matrix:")
        print(tabulate(table_data, headers="firstrow", tablefmt="grid"))
        print("="*50 + "\n")



#vertical模型trainer调整
class VarLenCreditTrainer_Adjust:
    def __init__(self, model, train_loader, val_loader,test_loader=None, config=None, experiment_name="VarTransformer_Adjust_vertical"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.experiment_name = experiment_name
        
        # 1. 实验环境设置 (参考原类)
        self._setup_experiment()
        
        # 2. 核心组件 (适配 Packed 架构)
        self.criterion = AUCFocalLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=config.get('learning_rate', 1e-4), 
            weight_decay=config.get('weight_decay', 1e-5)
        )
        # 实施余弦退火学习率调度，使用WarmRestarts
        # T_0=10, T_mult=2，在第10轮时突然把学习率调大，利用震荡让模型跳出局部最优
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=5, T_mult=2)
        # 3. 监控工具
        self.grad_checker = SimpleGradientChecker(self.model, log_dir=self.log_dir)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        self.early_stopping = EarlyStopping(patience=config.get('early_patience', 10), mode='max')
        self.metrics_calc = MetricsCalculator()

    def _setup_experiment(self):
        """创建实验目录在src同级文件夹"""
        timestamp = datetime.now().strftime("%m%d_%H%M")
        # 获取项目根目录（src的同级目录）
        project_root = Path(__file__).parent.parent.parent
        self.exp_dir = project_root / "experiments" / f"{self.experiment_name}_{timestamp}"
        self.checkpoint_dir = self.exp_dir / "checkpoints"
        self.log_dir = self.exp_dir / "logs"
        
        for d in [self.checkpoint_dir, self.log_dir]:
            d.mkdir(parents=True, exist_ok=True)
            
        with open(self.exp_dir / "config.json", 'w') as f:
            json.dump(self.config, f, indent=4)

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        preds_list, targets_list = [], []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            # 解构变长数据包（处理字典格式）
            # static_x: [Total_L, D], dynamic_x: [Total_L, D], x_mark: [Total_L], is_single_flag: [Total_L], cu_seqlens: [B+1], labels: [B, 1]
            static_x = batch["static_x"].to(self.device)
            dynamic_x = batch["dynamic_x"].to(self.device)
            x_mark = batch["x_mark"].to(self.device)
            is_single_flag = batch["is_single_flag"].to(self.device)
            cu_seqlens = batch["cu_seqlens"].to(self.device)
            y = batch["labels"].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward: 传入 cu_seqlens 用于 block-diagonal mask
            outputs = self.model(static_x, dynamic_x, x_mark, is_single_flag, cu_seqlens)
            
            # 结果提取：信贷预测通常取每个序列的最后一步
            # 你的 model.py 如果没在内部处理，这里需要做索引提取
            
            logits = outputs

            loss = self.criterion(logits, y.float())
            loss.backward()
            
            # 梯度检查 (防止变长计算中的梯度爆炸)
            self.grad_checker.check(batch_idx, loss, logits)
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get('grad_clip', 5.0))
            self.optimizer.step()

            total_loss += loss.item()
            preds_list.append(torch.sigmoid(logits).detach().cpu().numpy())
            targets_list.append(y.detach().cpu().numpy())

        avg_loss = total_loss / len(self.train_loader)
        self.writer.add_scalar('Loss/Train', avg_loss, epoch)
        return avg_loss

    @torch.no_grad()
    def evaluate(self, loader, epoch, tag="Val"):
        self.model.eval()
        all_preds, all_targets = [], []
        
        for batch in loader:
            # 解构变长数据包（处理字典格式）
            static_x = batch["static_x"].to(self.device)
            dynamic_x = batch["dynamic_x"].to(self.device)
            x_mark = batch["x_mark"].to(self.device)
            is_single_flag = batch["is_single_flag"].to(self.device)
            cu_seqlens = batch["cu_seqlens"].to(self.device)
            y = batch["labels"].to(self.device)
            
            outputs = self.model(static_x, dynamic_x, x_mark, is_single_flag, cu_seqlens)
            
            logits = outputs
            all_preds.append(torch.sigmoid(logits).cpu().numpy())
            all_targets.append(y.cpu().numpy())
            
        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)
        
        # 寻找最优阈值（基于F1-score）
        def find_optimal_threshold(y_true, y_score):
            thresholds = np.arange(0.4, 0.61, 0.01)
            f1_scores = []
            for threshold in thresholds:
                y_pred = (y_score > threshold).astype(int)
                f1 = f1_score(y_true, y_pred)
                f1_scores.append(f1)
            optimal_idx = np.argmax(f1_scores)
            return thresholds[optimal_idx], f1_scores[optimal_idx]
        
        # 计算最优阈值
        optimal_threshold, best_f1 = find_optimal_threshold(targets, preds)
        
        # 使用最优阈值计算指标
        metrics = self.metrics_calc.calculate_metrics(targets, (preds > optimal_threshold).astype(int), preds)
        metrics['optimal_threshold'] = optimal_threshold
        metrics['best_f1'] = best_f1
        
        # 记录到 Tensorboard
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                self.writer.add_scalar(f'{tag}/{k}', v, epoch)
        
        return metrics

    def _save_checkpoint(self, epoch, metrics, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),#self.model的状态
            'optimizer_state_dict': self.optimizer.state_dict(),#self.optimizer的状态信息 是一个python字典
            'metrics': metrics,
            'config': self.config
        }
        filename = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, filename)
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / "best_model.pth")

    def fit(self):
        print(f"🚀 开始训练实验: {self.experiment_name}")
        for epoch in range(self.config['epochs']):
            train_loss = self.train_epoch(epoch)
            val_metrics = self.evaluate(self.val_loader, epoch)
            
            self.scheduler.step()
            
            # 打印表格（参考原有的整洁输出）
            print(f"\nEpoch {epoch} | Loss: {train_loss:.4f} | AUC: {val_metrics['auc']:.4f} | KS: {val_metrics['ks']:.4f}")
            
            # 保存与早停
            is_best = self.early_stopping(val_metrics['auc'])
            self._save_checkpoint(epoch, val_metrics, is_best)
            
            # 清理内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("🧹 清理GPU内存")
            
            if self.early_stopping.early_stop:
                print("🛑 触发早停，训练结束")
                break
        self.writer.close()

        """在训练结束后，加载最佳权重进行最终测试"""
        print("🔍 开始在测试集上进行最终评估...")
    
       # 1. 加载训练过程中保存的最佳模型权重
        best_path = self.checkpoint_dir / "best_model.pth"
        if best_path.exists():
           checkpoint = torch.load(best_path, weights_only=False)
           self.model.load_state_dict(checkpoint['model_state_dict'])
           print(f"✅ 已加载第 {checkpoint['epoch']} 轮的最佳模型")

       # 2. 调用原有的 evaluate 逻辑
        if self.test_loader is not None:
            test_metrics = self.evaluate(self.test_loader, epoch=999, tag="Test")
        
         # 3. 打印最终结果
            print("\n" + "="*30)
            print("Final Test Results:")
            for k, v in test_metrics.items():
               print(f"{k.upper()}: {v:.4f}")
            print("="*30)
            return test_metrics
        else:
            print("⚠️ 未提供测试集加载器，跳过测试。")
