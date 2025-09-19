"""
自适应加权模块实现

该模块用于处理多组学数据的噪声，通过动态学习样本权重来缓解数据异质性噪声，
提升模型鲁棒性和药物反应预测准确性。

核心功能：
1. 权重学习网络：基于训练损失学习样本权重
2. 动态权重调整：根据样本贡献调整权重
3. 权重动态更新：跟踪样本历史表现
4. 加权损失函数：将权重融入损失计算
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import defaultdict


class WeightNetwork(nn.Module):
    """
    权重学习网络
    
    输入：训练损失值
    输出：样本权重 [0, 1]
    """
    
    def __init__(self, input_dim: int = 1, hidden_dim: int = 64, dropout: float = 0.1):
        """
        初始化权重网络
        
        Args:
            input_dim: 输入维度（损失值维度）
            hidden_dim: 隐藏层维度
            dropout: dropout概率
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 网络结构：输入层 -> 隐藏层 -> 输出层
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(), 
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # 确保权重在[0,1]范围内
        )
        
        # 权重初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, losses: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            losses: 样本损失值，形状为 [batch_size] 或 [batch_size, 1]
            
        Returns:
            weights: 样本权重，形状为 [batch_size, 1]
        """
        if losses.dim() == 1:
            losses = losses.unsqueeze(-1)  # [batch_size, 1]
        
        # 标准化损失值，避免数值不稳定
        losses_normalized = F.layer_norm(losses, losses.shape[1:])
        
        weights = self.layers(losses_normalized)
        return weights


class SampleTracker:
    """
    样本追踪器
    
    跟踪每个样本的历史表现，用于动态权重调整
    """
    
    def __init__(self, ema_decay: float = 0.9, min_weight: float = 0.1):
        """
        初始化样本追踪器
        
        Args:
            ema_decay: 指数移动平均衰减因子
            min_weight: 最小权重值
        """
        self.ema_decay = ema_decay
        self.min_weight = min_weight
        self.sample_stats = defaultdict(lambda: {'loss_ema': 0.0, 'count': 0, 'contribution': 0.0})
    
    def update_sample_stats(self, sample_ids: List[int], losses: torch.Tensor, 
                           weights: torch.Tensor) -> None:
        """
        更新样本统计信息
        
        Args:
            sample_ids: 样本ID列表
            losses: 样本损失值
            weights: 当前样本权重
        """
        # Convert to numpy, handling different tensor shapes
        if losses.dim() == 1:
            losses_np = losses.detach().cpu().numpy()
        else:
            losses_np = losses.detach().cpu().numpy().flatten()
            
        if weights.dim() == 1:
            weights_np = weights.detach().cpu().numpy()
        else:
            weights_np = weights.detach().cpu().numpy().flatten()
        
        for i, sample_id in enumerate(sample_ids):
            stats = self.sample_stats[sample_id]
            current_loss = float(losses_np[i])
            current_weight = float(weights_np[i])
            
            # 更新指数移动平均损失
            if stats['count'] == 0:
                stats['loss_ema'] = current_loss
            else:
                stats['loss_ema'] = (self.ema_decay * stats['loss_ema'] + 
                                    (1 - self.ema_decay) * current_loss)
            
            # 更新贡献度（权重越高，损失越低，贡献越大）
            contribution_score = current_weight / (1 + current_loss)
            stats['contribution'] = (self.ema_decay * stats['contribution'] + 
                                   (1 - self.ema_decay) * contribution_score)
            
            stats['count'] += 1
    
    def get_historical_weights(self, sample_ids: List[int]) -> torch.Tensor:
        """
        基于历史表现获取样本权重
        
        Args:
            sample_ids: 样本ID列表
            
        Returns:
            weights: 基于历史的权重，形状为 [batch_size, 1]
        """
        historical_weights = []
        
        for sample_id in sample_ids:
            stats = self.sample_stats[sample_id]
            if stats['count'] == 0:
                # 新样本给予中等权重
                weight = 0.5
            else:
                # 基于历史贡献计算权重
                # 贡献度高的样本获得较高权重
                contribution = stats['contribution']
                weight = max(self.min_weight, min(1.0, contribution))
            
            historical_weights.append(weight)
        
        # Convert to numpy array first, then to tensor
        weights_array = np.array(historical_weights, dtype=np.float32)
        return torch.from_numpy(weights_array).unsqueeze(-1)


class AdaptiveWeightedLoss(nn.Module):
    """
    自适应加权损失函数
    
    将样本权重融入交叉熵损失计算
    """
    
    def __init__(self, ignore_index: int = 0, reduction: str = 'mean'):
        """
        初始化加权损失函数
        
        Args:
            ignore_index: 忽略的索引值（通常是padding token）
            reduction: 损失聚合方式 ('mean', 'sum', 'none')
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor, 
                weights: torch.Tensor, loss_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算加权交叉熵损失
        
        Args:
            logits: 模型预测logits，形状为 [batch_size, seq_len, vocab_size]
            targets: 目标标签，形状为 [batch_size, seq_len]
            weights: 样本权重，形状为 [batch_size, 1]
            loss_mask: 损失掩码，形状为 [batch_size, seq_len]
            
        Returns:
            weighted_loss: 加权损失值
        """
        # 计算基础交叉熵损失（不聚合）
        flat_logits = logits.view(-1, logits.size(-1))
        flat_targets = targets.view(-1)
        
        # 计算每个位置的损失
        base_loss = F.cross_entropy(flat_logits, flat_targets, 
                                   ignore_index=self.ignore_index, reduction='none')
        
        # 重塑损失为 [batch_size, seq_len]
        loss = base_loss.view(targets.shape)
        
        # 应用loss_mask
        if loss_mask is not None:
            loss = loss * loss_mask
            valid_positions = loss_mask.sum(dim=1, keepdim=True).float()  # [batch_size, 1]
            # 计算每个样本的平均损失
            sample_loss = loss.sum(dim=1, keepdim=True) / (valid_positions + 1e-8)
        else:
            sample_loss = loss.mean(dim=1, keepdim=True)  # [batch_size, 1]
        
        # 应用样本权重
        weighted_sample_loss = sample_loss * weights
        
        # 根据reduction方式聚合
        if self.reduction == 'mean':
            return weighted_sample_loss.mean()
        elif self.reduction == 'sum':
            return weighted_sample_loss.sum()
        else:  # 'none'
            return weighted_sample_loss


class AdaptiveWeightingModule:
    """
    自适应加权模块主类
    
    整合权重网络、样本追踪器和加权损失函数
    """
    
    def __init__(self, weight_net_hidden_dim: int = 64, 
                 ema_decay: float = 0.9, min_weight: float = 0.1,
                 device: str = 'cuda', update_frequency: int = 100):
        """
        初始化自适应加权模块
        
        Args:
            weight_net_hidden_dim: 权重网络隐藏层维度
            ema_decay: 指数移动平均衰减因子
            min_weight: 最小权重值
            device: 计算设备
            update_frequency: 权重网络更新频率（每多少步更新一次）
        """
        self.device = device
        self.update_frequency = update_frequency
        self.step_count = 0
        
        # 初始化组件
        self.weight_network = WeightNetwork(hidden_dim=weight_net_hidden_dim).to(device)
        self.sample_tracker = SampleTracker(ema_decay=ema_decay, min_weight=min_weight)
        self.weighted_loss_fn = AdaptiveWeightedLoss()
        
        # 权重网络优化器
        self.weight_optimizer = torch.optim.Adam(self.weight_network.parameters(), lr=1e-4)
    
    def compute_adaptive_weights(self, losses: torch.Tensor, 
                                sample_ids: Optional[List[int]] = None) -> torch.Tensor:
        """
        计算自适应权重
        
        Args:
            losses: 样本损失值
            sample_ids: 样本ID（用于历史跟踪）
            
        Returns:
            weights: 自适应权重
        """
        # 使用权重网络预测权重
        predicted_weights = self.weight_network(losses)
        
        # 如果有样本ID，结合历史信息
        if sample_ids is not None:
            historical_weights = self.sample_tracker.get_historical_weights(sample_ids).to(self.device)
            # 结合预测权重和历史权重
            combined_weights = 0.7 * predicted_weights + 0.3 * historical_weights
        else:
            combined_weights = predicted_weights
        
        return combined_weights
    
    def compute_weighted_loss(self, logits: torch.Tensor, targets: torch.Tensor,
                             loss_mask: Optional[torch.Tensor] = None,
                             sample_ids: Optional[List[int]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算加权损失
        
        Args:
            logits: 模型预测logits
            targets: 目标标签
            loss_mask: 损失掩码
            sample_ids: 样本ID
            
        Returns:
            weighted_loss: 加权损失
            weights: 使用的权重
        """
        # 首先计算基础损失以获得每个样本的损失值
        with torch.no_grad():
            base_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), 
                                      targets.view(-1), ignore_index=0, reduction='none')
            base_loss = base_loss.view(targets.shape)
            
            if loss_mask is not None:
                base_loss = base_loss * loss_mask
                valid_positions = loss_mask.sum(dim=1, keepdim=True).float()
                sample_losses = base_loss.sum(dim=1, keepdim=True) / (valid_positions + 1e-8)
            else:
                sample_losses = base_loss.mean(dim=1, keepdim=True)
        
        # 计算自适应权重（不参与梯度计算）
        with torch.no_grad():
            weights = self.compute_adaptive_weights(sample_losses, sample_ids)
        
        # 计算加权损失（权重作为常量）
        weighted_loss = self.weighted_loss_fn(logits, targets, weights, loss_mask)
        
        # 更新样本统计信息
        if sample_ids is not None:
            with torch.no_grad():
                self.sample_tracker.update_sample_stats(sample_ids, sample_losses, weights)
        
        # 分离式更新权重网络（仅在特定步骤） - 暂时禁用以避免梯度冲突
        # if self.step_count % self.update_frequency == 0:
        #     with torch.no_grad():
        #         self._update_weight_network_detached(sample_losses, weights)
        
        self.step_count += 1
        
        return weighted_loss, weights
    
    def _update_weight_network_detached(self, losses: torch.Tensor, current_weights: torch.Tensor):
        """
        分离式更新权重网络参数（避免梯度冲突）
        
        Args:
            losses: 当前损失
            current_weights: 当前权重
        """
        # 确保权重网络处于训练模式
        self.weight_network.train()
        
        # 创建新的张量用于权重网络训练，需要梯度
        losses_input = losses.detach().clone().requires_grad_(False)
        
        # 权重网络的训练目标：预测能够最小化损失的权重
        predicted_weights = self.weight_network(losses_input)
        
        # 构造训练目标：理想权重应该与损失成反比
        with torch.no_grad():
            normalized_losses = (losses_input - losses_input.mean()) / (losses_input.std() + 1e-8)
            target_weights = torch.sigmoid(-normalized_losses)  # 损失高 -> 权重低
        
        # 计算权重网络损失
        weight_loss = F.mse_loss(predicted_weights, target_weights)
        
        # 更新权重网络
        self.weight_optimizer.zero_grad()
        weight_loss.backward()
        self.weight_optimizer.step()
    
    def get_sample_statistics(self) -> Dict:
        """获取样本统计信息"""
        stats = {}
        for sample_id, sample_stats in self.sample_tracker.sample_stats.items():
            stats[sample_id] = {
                'loss_ema': sample_stats['loss_ema'],
                'contribution': sample_stats['contribution'],
                'count': sample_stats['count']
            }
        return stats
    
    def reset_statistics(self):
        """重置统计信息"""
        self.sample_tracker.sample_stats.clear()
        self.step_count = 0


def create_adaptive_weighting_module(config: Dict) -> AdaptiveWeightingModule:
    """
    创建自适应加权模块的工厂函数
    
    Args:
        config: 配置字典，包含以下键值：
            - weight_net_hidden_dim: 权重网络隐藏层维度
            - ema_decay: 指数移动平均衰减因子
            - min_weight: 最小权重值
            - device: 计算设备
            - update_frequency: 更新频率
            
    Returns:
        AdaptiveWeightingModule实例
    """
    return AdaptiveWeightingModule(
        weight_net_hidden_dim=config.get('weight_net_hidden_dim', 64),
        ema_decay=config.get('ema_decay', 0.9),
        min_weight=config.get('min_weight', 0.1),
        device=config.get('device', 'cuda'),
        update_frequency=config.get('update_frequency', 100)
    )