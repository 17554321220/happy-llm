"""
自适应加权模块 (Adaptive Weighting Module)
用于缓解多组学数据的异质性噪声并提升模型鲁棒性

该模块的核心目标是通过动态调整样本权重，降低噪声样本对模型训练的干扰，
同时强化高可靠性样本的作用，提高药物反应预测的精度与稳定性。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


class WeightNetwork(nn.Module):
    """权重网络，用于学习每批数据中每个样本的权重"""
    
    def __init__(self, input_dim: int = 1, hidden_dim: int = 64, output_dim: int = 1):
        super(WeightNetwork, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 定义权重网络结构：输入层 -> 隐藏层 -> 输出层
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        # 添加批归一化层提升训练稳定性
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # Dropout层防止过拟合
        self.dropout = nn.Dropout(p=0.2)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(self, loss_values: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            loss_values: 训练损失值 [batch_size, 1] 或 [batch_size]
        Returns:
            weights: 样本权重 [batch_size, 1]，范围 [0, 1]
        """
        if loss_values.dim() == 1:
            loss_values = loss_values.unsqueeze(1)
        
        # 输入层 -> ReLU激活
        x = F.relu(self.input_layer(loss_values))
        if x.size(0) > 1:  # 批归一化需要至少2个样本
            x = self.bn1(x)
        x = self.dropout(x)
        
        # 隐藏层 -> ReLU激活
        x = F.relu(self.hidden_layer(x))
        if x.size(0) > 1:
            x = self.bn2(x)
        x = self.dropout(x)
        
        # 输出层 -> Sigmoid激活，确保权重在[0, 1]范围内
        weights = torch.sigmoid(self.output_layer(x))
        
        return weights


class WeightedCrossEntropyLoss(nn.Module):
    """加权交叉熵损失函数"""
    
    def __init__(self, ignore_index: int = 0):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor, 
                weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算加权交叉熵损失
        Args:
            logits: 模型输出logits [batch_size * seq_len, vocab_size]
            targets: 目标标签 [batch_size * seq_len]
            weights: 样本权重 [batch_size, 1] 或 None
        Returns:
            weighted_loss: 加权后的损失值
        """
        # 计算基础交叉熵损失 (不进行reduction)
        loss = self.ce_loss(logits, targets)  # [batch_size * seq_len]
        
        if weights is not None:
            # 确保weights的维度匹配
            if weights.dim() == 2 and weights.size(1) == 1:
                weights = weights.squeeze(1)  # [batch_size]
            
            # 将权重扩展到与loss相同的维度
            seq_len = loss.size(0) // weights.size(0)
            expanded_weights = weights.repeat_interleave(seq_len)  # [batch_size * seq_len]
            
            # 应用权重
            weighted_loss = loss * expanded_weights
        else:
            weighted_loss = loss
        
        # 计算平均损失，忽略被ignore_index标记的位置
        valid_mask = (targets != self.ignore_index).float()
        if weights is not None:
            total_weight = (valid_mask * expanded_weights).sum()
        else:
            total_weight = valid_mask.sum()
        
        if total_weight > 0:
            final_loss = (weighted_loss * valid_mask).sum() / total_weight
        else:
            final_loss = weighted_loss.mean()
        
        return final_loss


class SampleContributionTracker:
    """样本贡献度追踪器"""
    
    def __init__(self, decay_factor: float = 0.9):
        self.decay_factor = decay_factor
        self.sample_losses = defaultdict(list)  # 存储每个样本的历史损失
        self.sample_improvements = defaultdict(list)  # 存储每个样本的改进情况
        self.global_loss_history = []  # 全局损失历史
        
    def update_sample_loss(self, sample_ids: List[int], losses: torch.Tensor):
        """更新样本损失记录"""
        losses_np = losses.detach().cpu().numpy()
        
        for sample_id, loss in zip(sample_ids, losses_np):
            self.sample_losses[sample_id].append(loss)
            # 保持历史记录在合理范围内
            if len(self.sample_losses[sample_id]) > 100:
                self.sample_losses[sample_id] = self.sample_losses[sample_id][-100:]
    
    def calculate_sample_contribution(self, sample_ids: List[int]) -> torch.Tensor:
        """
        计算样本对模型性能的贡献度
        贡献度基于样本损失的稳定性和改进趋势
        """
        contributions = []
        
        for sample_id in sample_ids:
            if sample_id not in self.sample_losses or len(self.sample_losses[sample_id]) < 2:
                # 新样本或历史数据不足，给予中等权重
                contributions.append(0.5)
                continue
            
            losses = self.sample_losses[sample_id]
            
            # 计算损失趋势（下降趋势表示良好的学习效果）
            if len(losses) >= 5:
                recent_losses = losses[-5:]
                early_losses = losses[-10:-5] if len(losses) >= 10 else losses[:-5]
                
                recent_avg = np.mean(recent_losses)
                early_avg = np.mean(early_losses) if early_losses else recent_avg
                
                # 损失下降表示样本有助于学习
                improvement = max(0, (early_avg - recent_avg) / (early_avg + 1e-8))
            else:
                improvement = 0.0
            
            # 计算损失稳定性（低方差表示稳定的样本）
            loss_std = np.std(losses[-10:]) if len(losses) >= 10 else np.std(losses)
            loss_mean = np.mean(losses[-10:]) if len(losses) >= 10 else np.mean(losses)
            stability = 1.0 / (1.0 + loss_std / (loss_mean + 1e-8))
            
            # 综合计算贡献度
            contribution = 0.6 * stability + 0.4 * improvement
            contribution = np.clip(contribution, 0.1, 1.0)  # 确保权重在合理范围内
            
            contributions.append(contribution)
        
        return torch.tensor(contributions, dtype=torch.float32)


class AdaptiveWeightingModule(nn.Module):
    """自适应加权模块主类"""
    
    def __init__(self, 
                 weight_network_hidden_dim: int = 64,
                 update_frequency: int = 1,
                 min_weight: float = 0.1,
                 max_weight: float = 1.0,
                 device: torch.device = None):
        super(AdaptiveWeightingModule, self).__init__()
        
        self.weight_network = WeightNetwork(
            input_dim=1,
            hidden_dim=weight_network_hidden_dim,
            output_dim=1
        )
        
        self.weighted_loss_fn = WeightedCrossEntropyLoss()
        self.contribution_tracker = SampleContributionTracker()
        
        self.update_frequency = update_frequency
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.epoch_count = 0
        self.current_weights = None
        
        self.to(self.device)
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor, 
                individual_losses: torch.Tensor,
                sample_ids: Optional[List[int]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播计算加权损失
        Args:
            logits: 模型输出 [batch_size * seq_len, vocab_size]
            targets: 目标标签 [batch_size * seq_len]  
            individual_losses: 每个样本的个体损失 [batch_size]
            sample_ids: 样本ID列表，用于追踪样本贡献度
        Returns:
            weighted_loss: 加权损失
            sample_weights: 计算出的样本权重
        """
        batch_size = individual_losses.size(0)
        
        if sample_ids is None:
            sample_ids = list(range(batch_size))
        
        # 更新样本贡献度追踪
        self.contribution_tracker.update_sample_loss(sample_ids, individual_losses)
        
        # 计算基于贡献度的权重
        contribution_weights = self.contribution_tracker.calculate_sample_contribution(sample_ids)
        
        # 使用权重网络学习额外的权重调整
        network_weights = self.weight_network(individual_losses.unsqueeze(1)).squeeze(1)
        
        # 结合贡献度权重和网络学习的权重
        combined_weights = 0.7 * contribution_weights.to(self.device) + 0.3 * network_weights
        
        # 确保权重在指定范围内
        final_weights = torch.clamp(combined_weights, self.min_weight, self.max_weight)
        
        # 计算加权损失
        weighted_loss = self.weighted_loss_fn(logits, targets, final_weights.unsqueeze(1))
        
        self.current_weights = final_weights
        
        return weighted_loss, final_weights
    
    def update_epoch(self):
        """每个epoch后更新计数"""
        self.epoch_count += 1
    
    def get_weight_statistics(self) -> Dict[str, float]:
        """获取权重统计信息"""
        if self.current_weights is None:
            return {}
        
        weights_np = self.current_weights.detach().cpu().numpy()
        return {
            'mean_weight': float(np.mean(weights_np)),
            'std_weight': float(np.std(weights_np)),
            'min_weight': float(np.min(weights_np)),
            'max_weight': float(np.max(weights_np)),
            'num_low_weights': int(np.sum(weights_np < 0.3)),
            'num_high_weights': int(np.sum(weights_np > 0.7))
        }


class FederatedWeightingModule(AdaptiveWeightingModule):
    """联邦学习加权模块，支持加密加权处理"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encrypted_losses = {}  # 存储各参与方的加密加权损失
        
    def compute_encrypted_weighted_loss(self, 
                                      participant_id: str,
                                      logits: torch.Tensor, 
                                      targets: torch.Tensor,
                                      individual_losses: torch.Tensor,
                                      encryption_fn=None) -> torch.Tensor:
        """
        计算加密加权损失用于联邦学习
        Args:
            participant_id: 参与方ID
            logits: 模型输出
            targets: 目标标签
            individual_losses: 个体损失
            encryption_fn: 加密函数（实际应用中会使用同态加密）
        Returns:
            encrypted_weighted_loss: 加密的加权损失
        """
        # 计算加权损失
        weighted_loss, weights = self.forward(logits, targets, individual_losses)
        
        # 在实际应用中，这里会使用同态加密
        # 这里用简单的示例代替真实的加密过程
        if encryption_fn is not None:
            encrypted_loss = encryption_fn(weighted_loss)
        else:
            # 简单的模拟加密（实际应用中不应该这样做）
            encrypted_loss = weighted_loss * torch.randn(1, device=self.device) + weighted_loss
        
        self.encrypted_losses[participant_id] = encrypted_loss
        
        return encrypted_loss
    
    def aggregate_encrypted_losses(self, encryption_scheme=None) -> torch.Tensor:
        """
        中心服务器聚合各参与方的加密加权损失
        Args:
            encryption_scheme: 加密方案
        Returns:
            aggregated_loss: 聚合后的损失
        """
        if not self.encrypted_losses:
            raise ValueError("No encrypted losses to aggregate")
        
        # 简单的聚合示例（实际应用中需要使用同态加密的聚合操作）
        total_loss = sum(self.encrypted_losses.values())
        avg_loss = total_loss / len(self.encrypted_losses)
        
        # 清空缓存
        self.encrypted_losses.clear()
        
        return avg_loss


def create_adaptive_weighting_module(config: Dict) -> AdaptiveWeightingModule:
    """
    创建自适应加权模块的工厂函数
    Args:
        config: 配置字典，包含模块参数
    Returns:
        AdaptiveWeightingModule实例
    """
    return AdaptiveWeightingModule(
        weight_network_hidden_dim=config.get('hidden_dim', 64),
        update_frequency=config.get('update_frequency', 1),
        min_weight=config.get('min_weight', 0.1),
        max_weight=config.get('max_weight', 1.0),
        device=torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    )


# 示例用法和测试代码
if __name__ == "__main__":
    # 测试自适应加权模块
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模块
    weighting_module = AdaptiveWeightingModule(device=device)
    
    # 模拟数据
    batch_size = 8
    seq_len = 128
    vocab_size = 1000
    
    logits = torch.randn(batch_size * seq_len, vocab_size, device=device)
    targets = torch.randint(0, vocab_size, (batch_size * seq_len,), device=device)
    individual_losses = torch.rand(batch_size, device=device)
    
    # 前向传播
    weighted_loss, weights = weighting_module(logits, targets, individual_losses)
    
    print(f"Weighted Loss: {weighted_loss.item():.4f}")
    print(f"Sample Weights: {weights.tolist()}")
    print(f"Weight Statistics: {weighting_module.get_weight_statistics()}")
    
    # 测试联邦学习模块
    fed_module = FederatedWeightingModule(device=device)
    
    # 模拟多个参与方
    for i in range(3):
        participant_id = f"participant_{i}"
        encrypted_loss = fed_module.compute_encrypted_weighted_loss(
            participant_id, logits, targets, individual_losses
        )
        print(f"{participant_id} encrypted loss: {encrypted_loss.item():.4f}")
    
    # 聚合损失
    aggregated_loss = fed_module.aggregate_encrypted_losses()
    print(f"Aggregated loss: {aggregated_loss.item():.4f}")