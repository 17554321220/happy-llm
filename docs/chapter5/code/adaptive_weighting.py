"""
自适应加权模块 (Adaptive Weighting Module)
用于多组学数据的去噪处理，缓解多源数据异质性噪声，提升模型鲁棒性

核心功能:
1. 权重学习的初始化：初始化权重网络，用于学习每批数据中每个样本的权重
2. 权重调整机制：基于样本对模型预测性能的贡献动态分配权重  
3. 权重更新频率：在每个训练epoch后，根据样本贡献重新调整权重
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class AdaptiveWeightNetwork(nn.Module):
    """
    自适应权重网络
    
    根据训练损失值学习每个样本的重要性权重，权重范围为[0,1]
    持续提升模型性能的样本赋予更高权重，噪声样本赋予较低权重
    """
    
    def __init__(
        self, 
        input_dim: int = 1, 
        hidden_dim: int = 64, 
        output_dim: int = 1,
        dropout: float = 0.1,
        activation: str = 'relu'
    ):
        """
        初始化权重网络
        
        Args:
            input_dim: 输入维度，默认为1 (训练损失值)
            hidden_dim: 隐藏层维度
            output_dim: 输出维度，默认为1 (样本权重)
            dropout: dropout率，用于防止过拟合
            activation: 激活函数类型 ('relu', 'tanh', 'sigmoid')
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout
        
        # 定义权重网络结构
        # 输入层 -> 隐藏层
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        
        # 隐藏层 -> 隐藏层 (可扩展多层)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2)
        ])
        
        # 输出层
        self.output_layer = nn.Linear(hidden_dim // 2, output_dim)
        
        # Dropout层
        self.dropout_layer = nn.Dropout(dropout)
        
        # 选择激活函数
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for layer in [self.input_layer] + list(self.hidden_layers) + [self.output_layer]:
            if isinstance(layer, nn.Linear):
                # 使用Xavier初始化
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def forward(self, loss_values: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            loss_values: 训练损失值张量，形状为 [batch_size, 1] 或 [batch_size]
            
        Returns:
            weights: 样本权重张量，形状为 [batch_size, 1]，权重范围为[0, 1]
        """
        # 确保输入形状正确
        if loss_values.dim() == 1:
            loss_values = loss_values.unsqueeze(-1)
        
        # 输入层
        x = self.input_layer(loss_values)
        x = self.activation(x)
        x = self.dropout_layer(x)
        
        # 隐藏层
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
            x = self.activation(x)
            x = self.dropout_layer(x)
        
        # 输出层，使用sigmoid确保输出在[0,1]范围内
        weights = torch.sigmoid(self.output_layer(x))
        
        return weights


class AdaptiveWeightingModule(nn.Module):
    """
    自适应加权模块
    
    集成权重网络和加权损失计算，提供完整的自适应加权解决方案
    """
    
    def __init__(
        self,
        weight_network_config: Optional[dict] = None,
        weight_momentum: float = 0.9,
        min_weight: float = 0.1,
        max_weight: float = 1.0,
        weight_update_freq: int = 1
    ):
        """
        初始化自适应加权模块
        
        Args:
            weight_network_config: 权重网络配置参数
            weight_momentum: 权重更新的动量系数
            min_weight: 最小权重值，防止完全忽略样本
            max_weight: 最大权重值
            weight_update_freq: 权重更新频率(每多少个epoch更新一次)
        """
        super().__init__()
        
        # 权重网络配置
        if weight_network_config is None:
            weight_network_config = {
                'input_dim': 1,
                'hidden_dim': 64,
                'output_dim': 1,
                'dropout': 0.1,
                'activation': 'relu'
            }
        
        # 初始化权重网络
        self.weight_network = AdaptiveWeightNetwork(**weight_network_config)
        
        # 超参数
        self.weight_momentum = weight_momentum
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.weight_update_freq = weight_update_freq
        
        # 历史权重记录
        self.register_buffer('historical_weights', None)
        self.register_buffer('sample_performance_history', None)
        
        # 统计信息
        self.epoch_count = 0
        
    def compute_sample_weights(self, loss_values: torch.Tensor) -> torch.Tensor:
        """
        计算样本权重
        
        Args:
            loss_values: 每个样本的损失值 [batch_size]
            
        Returns:
            sample_weights: 样本权重 [batch_size]
        """
        # 通过权重网络计算原始权重
        raw_weights = self.weight_network(loss_values).squeeze(-1)
        
        # 应用权重范围约束
        sample_weights = torch.clamp(raw_weights, self.min_weight, self.max_weight)
        
        # 如果有历史权重，应用动量更新
        if self.historical_weights is not None and self.historical_weights.size(0) == sample_weights.size(0):
            sample_weights = (self.weight_momentum * self.historical_weights + 
                            (1 - self.weight_momentum) * sample_weights)
        
        return sample_weights
    
    def compute_weighted_loss(
        self, 
        loss_values: torch.Tensor, 
        weights: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算加权损失
        
        Args:
            loss_values: 每个样本的损失值 [batch_size]
            weights: 可选的预计算权重 [batch_size]
            
        Returns:
            weighted_loss: 加权后的总损失 (标量)
            sample_weights: 使用的样本权重 [batch_size]
        """
        if weights is None:
            # 计算样本权重
            weights = self.compute_sample_weights(loss_values)
        
        # 计算加权损失
        weighted_losses = loss_values * weights
        
        # 归一化，避免权重影响总体损失规模
        weight_sum = torch.sum(weights)
        if weight_sum > 0:
            weighted_loss = torch.sum(weighted_losses) / weight_sum
        else:
            weighted_loss = torch.mean(loss_values)
        
        return weighted_loss, weights
    
    def update_weights(self, epoch: int, sample_losses: Optional[torch.Tensor] = None):
        """
        更新权重（在epoch结束时调用）
        
        Args:
            epoch: 当前epoch数
            sample_losses: 当前epoch的样本损失
        """
        self.epoch_count = epoch
        
        # 只在指定频率更新权重
        if epoch % self.weight_update_freq == 0 and sample_losses is not None:
            with torch.no_grad():
                # 计算新权重
                new_weights = self.compute_sample_weights(sample_losses)
                
                # 更新历史权重
                self.historical_weights = new_weights.clone()
                
    def get_weight_statistics(self) -> dict:
        """获取权重统计信息"""
        if self.historical_weights is None:
            return {}
        
        weights = self.historical_weights
        return {
            'mean_weight': weights.mean().item(),
            'std_weight': weights.std().item(),
            'min_weight': weights.min().item(),
            'max_weight': weights.max().item(),
            'num_low_weight_samples': (weights < 0.5).sum().item(),
            'num_high_weight_samples': (weights > 0.8).sum().item()
        }


def create_adaptive_weighting_module(
    model_dim: int = 768,
    **kwargs
) -> AdaptiveWeightingModule:
    """
    创建适用于特定模型的自适应加权模块的工厂函数
    
    Args:
        model_dim: 模型维度，用于自动配置权重网络
        **kwargs: 其他配置参数
        
    Returns:
        AdaptiveWeightingModule: 配置好的自适应加权模块
    """
    # 根据模型维度自动调整权重网络配置
    hidden_dim = min(64, model_dim // 12)  # 自适应隐藏层大小
    
    weight_network_config = {
        'input_dim': 1,
        'hidden_dim': hidden_dim,
        'output_dim': 1,
        'dropout': kwargs.get('dropout', 0.1),
        'activation': kwargs.get('activation', 'relu')
    }
    
    return AdaptiveWeightingModule(
        weight_network_config=weight_network_config,
        weight_momentum=kwargs.get('weight_momentum', 0.9),
        min_weight=kwargs.get('min_weight', 0.1),
        max_weight=kwargs.get('max_weight', 1.0),
        weight_update_freq=kwargs.get('weight_update_freq', 1)
    )


# 测试代码
if __name__ == "__main__":
    # 创建自适应加权模块
    weighting_module = create_adaptive_weighting_module()
    
    # 模拟损失值
    batch_size = 32
    loss_values = torch.randn(batch_size).abs()  # 模拟正损失值
    
    # 计算样本权重
    sample_weights = weighting_module.compute_sample_weights(loss_values)
    print(f"Sample weights shape: {sample_weights.shape}")
    print(f"Sample weights range: [{sample_weights.min():.3f}, {sample_weights.max():.3f}]")
    
    # 计算加权损失
    weighted_loss, weights = weighting_module.compute_weighted_loss(loss_values)
    print(f"Original loss mean: {loss_values.mean():.3f}")
    print(f"Weighted loss: {weighted_loss:.3f}")
    
    # 权重统计
    weighting_module.update_weights(1, loss_values)
    stats = weighting_module.get_weight_statistics()
    print(f"Weight statistics: {stats}")