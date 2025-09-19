# 自适应加权模块使用指南

## 概述

自适应加权模块是专为多组学数据去噪处理设计的模块，旨在缓解多源数据异质性噪声，提升模型鲁棒性。该模块通过动态学习样本权重，为持续提升模型性能的样本分配更高权重，为噪声样本分配较低权重。

## 核心功能

1. **权重学习的初始化**：初始化权重网络，用于学习每批数据中每个样本的权重
2. **权重调整机制**：基于样本对模型预测性能的贡献动态分配权重  
3. **权重更新频率**：在每个训练epoch后，根据样本贡献重新调整权重

## 快速开始

### 1. 基本使用

```python
from adaptive_weighting import create_adaptive_weighting_module
from k_model import ModelConfig, Transformer

# 创建启用自适应加权的模型配置
config = ModelConfig(
    dim=768,
    n_layers=12,
    n_heads=16,
    use_adaptive_weighting=True,  # 启用自适应加权
    adaptive_weighting_config={
        'weight_momentum': 0.9,     # 权重更新动量
        'min_weight': 0.1,          # 最小权重值  
        'max_weight': 1.0,          # 最大权重值
        'weight_update_freq': 1,    # 权重更新频率(每epoch)
        'dropout': 0.1              # dropout率
    }
)

# 创建模型
model = Transformer(config)

# 使用模型进行训练
# 模型会自动应用自适应加权
output = model(input_tokens, target_tokens)
loss = output.last_loss
```

### 2. 训练脚本使用

```bash
# 启用自适应加权的训练
python adaptive_training.py \
    --use_adaptive_weighting \
    --weight_momentum 0.9 \
    --min_weight 0.1 \
    --max_weight 1.0 \
    --weight_update_freq 1 \
    --epochs 10 \
    --batch_size 32
```

### 3. 独立使用自适应加权模块

```python
import torch
from adaptive_weighting import create_adaptive_weighting_module

# 创建自适应加权模块
weighting_module = create_adaptive_weighting_module(
    model_dim=768,
    weight_momentum=0.9,
    min_weight=0.1,
    max_weight=1.0
)

# 计算样本权重
batch_size = 32
loss_values = torch.randn(batch_size).abs()  # 模拟损失值
sample_weights = weighting_module.compute_sample_weights(loss_values)

# 计算加权损失
weighted_loss, weights = weighting_module.compute_weighted_loss(loss_values)

# 更新权重（在epoch结束时）
weighting_module.update_weights(epoch=1, sample_losses=loss_values)

# 获取权重统计信息
stats = weighting_module.get_weight_statistics()
print(f"权重统计: {stats}")
```

## 配置参数

### 模型配置参数

- `use_adaptive_weighting: bool`: 是否启用自适应加权模块，默认False
- `adaptive_weighting_config: dict`: 自适应加权模块的详细配置

### 自适应加权配置参数

- `weight_momentum: float`: 权重更新的动量系数，范围[0,1]，默认0.9
- `min_weight: float`: 最小权重值，防止完全忽略样本，默认0.1
- `max_weight: float`: 最大权重值，默认1.0
- `weight_update_freq: int`: 权重更新频率（每多少个epoch更新一次），默认1
- `dropout: float`: 权重网络的dropout率，默认0.1
- `activation: str`: 激活函数类型，可选'relu'、'tanh'、'sigmoid'，默认'relu'

### 权重网络配置参数

- `input_dim: int`: 输入维度，默认1（训练损失值）
- `hidden_dim: int`: 隐藏层维度，默认64
- `output_dim: int`: 输出维度，默认1（样本权重）

## 工作原理

### 1. 权重网络架构

```
输入层(损失值) -> 隐藏层1 -> 隐藏层2 -> 输出层(权重) -> Sigmoid激活
     1          ->   64   ->   32    ->    1    ->  [0,1]
```

### 2. 权重计算流程

1. **损失输入**: 将每个样本的训练损失作为输入
2. **权重预测**: 通过神经网络预测样本重要性权重
3. **范围约束**: 使用Sigmoid函数确保权重在[0,1]范围内
4. **动量更新**: 结合历史权重进行平滑更新
5. **加权损失**: 计算样本权重与损失的加权平均

### 3. 自适应机制

- **动态调整**: 根据样本对模型性能的贡献动态分配权重
- **噪声抑制**: 为噪声样本分配较低权重，减少其对训练的负面影响
- **性能增强**: 为高质量样本分配更高权重，增强模型学习效果

## 使用场景

### 1. 多组学数据处理
```python
# 适合处理来自不同来源的异质数据
config = ModelConfig(
    use_adaptive_weighting=True,
    adaptive_weighting_config={
        'weight_momentum': 0.95,  # 较高动量，稳定权重变化
        'min_weight': 0.05,       # 较低最小权重，更好过滤噪声
        'weight_update_freq': 1   # 每epoch更新，快速适应
    }
)
```

### 2. 噪声数据处理
```python
# 适合处理含有大量噪声的数据
config = ModelConfig(
    use_adaptive_weighting=True,
    adaptive_weighting_config={
        'weight_momentum': 0.8,   # 中等动量，快速响应
        'min_weight': 0.1,        # 保留一定权重避免过度抑制
        'max_weight': 1.0,
        'dropout': 0.2            # 较高dropout，增强泛化
    }
)
```

### 3. 不平衡数据处理
```python
# 适合处理样本质量差异较大的数据
config = ModelConfig(
    use_adaptive_weighting=True,
    adaptive_weighting_config={
        'weight_momentum': 0.9,
        'min_weight': 0.2,        # 较高最小权重，保护少数样本
        'max_weight': 0.9,        # 限制最大权重，避免过度偏向
        'weight_update_freq': 2   # 较低更新频率，稳定权重
    }
)
```

## 监控和调试

### 1. 权重统计监控

```python
# 在训练过程中监控权重分布
stats = model.get_adaptive_weight_statistics()
print(f"平均权重: {stats['mean_weight']:.3f}")
print(f"权重标准差: {stats['std_weight']:.3f}")
print(f"低权重样本数: {stats['num_low_weight_samples']}")
print(f"高权重样本数: {stats['num_high_weight_samples']}")
```

### 2. 使用SwanLab跟踪

```python
# 训练脚本会自动记录权重统计到SwanLab
python adaptive_training.py --use_swanlab --use_adaptive_weighting
```

### 3. 权重可视化

```python
import matplotlib.pyplot as plt

# 绘制权重分布
def plot_weight_distribution(model):
    if hasattr(model, 'last_sample_weights') and model.last_sample_weights is not None:
        weights = model.last_sample_weights.cpu().numpy()
        plt.hist(weights, bins=20, alpha=0.7)
        plt.xlabel('Sample Weight')
        plt.ylabel('Frequency')
        plt.title('Sample Weight Distribution')
        plt.show()
```

## 性能优化建议

### 1. 超参数调优

- **weight_momentum**: 
  - 高值(0.9-0.99): 适合稳定的数据分布
  - 低值(0.7-0.8): 适合快速变化的数据分布

- **min_weight/max_weight**:
  - 较大范围[0.1, 1.0]: 允许更大的权重差异
  - 较小范围[0.3, 0.8]: 保持权重相对平衡

- **weight_update_freq**:
  - 1: 每epoch更新，快速适应
  - 2-5: 较慢更新，更稳定的权重

### 2. 内存优化

```python
# 对于大批次，可以考虑梯度检查点
config = ModelConfig(
    use_adaptive_weighting=True,
    adaptive_weighting_config={
        'dropout': 0.0,  # 降低dropout减少计算
    }
)
```

### 3. 训练策略

- 前几个epoch关闭自适应加权，让模型先适应数据
- 逐步启用自适应加权，避免训练初期的不稳定

```python
# 动态控制自适应加权
for epoch in range(total_epochs):
    if epoch > warmup_epochs:
        model.set_adaptive_weighting(True)
    else:
        model.set_adaptive_weighting(False)
```

## 测试和验证

运行测试脚本验证功能：

```bash
python test_adaptive_weighting.py
```

测试内容包括：
- 权重网络基础功能测试
- 自适应加权模块完整性测试  
- 模型集成测试
- 权重动态调整机制测试

## 常见问题

### Q: 自适应加权会显著增加训练时间吗？
A: 权重网络很轻量（通常<1K参数），对训练时间影响很小（<5%）。

### Q: 如何判断自适应加权是否起作用？
A: 监控权重统计，查看low_weight_samples和high_weight_samples的数量变化。

### Q: 权重更新频率应该如何设置？
A: 建议从1开始，如果权重变化过于频繁导致训练不稳定，可以增加到2-3。

### Q: min_weight设置为0可以吗？
A: 不建议，设置过小的min_weight可能导致某些样本完全被忽略，影响模型的泛化能力。

## 扩展和自定义

### 1. 自定义权重网络

```python
class CustomWeightNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # 自定义网络结构
        
    def forward(self, loss_values):
        # 自定义前向传播
        return weights

# 使用自定义网络
weighting_module = AdaptiveWeightingModule(
    weight_network=CustomWeightNetwork()
)
```

### 2. 多输入权重网络

```python
# 支持多种输入特征的权重网络
config = {
    'input_dim': 3,  # 损失值 + 梯度范数 + 置信度
    'hidden_dim': 128,
}
```

## 参考文献

该实现基于自适应样本重加权的理论基础，主要参考了：

1. 元学习中的样本重要性权重学习
2. 噪声标签学习中的样本选择策略
3. 多任务学习中的任务权重平衡机制

通过这些方法的结合，实现了专门针对多组学数据去噪的自适应加权机制。