# 自适应加权模块 (Adaptive Weighting Module)

用于缓解多组学数据异质性噪声并提升模型鲁棒性的深度学习模块。

## 概述

自适应加权模块通过动态调整样本权重，降低噪声样本对模型训练的干扰，同时强化高可靠性样本的作用，提高药物反应预测的精度与稳定性。该模块特别适用于多组学数据分析和联邦学习环境。

## 核心功能

### 1. 权重学习的初始化
- **权重网络结构**: 输入层 → 隐藏层 → 输出层
- **输入**: 训练损失值 
- **输出**: 样本权重 [0,1]
- **网络架构**: Linear(1, 64) → ReLU → Linear(64, 1) → Sigmoid

### 2. 权重调整机制
- **贡献度评估**: 基于样本对模型预测性能的历史贡献
- **动态分配**: 高贡献样本获得更高权重，噪声样本权重降低
- **综合计算**: 0.6 × 稳定性 + 0.4 × 改进趋势

### 3. 权重更新频率
- **Epoch级更新**: 每个训练epoch后重新调整权重
- **历史追踪**: 维护样本损失的滑动窗口历史
- **实时统计**: 提供权重分布的实时监控

### 4. 联邦学习支持
- **本地加权**: 各参与方独立计算加权损失
- **加密聚合**: 支持同态加密的损失聚合
- **隐私保护**: 确保数据本地化，不泄露原始数据

## 文件结构

```
adaptive_weighting.py           # 主要模块实现
ddp_sft_weighted.py            # 集成自适应加权的训练脚本  
demo_adaptive_weighting.py     # 无依赖演示脚本
test_adaptive_weighting.py     # 单元测试文件
README.md                      # 本文档
```

## 核心组件

### WeightNetwork
权重网络，学习样本权重映射关系
```python
class WeightNetwork(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=1)
    def forward(self, loss_values: torch.Tensor) -> torch.Tensor
```

### WeightedCrossEntropyLoss
加权交叉熵损失函数
```python  
class WeightedCrossEntropyLoss(nn.Module):
    def forward(self, logits, targets, weights=None) -> torch.Tensor
```

### SampleContributionTracker
样本贡献度追踪器
```python
class SampleContributionTracker:
    def update_sample_loss(self, sample_ids: List[int], losses: torch.Tensor)
    def calculate_sample_contribution(self, sample_ids: List[int]) -> torch.Tensor
```

### AdaptiveWeightingModule
自适应加权模块主类
```python
class AdaptiveWeightingModule(nn.Module):
    def forward(self, logits, targets, individual_losses, sample_ids=None) -> Tuple[torch.Tensor, torch.Tensor]
    def get_weight_statistics(self) -> Dict[str, float]
```

### FederatedWeightingModule
联邦学习加权模块
```python
class FederatedWeightingModule(AdaptiveWeightingModule):
    def compute_encrypted_weighted_loss(self, participant_id, logits, targets, individual_losses) -> torch.Tensor
    def aggregate_encrypted_losses(self) -> torch.Tensor
```

## 使用方法

### 基本使用

```python
from adaptive_weighting import AdaptiveWeightingModule

# 创建模块
weighting_module = AdaptiveWeightingModule(
    weight_network_hidden_dim=64,
    min_weight=0.1,
    max_weight=1.0,
    device=device
)

# 训练中使用
logits = model(X, Y).logits.view(-1, vocab_size)
targets = Y.view(-1)
individual_losses = compute_individual_losses(X, Y)

weighted_loss, sample_weights = weighting_module(
    logits, targets, individual_losses, sample_ids
)

# 使用加权损失进行反向传播
weighted_loss.backward()
```

### 集成训练脚本

```bash
# 使用自适应加权训练
python ddp_sft_weighted.py \
    --use_adaptive_weighting \
    --weight_hidden_dim 64 \
    --min_weight 0.1 \
    --max_weight 1.0 \
    --batch_size 8 \
    --epochs 3

# 常规训练（对比基准）
python ddp_sft_weighted.py \
    --batch_size 8 \
    --epochs 3
```

### 联邦学习场景

```python
from adaptive_weighting import FederatedWeightingModule

fed_module = FederatedWeightingModule(device=device)

# 各参与方计算本地加权损失
for participant_id in participants:
    encrypted_loss = fed_module.compute_encrypted_weighted_loss(
        participant_id, logits, targets, individual_losses
    )

# 中心服务器聚合
aggregated_loss = fed_module.aggregate_encrypted_losses()
```

## 算法原理

### 权重计算公式

1. **损失趋势**: `improvement = max(0, (early_avg - recent_avg) / early_avg)`
2. **损失稳定性**: `stability = 1 / (1 + std_dev / (mean_loss + ε))`
3. **贡献度**: `contribution = 0.6 × stability + 0.4 × improvement`
4. **网络权重**: `network_weight = sigmoid(MLP(individual_loss))`
5. **最终权重**: `final_weight = 0.7 × contribution + 0.3 × network_weight`

### 加权损失

```
weighted_loss = Σ(loss[i] × weight[i]) / Σweight[i]
```

## 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `weight_network_hidden_dim` | 64 | 权重网络隐藏层维度 |
| `min_weight` | 0.1 | 最小样本权重 |
| `max_weight` | 1.0 | 最大样本权重 |
| `update_frequency` | 1 | 权重更新频率(epoch) |
| `decay_factor` | 0.9 | 历史损失衰减因子 |

## 演示运行

运行无依赖演示：
```bash
python demo_adaptive_weighting.py
```

该演示展示了：
- 不同类型样本的权重变化
- 联邦学习场景模拟
- 算法实现细节说明

## 性能特点

### 优势
- ✅ **提升鲁棒性**: 降低噪声样本影响
- ✅ **增强稳定性**: 基于历史表现动态调整
- ✅ **保护隐私**: 支持联邦学习框架
- ✅ **实时监控**: 提供详细权重统计
- ✅ **易于集成**: 模块化设计，便于集成

### 适用场景
- 🧬 多组学数据分析
- 💊 药物反应预测
- 🏥 医疗数据联邦学习
- 🔍 含噪声标签的监督学习
- 🚀 提升模型鲁棒性的任何场景

## 实验效果

通过演示可以观察到：
- 好样本（损失稳定下降）获得较高权重
- 噪声样本（损失波动大）权重被降低
- 权重分配随训练动态优化
- 加权损失相比常规损失更稳定

## 技术特色

### 1. 智能权重分配
基于样本历史表现和实时损失，智能分配训练权重

### 2. 双重权重机制  
结合贡献度分析和神经网络学习的权重

### 3. 联邦学习友好
天然支持分布式训练和隐私保护

### 4. 可扩展架构
模块化设计，支持自定义权重策略

## 引用

如果您在研究中使用了此模块，请引用：

```bibtex
@software{adaptive_weighting_2024,
  title={Adaptive Weighting Module for Multi-omics Data Heterogeneity},
  author={Happy-LLM Contributors},
  year={2024},
  url={https://github.com/17554321220/happy-llm}
}
```

## 许可证

此模块遵循与主项目相同的许可证。详情请参见 [LICENSE](../../LICENSE.txt) 文件。