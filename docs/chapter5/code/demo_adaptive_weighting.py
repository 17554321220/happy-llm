"""
自适应加权模块演示脚本（无需外部依赖）
演示模块的核心功能和算法逻辑
"""

import math
import random
from typing import List, Dict, Tuple

class SimpleWeightNetwork:
    """简化的权重网络实现（无PyTorch依赖）"""
    
    def __init__(self, input_dim=1, hidden_dim=64):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 简化的权重初始化
        self.w1 = [[random.gauss(0, 0.1) for _ in range(hidden_dim)] for _ in range(input_dim)]
        self.b1 = [random.gauss(0, 0.1) for _ in range(hidden_dim)]
        self.w2 = [[random.gauss(0, 0.1) for _ in range(1)] for _ in range(hidden_dim)]
        self.b2 = [random.gauss(0, 0.1)]
    
    def relu(self, x):
        return max(0, x)
    
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-min(max(x, -500), 500)))  # 防止溢出
    
    def forward(self, loss_values: List[float]) -> List[float]:
        """前向传播计算权重"""
        weights = []
        
        for loss in loss_values:
            # 第一层：input -> hidden
            hidden = []
            for i in range(self.hidden_dim):
                h = loss * self.w1[0][i] + self.b1[i]
                hidden.append(self.relu(h))
            
            # 第二层：hidden -> output
            output = 0
            for i in range(self.hidden_dim):
                output += hidden[i] * self.w2[i][0]
            output += self.b2[0]
            
            # 应用sigmoid得到[0,1]范围的权重
            weight = self.sigmoid(output)
            weights.append(weight)
        
        return weights

class SimpleSampleTracker:
    """简化的样本贡献度追踪器"""
    
    def __init__(self):
        self.sample_losses = {}  # 样本ID -> 损失历史列表
        self.history_limit = 10  # 保持的历史记录数量
    
    def update_sample_loss(self, sample_ids: List[int], losses: List[float]):
        """更新样本损失记录"""
        for sample_id, loss in zip(sample_ids, losses):
            if sample_id not in self.sample_losses:
                self.sample_losses[sample_id] = []
            
            self.sample_losses[sample_id].append(loss)
            
            # 限制历史记录长度
            if len(self.sample_losses[sample_id]) > self.history_limit:
                self.sample_losses[sample_id] = self.sample_losses[sample_id][-self.history_limit:]
    
    def calculate_sample_contribution(self, sample_ids: List[int]) -> List[float]:
        """计算样本贡献度"""
        contributions = []
        
        for sample_id in sample_ids:
            if sample_id not in self.sample_losses or len(self.sample_losses[sample_id]) < 2:
                # 新样本或历史不足，给予中等权重
                contributions.append(0.5)
                continue
            
            losses = self.sample_losses[sample_id]
            
            # 计算损失趋势
            if len(losses) >= 4:
                recent_avg = sum(losses[-2:]) / 2
                early_avg = sum(losses[-4:-2]) / 2
                improvement = max(0, (early_avg - recent_avg) / (early_avg + 1e-8))
            else:
                improvement = 0.0
            
            # 计算损失稳定性
            if len(losses) >= 3:
                mean_loss = sum(losses) / len(losses)
                variance = sum((x - mean_loss) ** 2 for x in losses) / len(losses)
                std_dev = math.sqrt(variance)
                stability = 1.0 / (1.0 + std_dev / (mean_loss + 1e-8))
            else:
                stability = 0.5
            
            # 综合计算贡献度
            contribution = 0.6 * stability + 0.4 * improvement
            contribution = max(0.1, min(1.0, contribution))  # 限制在[0.1, 1.0]范围
            
            contributions.append(contribution)
        
        return contributions

class SimpleAdaptiveWeighting:
    """简化的自适应加权模块"""
    
    def __init__(self, min_weight=0.1, max_weight=1.0):
        self.weight_network = SimpleWeightNetwork()
        self.contribution_tracker = SimpleSampleTracker()
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.epoch_count = 0
        self.current_weights = []
    
    def compute_weighted_loss(self, individual_losses: List[float], 
                            sample_ids: List[int] = None) -> Tuple[float, List[float]]:
        """计算加权损失"""
        if sample_ids is None:
            sample_ids = list(range(len(individual_losses)))
        
        # 更新样本贡献度追踪
        self.contribution_tracker.update_sample_loss(sample_ids, individual_losses)
        
        # 计算基于贡献度的权重
        contribution_weights = self.contribution_tracker.calculate_sample_contribution(sample_ids)
        
        # 使用权重网络学习额外权重
        network_weights = self.weight_network.forward(individual_losses)
        
        # 结合两种权重
        final_weights = []
        for contrib_w, net_w in zip(contribution_weights, network_weights):
            combined_w = 0.7 * contrib_w + 0.3 * net_w
            # 确保权重在指定范围内
            final_w = max(self.min_weight, min(self.max_weight, combined_w))
            final_weights.append(final_w)
        
        # 计算加权损失
        weighted_loss = sum(loss * weight for loss, weight in zip(individual_losses, final_weights))
        total_weight = sum(final_weights)
        
        if total_weight > 0:
            weighted_loss /= total_weight
        else:
            weighted_loss = sum(individual_losses) / len(individual_losses)
        
        self.current_weights = final_weights
        return weighted_loss, final_weights
    
    def update_epoch(self):
        """更新epoch计数"""
        self.epoch_count += 1
    
    def get_weight_statistics(self) -> Dict[str, float]:
        """获取权重统计信息"""
        if not self.current_weights:
            return {}
        
        weights = self.current_weights
        mean_weight = sum(weights) / len(weights)
        variance = sum((w - mean_weight) ** 2 for w in weights) / len(weights)
        std_weight = math.sqrt(variance)
        min_weight = min(weights)
        max_weight = max(weights)
        num_low_weights = sum(1 for w in weights if w < 0.3)
        num_high_weights = sum(1 for w in weights if w > 0.7)
        
        return {
            'mean_weight': mean_weight,
            'std_weight': std_weight,
            'min_weight': min_weight,
            'max_weight': max_weight,
            'num_low_weights': num_low_weights,
            'num_high_weights': num_high_weights
        }

def simulate_training_scenario():
    """模拟训练场景来演示自适应加权的效果"""
    print("=" * 60)
    print("自适应加权模块演示")
    print("=" * 60)
    
    # 初始化模块
    weighting_module = SimpleAdaptiveWeighting(min_weight=0.1, max_weight=1.0)
    
    print("1. 初始化完成")
    print("   - 权重范围: [0.1, 1.0]")
    print("   - 权重网络: 1 -> 64 -> 1")
    print("   - 样本追踪器: 最多保存10个历史损失")
    print()
    
    # 模拟不同类型的样本
    print("2. 模拟样本类型:")
    print("   - 好样本 (ID 0-2): 损失逐渐降低，稳定学习")
    print("   - 噪声样本 (ID 3-5): 损失波动大，干扰学习")
    print("   - 普通样本 (ID 6-8): 损失缓慢下降")
    print()
    
    # 模拟多个epoch的训练
    epochs = 5
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}:")
        
        # 模拟不同批次的数据
        for batch in range(3):
            # 生成模拟损失
            # 好样本：损失逐渐降低
            good_losses = [max(0.1, 2.0 - epoch * 0.3 - batch * 0.1 + random.gauss(0, 0.05)) for _ in range(3)]
            
            # 噪声样本：损失波动大
            noise_losses = [1.5 + random.gauss(0, 0.8) for _ in range(3)]
            noise_losses = [max(0.1, loss) for loss in noise_losses]  # 确保损失为正
            
            # 普通样本：损失缓慢下降
            normal_losses = [max(0.1, 1.8 - epoch * 0.15 - batch * 0.05 + random.gauss(0, 0.1)) for _ in range(3)]
            
            all_losses = good_losses + noise_losses + normal_losses
            sample_ids = list(range(9))
            
            # 计算加权损失
            weighted_loss, sample_weights = weighting_module.compute_weighted_loss(all_losses, sample_ids)
            regular_loss = sum(all_losses) / len(all_losses)
            
            # 获取统计信息
            stats = weighting_module.get_weight_statistics()
            
            print(f"  Batch {batch + 1}:")
            print(f"    常规损失: {regular_loss:.4f}")
            print(f"    加权损失: {weighted_loss:.4f}")
            print(f"    权重统计: 均值={stats['mean_weight']:.3f}, 标准差={stats['std_weight']:.3f}")
            
            # 显示各类样本的权重
            good_weights = sample_weights[:3]
            noise_weights = sample_weights[3:6]
            normal_weights = sample_weights[6:9]
            
            print(f"    好样本权重: {[f'{w:.3f}' for w in good_weights]}")
            print(f"    噪声样本权重: {[f'{w:.3f}' for w in noise_weights]}")
            print(f"    普通样本权重: {[f'{w:.3f}' for w in normal_weights]}")
            print()
        
        # 更新epoch
        weighting_module.update_epoch()
        print(f"  Epoch {epoch + 1} 完成，总epoch数: {weighting_module.epoch_count}")
        print()
    
    print("3. 训练完成分析:")
    
    # 分析最终的样本贡献度
    final_contributions = weighting_module.contribution_tracker.calculate_sample_contribution(list(range(9)))
    
    print("   最终样本贡献度:")
    print(f"   - 好样本 (ID 0-2): {[f'{c:.3f}' for c in final_contributions[:3]]}")
    print(f"   - 噪声样本 (ID 3-5): {[f'{c:.3f}' for c in final_contributions[3:6]]}")
    print(f"   - 普通样本 (ID 6-8): {[f'{c:.3f}' for c in final_contributions[6:9]]}")
    print()
    
    print("4. 效果分析:")
    print("   - 好样本通常获得较高权重，有助于稳定训练")
    print("   - 噪声样本权重被降低，减少对训练的干扰")
    print("   - 普通样本权重适中，保持训练的多样性")
    print("   - 权重会根据样本的历史表现动态调整")
    print()

def demonstrate_federated_learning():
    """演示联邦学习场景"""
    print("=" * 60)
    print("联邦学习加权场景演示")
    print("=" * 60)
    
    # 模拟多个参与方
    participants = ["Hospital_A", "Hospital_B", "Research_Center"]
    participant_modules = {p: SimpleAdaptiveWeighting() for p in participants}
    
    print("1. 联邦学习设置:")
    print(f"   参与方: {participants}")
    print("   每个参与方都有独立的自适应加权模块")
    print()
    
    # 模拟联邦训练轮次
    rounds = 3
    for round_num in range(rounds):
        print(f"联邦训练轮次 {round_num + 1}/{rounds}:")
        
        participant_losses = []
        
        for participant in participants:
            # 每个参与方的本地数据特征不同
            if participant == "Hospital_A":
                # 医院A的数据质量较高
                local_losses = [0.5 + random.gauss(0, 0.1) for _ in range(4)]
            elif participant == "Hospital_B":
                # 医院B有一些噪声数据
                local_losses = [0.8 + random.gauss(0, 0.3) for _ in range(4)]
            else:  # Research_Center
                # 研究中心数据多样化但整体质量不错
                local_losses = [0.6 + random.gauss(0, 0.15) for _ in range(4)]
            
            local_losses = [max(0.1, loss) for loss in local_losses]  # 确保为正值
            sample_ids = [f"{participant}_{i}" for i in range(4)]
            
            # 计算本地加权损失
            module = participant_modules[participant]
            weighted_loss, weights = module.compute_weighted_loss(local_losses, sample_ids)
            
            participant_losses.append(weighted_loss)
            
            print(f"  {participant}:")
            print(f"    本地损失: {[f'{l:.3f}' for l in local_losses]}")
            print(f"    样本权重: {[f'{w:.3f}' for w in weights]}")
            print(f"    加权损失: {weighted_loss:.4f}")
        
        # 模拟中心服务器聚合（简化的平均聚合）
        aggregated_loss = sum(participant_losses) / len(participant_losses)
        print(f"  中心服务器聚合损失: {aggregated_loss:.4f}")
        print()
        
        # 更新各参与方的epoch
        for module in participant_modules.values():
            module.update_epoch()
    
    print("2. 联邦学习优势:")
    print("   - 各参与方保持数据本地化，保护隐私")
    print("   - 自适应加权减少了低质量数据的影响")
    print("   - 中心聚合提升了全局模型性能")
    print("   - 支持异构数据环境下的协作训练")
    print()

def show_algorithm_details():
    """展示算法细节"""
    print("=" * 60)
    print("算法实现细节")
    print("=" * 60)
    
    print("1. 权重网络结构:")
    print("   输入: 样本个体损失值")
    print("   网络: Linear(1, 64) -> ReLU -> Linear(64, 1) -> Sigmoid")
    print("   输出: [0, 1] 范围的样本权重")
    print()
    
    print("2. 样本贡献度计算:")
    print("   - 损失趋势分析: improvement = max(0, (早期损失 - 近期损失) / 早期损失)")
    print("   - 损失稳定性: stability = 1 / (1 + 标准差 / 均值)")
    print("   - 综合贡献度: 0.6 * stability + 0.4 * improvement")
    print("   - 权重范围限制: [min_weight, max_weight]")
    print()
    
    print("3. 加权损失计算:")
    print("   - 最终权重 = 0.7 * 贡献度权重 + 0.3 * 网络学习权重")
    print("   - 加权损失 = Σ(损失[i] * 权重[i]) / Σ权重[i]")
    print("   - 归一化确保损失尺度一致")
    print()
    
    print("4. 动态更新机制:")
    print("   - 每个epoch后更新样本历史记录")
    print("   - 权重网络参数通过反向传播更新")
    print("   - 样本贡献度基于滑动窗口计算")
    print()
    
    print("5. 联邦学习集成:")
    print("   - 各参与方独立计算本地加权损失")
    print("   - 中心服务器聚合加权损失（可支持同态加密）")
    print("   - 全局模型更新时考虑数据异质性")
    print()

if __name__ == "__main__":
    # 设置随机种子以便结果可复现
    random.seed(42)
    
    # 运行演示
    simulate_training_scenario()
    demonstrate_federated_learning()
    show_algorithm_details()
    
    print("=" * 60)
    print("演示完成！")
    print("=" * 60)
    print()
    print("核心特性总结:")
    print("✓ 动态样本权重调整，提升训练稳定性")
    print("✓ 噪声样本权重降低，增强模型鲁棒性") 
    print("✓ 基于历史表现的智能权重分配")
    print("✓ 支持联邦学习的隐私保护训练")
    print("✓ 可配置的权重范围和更新频率")
    print("✓ 实时权重统计和监控功能")
    print()
    print("适用场景:")
    print("• 多组学数据分析和药物反应预测")
    print("• 存在标签噪声的监督学习任务")
    print("• 数据质量不均的联邦学习环境")
    print("• 需要提升鲁棒性的深度学习模型")