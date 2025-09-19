"""
测试自适应加权模块的功能和性能
"""

import torch
import numpy as np
import unittest
import sys
import os

# 添加路径以导入模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from adaptive_weighting import (
    WeightNetwork, 
    WeightedCrossEntropyLoss, 
    SampleContributionTracker,
    AdaptiveWeightingModule, 
    FederatedWeightingModule,
    create_adaptive_weighting_module
)


class TestWeightNetwork(unittest.TestCase):
    """测试权重网络"""
    
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.weight_network = WeightNetwork().to(self.device)
    
    def test_weight_network_forward(self):
        """测试权重网络前向传播"""
        batch_size = 4
        loss_values = torch.randn(batch_size, device=self.device)
        
        weights = self.weight_network(loss_values)
        
        # 检查输出形状
        self.assertEqual(weights.shape, (batch_size, 1))
        
        # 检查权重范围在[0, 1]
        self.assertTrue(torch.all(weights >= 0))
        self.assertTrue(torch.all(weights <= 1))
    
    def test_weight_network_single_sample(self):
        """测试单样本情况（批归一化会被跳过）"""
        loss_value = torch.tensor([0.5], device=self.device)
        weight = self.weight_network(loss_value)
        
        self.assertEqual(weight.shape, (1, 1))
        self.assertTrue(0 <= weight.item() <= 1)


class TestWeightedCrossEntropyLoss(unittest.TestCase):
    """测试加权交叉熵损失"""
    
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.weighted_loss = WeightedCrossEntropyLoss().to(self.device)
    
    def test_weighted_loss_without_weights(self):
        """测试不使用权重的情况"""
        batch_size = 4
        seq_len = 10
        vocab_size = 100
        
        logits = torch.randn(batch_size * seq_len, vocab_size, device=self.device)
        targets = torch.randint(0, vocab_size, (batch_size * seq_len,), device=self.device)
        
        loss = self.weighted_loss(logits, targets)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)  # 标量损失
    
    def test_weighted_loss_with_weights(self):
        """测试使用权重的情况"""
        batch_size = 4
        seq_len = 10
        vocab_size = 100
        
        logits = torch.randn(batch_size * seq_len, vocab_size, device=self.device)
        targets = torch.randint(0, vocab_size, (batch_size * seq_len,), device=self.device)
        weights = torch.rand(batch_size, 1, device=self.device)
        
        loss = self.weighted_loss(logits, targets, weights)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)  # 标量损失
    
    def test_ignore_index(self):
        """测试忽略特定索引"""
        batch_size = 4
        seq_len = 10
        vocab_size = 100
        
        logits = torch.randn(batch_size * seq_len, vocab_size, device=self.device)
        targets = torch.randint(0, vocab_size, (batch_size * seq_len,), device=self.device)
        
        # 将一些目标设置为忽略索引
        targets[:5] = 0  # 忽略索引为0
        
        loss = self.weighted_loss(logits, targets)
        self.assertIsInstance(loss, torch.Tensor)


class TestSampleContributionTracker(unittest.TestCase):
    """测试样本贡献度追踪器"""
    
    def setUp(self):
        self.tracker = SampleContributionTracker()
    
    def test_update_sample_loss(self):
        """测试更新样本损失"""
        sample_ids = [0, 1, 2]
        losses = torch.tensor([0.5, 0.3, 0.7])
        
        self.tracker.update_sample_loss(sample_ids, losses)
        
        self.assertEqual(len(self.tracker.sample_losses[0]), 1)
        self.assertEqual(self.tracker.sample_losses[0][0], 0.5)
    
    def test_calculate_sample_contribution(self):
        """测试计算样本贡献度"""
        sample_ids = [0, 1, 2]
        
        # 添加一些历史数据
        for i in range(10):
            losses = torch.tensor([0.5 - i*0.01, 0.3 + i*0.01, 0.7])  # ID 0下降，ID 1上升，ID 2稳定
            self.tracker.update_sample_loss(sample_ids, losses)
        
        contributions = self.tracker.calculate_sample_contribution(sample_ids)
        
        self.assertEqual(contributions.shape[0], 3)
        self.assertTrue(torch.all(contributions >= 0.1))
        self.assertTrue(torch.all(contributions <= 1.0))
        
        # 损失下降的样本应该有较高贡献度
        self.assertGreater(contributions[0].item(), contributions[1].item())


class TestAdaptiveWeightingModule(unittest.TestCase):
    """测试自适应加权模块"""
    
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.module = AdaptiveWeightingModule(device=self.device)
    
    def test_forward_pass(self):
        """测试前向传播"""
        batch_size = 4
        seq_len = 10
        vocab_size = 100
        
        logits = torch.randn(batch_size * seq_len, vocab_size, device=self.device)
        targets = torch.randint(0, vocab_size, (batch_size * seq_len,), device=self.device)
        individual_losses = torch.rand(batch_size, device=self.device)
        
        weighted_loss, weights = self.module(logits, targets, individual_losses)
        
        self.assertIsInstance(weighted_loss, torch.Tensor)
        self.assertEqual(weighted_loss.dim(), 0)  # 标量损失
        self.assertEqual(weights.shape[0], batch_size)
        self.assertTrue(torch.all(weights >= self.module.min_weight))
        self.assertTrue(torch.all(weights <= self.module.max_weight))
    
    def test_get_weight_statistics(self):
        """测试权重统计功能"""
        batch_size = 4
        seq_len = 10
        vocab_size = 100
        
        logits = torch.randn(batch_size * seq_len, vocab_size, device=self.device)
        targets = torch.randint(0, vocab_size, (batch_size * seq_len,), device=self.device)
        individual_losses = torch.rand(batch_size, device=self.device)
        
        # 进行一次前向传播以生成权重
        self.module(logits, targets, individual_losses)
        
        stats = self.module.get_weight_statistics()
        
        required_keys = ['mean_weight', 'std_weight', 'min_weight', 'max_weight', 
                        'num_low_weights', 'num_high_weights']
        for key in required_keys:
            self.assertIn(key, stats)
    
    def test_epoch_update(self):
        """测试epoch更新"""
        initial_count = self.module.epoch_count
        self.module.update_epoch()
        self.assertEqual(self.module.epoch_count, initial_count + 1)


class TestFederatedWeightingModule(unittest.TestCase):
    """测试联邦学习加权模块"""
    
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fed_module = FederatedWeightingModule(device=self.device)
    
    def test_compute_encrypted_weighted_loss(self):
        """测试计算加密加权损失"""
        batch_size = 4
        seq_len = 10
        vocab_size = 100
        
        logits = torch.randn(batch_size * seq_len, vocab_size, device=self.device)
        targets = torch.randint(0, vocab_size, (batch_size * seq_len,), device=self.device)
        individual_losses = torch.rand(batch_size, device=self.device)
        
        encrypted_loss = self.fed_module.compute_encrypted_weighted_loss(
            "participant_1", logits, targets, individual_losses
        )
        
        self.assertIsInstance(encrypted_loss, torch.Tensor)
        self.assertIn("participant_1", self.fed_module.encrypted_losses)
    
    def test_aggregate_encrypted_losses(self):
        """测试聚合加密损失"""
        batch_size = 4
        seq_len = 10
        vocab_size = 100
        
        logits = torch.randn(batch_size * seq_len, vocab_size, device=self.device)
        targets = torch.randint(0, vocab_size, (batch_size * seq_len,), device=self.device)
        individual_losses = torch.rand(batch_size, device=self.device)
        
        # 添加多个参与方的损失
        for i in range(3):
            self.fed_module.compute_encrypted_weighted_loss(
                f"participant_{i}", logits, targets, individual_losses
            )
        
        aggregated_loss = self.fed_module.aggregate_encrypted_losses()
        self.assertIsInstance(aggregated_loss, torch.Tensor)
        self.assertEqual(len(self.fed_module.encrypted_losses), 0)  # 应该被清空


class TestFactoryFunction(unittest.TestCase):
    """测试工厂函数"""
    
    def test_create_adaptive_weighting_module(self):
        """测试创建自适应加权模块"""
        config = {
            'hidden_dim': 128,
            'update_frequency': 2,
            'min_weight': 0.2,
            'max_weight': 0.9,
            'device': 'cpu'
        }
        
        module = create_adaptive_weighting_module(config)
        
        self.assertIsInstance(module, AdaptiveWeightingModule)
        self.assertEqual(module.weight_network.hidden_dim, 128)
        self.assertEqual(module.update_frequency, 2)
        self.assertEqual(module.min_weight, 0.2)
        self.assertEqual(module.max_weight, 0.9)


class TestRobustness(unittest.TestCase):
    """测试模块鲁棒性"""
    
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.module = AdaptiveWeightingModule(device=self.device)
    
    def test_extreme_losses(self):
        """测试极端损失值"""
        batch_size = 4
        seq_len = 10
        vocab_size = 100
        
        logits = torch.randn(batch_size * seq_len, vocab_size, device=self.device)
        targets = torch.randint(0, vocab_size, (batch_size * seq_len,), device=self.device)
        
        # 测试极大损失值
        extreme_losses = torch.tensor([100.0, 1000.0, 0.001, 0.0], device=self.device)
        
        try:
            weighted_loss, weights = self.module(logits, targets, extreme_losses)
            self.assertTrue(torch.isfinite(weighted_loss))
            self.assertTrue(torch.all(torch.isfinite(weights)))
        except Exception as e:
            self.fail(f"模块在极端损失值下失败: {e}")
    
    def test_small_batch(self):
        """测试小批量情况"""
        batch_size = 1
        seq_len = 5
        vocab_size = 10
        
        logits = torch.randn(batch_size * seq_len, vocab_size, device=self.device)
        targets = torch.randint(0, vocab_size, (batch_size * seq_len,), device=self.device)
        individual_losses = torch.rand(batch_size, device=self.device)
        
        try:
            weighted_loss, weights = self.module(logits, targets, individual_losses)
            self.assertTrue(torch.isfinite(weighted_loss))
            self.assertTrue(torch.all(torch.isfinite(weights)))
        except Exception as e:
            self.fail(f"模块在小批量情况下失败: {e}")


def run_performance_test():
    """运行性能测试"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    module = AdaptiveWeightingModule(device=device)
    
    # 大批量测试
    batch_size = 64
    seq_len = 512
    vocab_size = 30000
    
    print(f"运行性能测试 (设备: {device})")
    print(f"批量大小: {batch_size}, 序列长度: {seq_len}, 词汇表大小: {vocab_size}")
    
    logits = torch.randn(batch_size * seq_len, vocab_size, device=device)
    targets = torch.randint(0, vocab_size, (batch_size * seq_len,), device=device)
    individual_losses = torch.rand(batch_size, device=device)
    
    import time
    
    # 预热
    for _ in range(5):
        _, _ = module(logits, targets, individual_losses)
    
    # 实际测试
    start_time = time.time()
    num_iterations = 20
    
    for _ in range(num_iterations):
        weighted_loss, weights = module(logits, targets, individual_losses)
    
    end_time = time.time()
    avg_time = (end_time - start_time) / num_iterations
    
    print(f"平均处理时间: {avg_time*1000:.2f} ms/batch")
    print(f"内存使用情况:")
    if device.type == 'cuda':
        print(f"  GPU内存已分配: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
        print(f"  GPU内存最大使用: {torch.cuda.max_memory_allocated(device) / 1024**2:.2f} MB")


if __name__ == "__main__":
    # 运行单元测试
    print("运行自适应加权模块单元测试...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    print("\n" + "="*50)
    # 运行性能测试
    run_performance_test()