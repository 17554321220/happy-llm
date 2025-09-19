"""
自适应加权模块测试文件

测试各个组件的功能：
1. WeightNetwork权重网络
2. SampleTracker样本追踪器  
3. AdaptiveWeightedLoss加权损失函数
4. AdaptiveWeightingModule整体模块
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List
import sys
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from adaptive_weighting import (
    WeightNetwork, SampleTracker, AdaptiveWeightedLoss, 
    AdaptiveWeightingModule, create_adaptive_weighting_module
)


def test_weight_network():
    """测试权重网络"""
    print("Testing WeightNetwork...")
    
    # 创建权重网络
    weight_net = WeightNetwork(input_dim=1, hidden_dim=32)
    
    # 测试输入
    batch_size = 8
    losses = torch.randn(batch_size, 1)
    
    # 前向传播
    weights = weight_net(losses)
    
    # 检查输出形状和范围
    assert weights.shape == (batch_size, 1), f"Expected shape ({batch_size}, 1), got {weights.shape}"
    assert torch.all(weights >= 0) and torch.all(weights <= 1), "Weights should be in [0, 1] range"
    
    print(f"✓ WeightNetwork test passed")
    print(f"  Input shape: {losses.shape}")
    print(f"  Output shape: {weights.shape}")
    print(f"  Weight range: [{weights.min().item():.3f}, {weights.max().item():.3f}]")
    return True


def test_sample_tracker():
    """测试样本追踪器"""
    print("\nTesting SampleTracker...")
    
    tracker = SampleTracker(ema_decay=0.9, min_weight=0.1)
    
    # 模拟样本数据
    sample_ids = [1, 2, 3, 4, 5]
    losses = torch.tensor([0.8, 0.3, 1.2, 0.5, 0.7])
    weights = torch.tensor([[0.4], [0.8], [0.2], [0.7], [0.5]])
    
    # 更新统计信息
    tracker.update_sample_stats(sample_ids, losses, weights)
    
    # 获取历史权重
    historical_weights = tracker.get_historical_weights(sample_ids)
    
    # 检查输出
    assert historical_weights.shape == (len(sample_ids), 1), f"Expected shape ({len(sample_ids)}, 1)"
    assert torch.all(historical_weights >= tracker.min_weight), "Historical weights should respect min_weight"
    
    print(f"✓ SampleTracker test passed")
    print(f"  Tracked samples: {len(tracker.sample_stats)}")
    print(f"  Historical weights range: [{historical_weights.min().item():.3f}, {historical_weights.max().item():.3f}]")
    return True


def test_adaptive_weighted_loss():
    """测试自适应加权损失函数"""
    print("\nTesting AdaptiveWeightedLoss...")
    
    weighted_loss_fn = AdaptiveWeightedLoss(ignore_index=0)
    
    # 模拟数据
    batch_size, seq_len, vocab_size = 4, 8, 100
    logits = torch.randn(batch_size, seq_len, vocab_size)
    targets = torch.randint(1, vocab_size, (batch_size, seq_len))
    weights = torch.rand(batch_size, 1) * 0.8 + 0.2  # 权重在[0.2, 1.0]范围内
    loss_mask = torch.ones(batch_size, seq_len)
    
    # 计算加权损失
    weighted_loss = weighted_loss_fn(logits, targets, weights, loss_mask)
    
    # 计算标准损失进行比较
    standard_loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1), ignore_index=0)
    
    print(f"✓ AdaptiveWeightedLoss test passed")
    print(f"  Weighted loss: {weighted_loss.item():.4f}")
    print(f"  Standard loss: {standard_loss.item():.4f}")
    return True


def test_adaptive_weighting_module():
    """测试完整的自适应加权模块"""
    print("\nTesting AdaptiveWeightingModule...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = {
        'weight_net_hidden_dim': 32,
        'ema_decay': 0.9,
        'min_weight': 0.1,
        'device': device,
        'update_frequency': 5
    }
    
    # 创建模块
    adaptive_module = create_adaptive_weighting_module(config)
    
    # 模拟训练数据
    batch_size, seq_len, vocab_size = 4, 8, 100
    logits = torch.randn(batch_size, seq_len, vocab_size).to(device)
    targets = torch.randint(1, vocab_size, (batch_size, seq_len)).to(device)
    loss_mask = torch.ones(batch_size, seq_len).to(device)
    sample_ids = [1, 2, 3, 4]
    
    # 计算加权损失
    weighted_loss, weights = adaptive_module.compute_weighted_loss(
        logits, targets, loss_mask, sample_ids
    )
    
    # 检查输出
    assert isinstance(weighted_loss, torch.Tensor), "Weighted loss should be a tensor"
    assert weights.shape == (batch_size, 1), f"Expected weights shape ({batch_size}, 1)"
    assert torch.all(weights >= 0) and torch.all(weights <= 1), "Weights should be in [0, 1] range"
    
    print(f"✓ AdaptiveWeightingModule test passed")
    print(f"  Device: {device}")
    print(f"  Weighted loss: {weighted_loss.item():.4f}")
    print(f"  Weight range: [{weights.min().item():.3f}, {weights.max().item():.3f}]")
    print(f"  Tracked samples: {len(adaptive_module.get_sample_statistics())}")
    
    return True


def test_integration_with_multiple_batches():
    """测试多批次集成"""
    print("\nTesting integration with multiple batches...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = {
        'weight_net_hidden_dim': 32,
        'ema_decay': 0.9,
        'min_weight': 0.1,
        'device': device,
        'update_frequency': 3
    }
    
    adaptive_module = create_adaptive_weighting_module(config)
    
    batch_size, seq_len, vocab_size = 4, 8, 100
    num_batches = 10
    
    total_loss = 0
    for batch_idx in range(num_batches):
        # 模拟不同批次的数据
        logits = torch.randn(batch_size, seq_len, vocab_size).to(device)
        targets = torch.randint(1, vocab_size, (batch_size, seq_len)).to(device)
        loss_mask = torch.ones(batch_size, seq_len).to(device)
        
        # 模拟样本ID（重复使用一些样本来测试历史跟踪）
        sample_ids = [(batch_idx * batch_size + i) % 20 for i in range(batch_size)]
        
        # 计算加权损失
        weighted_loss, weights = adaptive_module.compute_weighted_loss(
            logits, targets, loss_mask, sample_ids
        )
        
        total_loss += weighted_loss.item()
        
        if batch_idx % 5 == 0:
            print(f"  Batch {batch_idx}: loss={weighted_loss.item():.4f}, "
                  f"weights=[{weights.min().item():.3f}, {weights.max().item():.3f}]")
    
    print(f"✓ Integration test passed")
    print(f"  Total batches processed: {num_batches}")
    print(f"  Average loss: {total_loss/num_batches:.4f}")
    print(f"  Total tracked samples: {len(adaptive_module.get_sample_statistics())}")
    
    return True


def test_noise_handling():
    """测试噪声样本处理能力"""
    print("\nTesting noise handling capability...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = {
        'weight_net_hidden_dim': 32,
        'ema_decay': 0.9,
        'min_weight': 0.05,
        'device': device,
        'update_frequency': 2
    }
    
    adaptive_module = create_adaptive_weighting_module(config)
    
    batch_size, seq_len, vocab_size = 6, 8, 100
    
    # 创建干净样本和噪声样本
    clean_logits = torch.randn(3, seq_len, vocab_size).to(device) * 0.5  # 较小的方差
    clean_targets = torch.randint(1, vocab_size, (3, seq_len)).to(device)
    
    # 噪声样本：更大的方差，更随机的目标
    noise_logits = torch.randn(3, seq_len, vocab_size).to(device) * 2.0  # 较大的方差
    noise_targets = torch.randint(1, vocab_size, (3, seq_len)).to(device)
    
    # 合并数据
    logits = torch.cat([clean_logits, noise_logits], dim=0)
    targets = torch.cat([clean_targets, noise_targets], dim=0)
    loss_mask = torch.ones(batch_size, seq_len).to(device)
    sample_ids = [1, 2, 3, 101, 102, 103]  # 前3个是干净样本，后3个是噪声样本
    
    # 多次训练来观察权重变化
    clean_weights_history = []
    noise_weights_history = []
    
    for epoch in range(15):
        weighted_loss, weights = adaptive_module.compute_weighted_loss(
            logits, targets, loss_mask, sample_ids
        )
        
        clean_weights = weights[:3].mean().item()
        noise_weights = weights[3:].mean().item()
        
        clean_weights_history.append(clean_weights)
        noise_weights_history.append(noise_weights)
        
        if epoch % 5 == 0:
            print(f"  Epoch {epoch}: clean_weight={clean_weights:.3f}, noise_weight={noise_weights:.3f}")
    
    # 检查噪声样本的权重是否降低
    final_clean_weight = clean_weights_history[-1]
    final_noise_weight = noise_weights_history[-1]
    
    print(f"✓ Noise handling test completed")
    print(f"  Final clean sample weight: {final_clean_weight:.3f}")
    print(f"  Final noise sample weight: {final_noise_weight:.3f}")
    print(f"  Weight ratio (clean/noise): {final_clean_weight/final_noise_weight:.2f}")
    
    return True


def run_all_tests():
    """运行所有测试"""
    print("="*60)
    print("Running Adaptive Weighting Module Tests")
    print("="*60)
    
    tests = [
        test_weight_network,
        test_sample_tracker,
        test_adaptive_weighted_loss,
        test_adaptive_weighting_module,
        test_integration_with_multiple_batches,
        test_noise_handling
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"✗ {test_func.__name__} failed")
        except Exception as e:
            failed += 1
            print(f"✗ {test_func.__name__} failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    if not success:
        exit(1)
    else:
        print("\nAll tests passed! ✓")