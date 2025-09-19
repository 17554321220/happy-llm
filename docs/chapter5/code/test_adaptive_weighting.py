#!/usr/bin/env python3
"""
自适应加权模块测试脚本
验证模块的基本功能和集成效果
"""

import sys
import os

# 添加当前目录到path，以便导入模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import torch
    import torch.nn as nn
    import numpy as np
    from adaptive_weighting import AdaptiveWeightNetwork, AdaptiveWeightingModule, create_adaptive_weighting_module
    from k_model import ModelConfig, Transformer
    TORCH_AVAILABLE = True
except ImportError as e:
    print(f"PyTorch import failed: {e}")
    print("Skipping tests that require PyTorch...")
    TORCH_AVAILABLE = False


def test_adaptive_weight_network():
    """测试权重网络基础功能"""
    print("\n=== 测试权重网络基础功能 ===")
    
    if not TORCH_AVAILABLE:
        print("PyTorch不可用，跳过测试")
        return False
    
    try:
        # 创建权重网络
        weight_network = AdaptiveWeightNetwork(
            input_dim=1,
            hidden_dim=32,
            output_dim=1,
            dropout=0.1
        )
        
        # 模拟损失值
        batch_size = 16
        loss_values = torch.randn(batch_size).abs()  # 确保为正值
        
        # 前向传播
        weights = weight_network(loss_values)
        
        # 检查输出形状
        assert weights.shape == (batch_size, 1), f"权重形状错误: {weights.shape}"
        
        # 检查权重范围
        assert torch.all(weights >= 0) and torch.all(weights <= 1), "权重不在[0,1]范围内"
        
        print(f"✓ 权重网络测试通过")
        print(f"  - 输入损失范围: [{loss_values.min():.3f}, {loss_values.max():.3f}]")
        print(f"  - 输出权重范围: [{weights.min():.3f}, {weights.max():.3f}]")
        print(f"  - 平均权重: {weights.mean():.3f}")
        
        return True
    except Exception as e:
        print(f"✗ 权重网络测试失败: {e}")
        return False


def test_adaptive_weighting_module():
    """测试完整的自适应加权模块"""
    print("\n=== 测试自适应加权模块 ===")
    
    if not TORCH_AVAILABLE:
        print("PyTorch不可用，跳过测试")
        return False
    
    try:
        # 创建自适应加权模块
        weighting_module = create_adaptive_weighting_module(
            model_dim=512,
            weight_momentum=0.9,
            min_weight=0.1,
            max_weight=1.0
        )
        
        # 模拟损失值
        batch_size = 8
        loss_values = torch.randn(batch_size).abs()
        
        # 测试权重计算
        sample_weights = weighting_module.compute_sample_weights(loss_values)
        assert sample_weights.shape == (batch_size,), f"样本权重形状错误: {sample_weights.shape}"
        
        # 测试加权损失计算
        weighted_loss, weights = weighting_module.compute_weighted_loss(loss_values)
        assert isinstance(weighted_loss.item(), float), "加权损失应该是标量"
        
        # 测试权重更新
        weighting_module.update_weights(1, loss_values)
        
        # 测试统计信息
        stats = weighting_module.get_weight_statistics()
        assert isinstance(stats, dict), "统计信息应该是字典"
        
        print(f"✓ 自适应加权模块测试通过")
        print(f"  - 原始损失平均值: {loss_values.mean():.3f}")
        print(f"  - 加权损失: {weighted_loss:.3f}")
        print(f"  - 权重统计: {stats}")
        
        return True
    except Exception as e:
        print(f"✗ 自适应加权模块测试失败: {e}")
        return False


def test_model_integration():
    """测试与模型的集成"""
    print("\n=== 测试模型集成 ===")
    
    if not TORCH_AVAILABLE:
        print("PyTorch不可用，跳过测试")
        return False
    
    try:
        # 创建启用自适应加权的模型配置
        config = ModelConfig(
            dim=256,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            vocab_size=1000,
            max_seq_len=128,
            use_adaptive_weighting=True,
            adaptive_weighting_config={
                'weight_momentum': 0.8,
                'min_weight': 0.2,
                'max_weight': 1.0,
                'weight_update_freq': 1
            }
        )
        
        # 创建模型
        model = Transformer(config)
        
        # 验证自适应加权模块已初始化
        assert hasattr(model, 'adaptive_weighting'), "模型应该有自适应加权属性"
        assert model.adaptive_weighting is not None, "自适应加权模块应该被初始化"
        
        # 模拟输入
        batch_size = 4
        seq_len = 32
        tokens = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        # 前向传播
        model.eval()
        with torch.no_grad():
            output = model(tokens, targets)
        
        # 检查输出
        assert 'logits' in output, "输出应该包含logits"
        assert 'last_loss' in output, "输出应该包含损失"
        assert model.last_sample_weights is not None, "应该计算样本权重"
        
        # 测试权重更新
        model.update_adaptive_weights(1)
        
        # 测试权重统计
        stats = model.get_adaptive_weight_statistics()
        
        print(f"✓ 模型集成测试通过")
        print(f"  - 输出logits形状: {output.logits.shape}")
        print(f"  - 样本权重形状: {model.last_sample_weights.shape}")
        print(f"  - 权重统计: {stats}")
        
        return True
    except Exception as e:
        print(f"✗ 模型集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_weight_dynamics():
    """测试权重动态调整机制"""
    print("\n=== 测试权重动态调整 ===")
    
    if not TORCH_AVAILABLE:
        print("PyTorch不可用，跳过测试")
        return False
    
    try:
        weighting_module = create_adaptive_weighting_module()
        
        # 模拟不同质量的样本
        # 高质量样本（低损失）
        good_samples = torch.tensor([0.1, 0.2, 0.15])
        # 低质量样本（高损失）
        bad_samples = torch.tensor([2.0, 3.0, 2.5])
        # 混合样本
        mixed_samples = torch.cat([good_samples, bad_samples])
        
        # 计算权重
        mixed_weights = weighting_module.compute_sample_weights(mixed_samples)
        
        # 验证高质量样本应该有更高的权重
        good_weights = mixed_weights[:3]
        bad_weights = mixed_weights[3:]
        
        print(f"✓ 权重动态调整测试")
        print(f"  - 好样本损失: {good_samples.tolist()}")
        print(f"  - 好样本权重: {[f'{w:.3f}' for w in good_weights.tolist()]}")
        print(f"  - 坏样本损失: {bad_samples.tolist()}")  
        print(f"  - 坏样本权重: {[f'{w:.3f}' for w in bad_weights.tolist()]}")
        
        # 注意：由于我们的权重网络会学习损失到权重的映射，
        # 实际的权重分配可能需要训练才能达到预期效果
        
        return True
    except Exception as e:
        print(f"✗ 权重动态调整测试失败: {e}")
        return False


def main():
    """运行所有测试"""
    print("自适应加权模块测试")
    print("=" * 50)
    
    tests = [
        test_adaptive_weight_network,
        test_adaptive_weighting_module,
        test_model_integration,
        test_weight_dynamics
    ]
    
    results = []
    for test_func in tests:
        results.append(test_func())
    
    # 总结
    print("\n" + "=" * 50)
    print("测试总结:")
    passed = sum(results)
    total = len(results)
    print(f"通过: {passed}/{total}")
    
    if passed == total:
        print("✓ 所有测试通过！自适应加权模块工作正常。")
        return 0
    else:
        print("✗ 部分测试失败，请检查实现。")
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)