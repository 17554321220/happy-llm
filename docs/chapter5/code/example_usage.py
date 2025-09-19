#!/usr/bin/env python3
"""
自适应加权模块使用示例

演示如何在实际训练中使用自适应加权模块
"""

import torch
from torch.utils.data import DataLoader
import argparse

from k_model import ModelConfig, Transformer
from dataset import SFTDataset  
from adaptive_weighting import create_adaptive_weighting_module


def train_with_adaptive_weighting(args):
    """使用自适应加权模块进行训练"""
    
    device = args.device
    print(f"Using device: {device}")
    
    # 1. 创建模型
    lm_config = ModelConfig(
        dim=args.model_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        vocab_size=args.vocab_size,
        max_seq_len=args.max_seq_len
    )
    
    model = Transformer(lm_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # 2. 创建自适应加权模块
    if args.use_adaptive_weighting:
        adaptive_config = {
            'weight_net_hidden_dim': args.weight_net_hidden_dim,
            'ema_decay': args.ema_decay,
            'min_weight': args.min_weight,
            'device': device,
            'update_frequency': args.weight_update_frequency
        }
        adaptive_module = create_adaptive_weighting_module(adaptive_config)
        print(f"Adaptive weighting enabled with config: {adaptive_config}")
    else:
        adaptive_module = None
        print("Using standard training")
    
    # 3. 训练循环
    model.train()
    
    for epoch in range(args.epochs):
        epoch_loss = 0
        epoch_weighted_loss = 0
        num_batches = 0
        
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # 模拟训练数据
        for step in range(args.steps_per_epoch):
            # 创建模拟批次数据
            batch_size = args.batch_size
            seq_len = args.max_seq_len
            
            X = torch.randint(1, args.vocab_size, (batch_size, seq_len)).to(device)
            Y = torch.randint(1, args.vocab_size, (batch_size, seq_len)).to(device)
            loss_mask = torch.ones(batch_size, seq_len).to(device)
            
            # 生成样本ID
            sample_ids = [epoch * args.steps_per_epoch * batch_size + step * batch_size + i 
                         for i in range(batch_size)]
            
            optimizer.zero_grad()
            
            # 前向传播
            output = model(X, Y)
            logits = output.logits
            
            if args.use_adaptive_weighting and adaptive_module:
                # 使用自适应加权损失
                weighted_loss, weights = adaptive_module.compute_weighted_loss(
                    logits, Y, loss_mask, sample_ids
                )
                loss = weighted_loss
                
                # 记录权重统计
                avg_weight = weights.mean().item()
                min_weight = weights.min().item()
                max_weight = weights.max().item()
                
            else:
                # 标准损失计算
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), Y.view(-1), reduction='mean'
                )
                avg_weight = min_weight = max_weight = 1.0
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            # 定期打印进度
            if step % args.log_interval == 0:
                print(f"  Step {step}: loss={loss.item():.4f}, "
                      f"weights=[{min_weight:.3f}, {avg_weight:.3f}, {max_weight:.3f}]")
        
        # Epoch结束统计
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch + 1} completed: avg_loss={avg_loss:.4f}")
        
        # 显示样本统计信息
        if args.use_adaptive_weighting and adaptive_module:
            stats = adaptive_module.get_sample_statistics()
            if stats:
                # 显示贡献度最高和最低的样本
                sorted_samples = sorted(stats.items(), key=lambda x: x[1]['contribution'], reverse=True)
                print(f"  Top samples by contribution:")
                for i, (sample_id, sample_stats) in enumerate(sorted_samples[:3]):
                    print(f"    Sample {sample_id}: contribution={sample_stats['contribution']:.3f}, "
                          f"loss_ema={sample_stats['loss_ema']:.3f}")
                
                print(f"  Bottom samples by contribution:")
                for i, (sample_id, sample_stats) in enumerate(sorted_samples[-3:]):
                    print(f"    Sample {sample_id}: contribution={sample_stats['contribution']:.3f}, "
                          f"loss_ema={sample_stats['loss_ema']:.3f}")
    
    print("\nTraining completed!")


def main():
    parser = argparse.ArgumentParser(description="Adaptive Weighting Training Example")
    
    # 模型参数
    parser.add_argument("--model_dim", type=int, default=256, help="Model dimension")
    parser.add_argument("--n_layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--vocab_size", type=int, default=1000, help="Vocabulary size")
    parser.add_argument("--max_seq_len", type=int, default=32, help="Maximum sequence length")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--steps_per_epoch", type=int, default=10, help="Steps per epoch")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--log_interval", type=int, default=5, help="Log interval")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                       help="Device to use")
    
    # 自适应加权参数
    parser.add_argument("--use_adaptive_weighting", action="store_true", 
                       help="Enable adaptive weighting")
    parser.add_argument("--weight_net_hidden_dim", type=int, default=32, 
                       help="Weight network hidden dimension")
    parser.add_argument("--ema_decay", type=float, default=0.9, 
                       help="EMA decay factor")
    parser.add_argument("--min_weight", type=float, default=0.1, 
                       help="Minimum sample weight")
    parser.add_argument("--weight_update_frequency", type=int, default=5, 
                       help="Weight network update frequency")
    
    args = parser.parse_args()
    
    print("="*60)
    print("Adaptive Weighting Training Example")
    print("="*60)
    print(f"Configuration:")
    print(f"  Model: {args.model_dim}d, {args.n_layers}L, {args.n_heads}H")
    print(f"  Training: {args.epochs} epochs, {args.steps_per_epoch} steps/epoch")
    print(f"  Adaptive weighting: {args.use_adaptive_weighting}")
    if args.use_adaptive_weighting:
        print(f"    Hidden dim: {args.weight_net_hidden_dim}")
        print(f"    EMA decay: {args.ema_decay}")
        print(f"    Min weight: {args.min_weight}")
    print("="*60)
    
    train_with_adaptive_weighting(args)


if __name__ == "__main__":
    main()