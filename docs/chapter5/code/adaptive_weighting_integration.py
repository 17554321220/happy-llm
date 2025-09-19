"""
自适应加权模块集成示例
展示如何将自适应加权模块集成到现有的训练脚本中
"""

# 1. 导入必要的模块
from adaptive_weighting import AdaptiveWeightingModule, create_adaptive_weighting_module

# 2. 在训练脚本初始化部分添加自适应加权模块
def init_adaptive_weighting(args, device):
    """初始化自适应加权模块"""
    if not args.use_adaptive_weighting:
        return None
    
    config = {
        'hidden_dim': getattr(args, 'weight_hidden_dim', 64),
        'min_weight': getattr(args, 'min_weight', 0.1),
        'max_weight': getattr(args, 'max_weight', 1.0),
        'update_frequency': getattr(args, 'weight_update_freq', 1),
        'device': str(device)
    }
    
    return create_adaptive_weighting_module(config)

# 3. 修改训练循环以支持自适应加权
def train_epoch_with_weighting(epoch, model, train_loader, optimizer, scaler, 
                              adaptive_weighting_module=None, args=None, ctx=None):
    """带自适应加权的训练epoch"""
    
    model.train()
    epoch_stats = {
        'total_loss': 0.0,
        'weighted_loss': 0.0,
        'regular_loss': 0.0,
        'num_batches': 0
    }
    
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)
        
        # 前向传播
        with ctx:
            out = model(X, Y)
            base_loss = out.last_loss
            
            if adaptive_weighting_module is not None:
                # 使用自适应加权
                logits = out.logits.view(-1, out.logits.size(-1))
                targets = Y.view(-1)
                
                # 计算每个样本的个体损失
                individual_losses = []
                batch_size = X.size(0)
                for i in range(batch_size):
                    sample_loss = base_loss[i * X.size(1):(i+1) * X.size(1)]
                    individual_losses.append(sample_loss.mean().item())
                
                individual_losses_tensor = torch.tensor(individual_losses, device=X.device)
                sample_ids = [step * batch_size + i for i in range(batch_size)]
                
                # 计算加权损失
                weighted_loss, sample_weights = adaptive_weighting_module(
                    logits, targets, individual_losses_tensor, sample_ids
                )
                
                final_loss = weighted_loss
                
                # 记录统计信息
                weight_stats = adaptive_weighting_module.get_weight_statistics()
                
            else:
                # 常规训练
                loss_mask = loss_mask.view(-1)
                final_loss = torch.sum(base_loss * loss_mask) / loss_mask.sum()
                weight_stats = {}
        
        # 反向传播
        scaler.scale(final_loss).backward()
        
        # 更新参数
        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        
        # 记录统计
        epoch_stats['total_loss'] += final_loss.item()
        epoch_stats['num_batches'] += 1
        
        # 定期打印日志
        if step % args.log_interval == 0:
            if adaptive_weighting_module:
                print(f"Epoch {epoch}, Step {step}: "
                     f"Weighted Loss: {final_loss.item():.4f}, "
                     f"Mean Weight: {weight_stats.get('mean_weight', 0):.3f}")
            else:
                print(f"Epoch {epoch}, Step {step}: "
                     f"Loss: {final_loss.item():.4f}")
    
    # Epoch结束时更新自适应加权模块
    if adaptive_weighting_module:
        adaptive_weighting_module.update_epoch()
    
    return epoch_stats

# 4. 主训练函数示例
def main_training_with_adaptive_weighting():
    """主训练函数示例"""
    
    # 假设已经初始化了这些变量
    # args, model, tokenizer, train_loader, optimizer, scaler, ctx = initialize_training()
    
    # 初始化自适应加权模块
    adaptive_weighting_module = init_adaptive_weighting(args, args.device)
    
    if adaptive_weighting_module:
        print("自适应加权模块已启用")
        # 如果使用自适应加权，需要将权重网络参数加入优化器
        all_params = list(model.parameters()) + list(adaptive_weighting_module.parameters())
        optimizer = torch.optim.AdamW(all_params, lr=args.learning_rate)
    else:
        print("使用常规训练模式")
    
    # 训练循环
    for epoch in range(args.epochs):
        print(f"开始训练 Epoch {epoch + 1}/{args.epochs}")
        
        epoch_stats = train_epoch_with_weighting(
            epoch, model, train_loader, optimizer, scaler,
            adaptive_weighting_module, args, ctx
        )
        
        avg_loss = epoch_stats['total_loss'] / epoch_stats['num_batches']
        print(f"Epoch {epoch + 1} 完成，平均损失: {avg_loss:.4f}")
        
        # 保存模型检查点
        if (epoch + 1) % args.save_frequency == 0:
            save_checkpoint(model, optimizer, adaptive_weighting_module, epoch, args)

# 5. 保存和加载检查点
def save_checkpoint(model, optimizer, adaptive_weighting_module, epoch, args):
    """保存训练检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    
    if adaptive_weighting_module:
        checkpoint['adaptive_weighting_state_dict'] = adaptive_weighting_module.state_dict()
    
    checkpoint_path = f"{args.save_dir}/checkpoint_epoch_{epoch+1}.pth"
    torch.save(checkpoint, checkpoint_path)
    print(f"检查点已保存: {checkpoint_path}")

def load_checkpoint(model, optimizer, adaptive_weighting_module, checkpoint_path):
    """加载训练检查点"""
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if adaptive_weighting_module and 'adaptive_weighting_state_dict' in checkpoint:
        adaptive_weighting_module.load_state_dict(checkpoint['adaptive_weighting_state_dict'])
    
    epoch = checkpoint['epoch']
    print(f"检查点已加载: {checkpoint_path}, Epoch: {epoch}")
    return epoch

# 6. 命令行参数示例
def add_adaptive_weighting_args(parser):
    """添加自适应加权相关的命令行参数"""
    
    # 自适应加权基础参数
    parser.add_argument("--use_adaptive_weighting", action="store_true", 
                       help="是否启用自适应加权模块")
    parser.add_argument("--weight_hidden_dim", type=int, default=64,
                       help="权重网络隐藏层维度")
    parser.add_argument("--min_weight", type=float, default=0.1,
                       help="最小样本权重")
    parser.add_argument("--max_weight", type=float, default=1.0,
                       help="最大样本权重")
    parser.add_argument("--weight_update_freq", type=int, default=1,
                       help="权重更新频率（每几个epoch）")
    
    return parser

# 7. 使用示例
"""
使用方法：

1. 在训练脚本中导入此模块：
   from adaptive_weighting_integration import *

2. 添加命令行参数：
   parser = add_adaptive_weighting_args(parser)

3. 在训练循环中使用：
   adaptive_weighting_module = init_adaptive_weighting(args, device)
   train_epoch_with_weighting(epoch, model, train_loader, optimizer, scaler, 
                             adaptive_weighting_module, args, ctx)

4. 运行训练：
   python train.py --use_adaptive_weighting --weight_hidden_dim 128 --min_weight 0.2
"""

if __name__ == "__main__":
    print("自适应加权模块集成示例")
    print("请查看代码注释了解集成方法")
    print()
    print("主要集成步骤：")
    print("1. 导入 adaptive_weighting 模块")
    print("2. 初始化自适应加权模块")
    print("3. 修改训练循环以支持加权损失")
    print("4. 将权重网络参数加入优化器")
    print("5. 保存/加载时包含权重模块状态")
    print()
    print("详细使用方法请参考代码注释和 ddp_sft_weighted.py 脚本")