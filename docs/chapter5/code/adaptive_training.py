"""
集成自适应加权模块的训练脚本
支持多组学数据的去噪处理，通过动态样本权重提升模型鲁棒性
"""

import os
import platform
import argparse
import time
import warnings
import math
import pandas as pd
import torch
from torch import optim
from torch.utils.data import DataLoader
from contextlib import nullcontext

from transformers import AutoTokenizer

from k_model import ModelConfig, Transformer
from dataset import SFTDataset

import swanlab

# 忽略警告
warnings.filterwarnings('ignore')


def Logger(content):
    """日志记录器"""
    print(content)

def get_lr(it, all):
    """获取学习率"""
    # 1) linear warmup for warmup_iters steps
    # 1) 预热迭代的线性预热
    warmup_iters = args.warmup_iters
    lr_decay_iters = all
    min_lr = args.learning_rate / 10

    if it < warmup_iters:
        return args.learning_rate * it / warmup_iters
    
    # 2) if it > lr_decay_iters, return min learning rate
    # 2) 如果迭代次数超过学习率衰减迭代次数，则返回最小学习率
    if it > lr_decay_iters:
        return min_lr
    
    # 3) in between, use cosine decay down to min learning rate
    # 3) 在两者之间，使用余弦衰减至最小学习率
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (args.learning_rate - min_lr)

def train_epoch(epoch):
    """训练一个epoch，支持自适应加权"""
    start_time = time.time()
    epoch_sample_losses = []  # 收集整个epoch的样本损失用于权重更新
    epoch_sample_weights = []  # 收集样本权重用于统计
    
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        # 获取学习率并更新优化器
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 前向传播
        with ctx:
            out = model(X, Y)
            loss = out.last_loss / args.accumulation_steps
            loss_mask = loss_mask.view(-1)
            loss = torch.sum(loss * loss_mask) / loss_mask.sum()
            
            # 收集样本信息用于自适应权重更新
            if hasattr(model, 'last_sample_weights') and model.last_sample_weights is not None:
                # 计算样本级别的损失
                sample_loss = (loss * loss_mask).view(X.size(0), -1).mean(dim=1)
                epoch_sample_losses.append(sample_loss.detach().cpu())
                epoch_sample_weights.append(model.last_sample_weights.detach().cpu())

        # 反向传播
        scaler.scale(loss).backward()

        # 更新权重
        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        # 打印日志
        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            
            # 获取自适应权重统计信息
            weight_stats = model.get_adaptive_weight_statistics()
            weight_info = ""
            if weight_stats:
                weight_info = f" | Weights: mean={weight_stats.get('mean_weight', 0):.3f}, " \
                            f"low={weight_stats.get('num_low_weight_samples', 0)}, " \
                            f"high={weight_stats.get('num_high_weight_samples', 0)}"
            
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.7f} epoch_Time:{}min{}'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60,
                    weight_info))
                    
            if args.use_swanlab:
                log_data = {
                    "loss": loss.item() * args.accumulation_steps,
                    "lr": optimizer.param_groups[-1]['lr']
                }
                # 添加权重统计到日志
                if weight_stats:
                    log_data.update({f"weight_{k}": v for k, v in weight_stats.items()})
                swanlab.log(log_data)

        # 保存模型检查点
        if step % args.save_interval == 0 and step != 0:
            model_save_name = f"{args.model_save_name}_epoch_{epoch}_step_{step}.pth"
            torch.save(model.state_dict(), model_save_name)
            Logger(f'Model saved as {model_save_name}')
    
    # 在epoch结束时更新自适应权重
    if epoch_sample_losses and hasattr(model, 'update_adaptive_weights'):
        # 合并所有batch的样本损失
        all_sample_losses = torch.cat(epoch_sample_losses, dim=0)
        model.update_adaptive_weights(epoch)
        
        # 输出权重更新信息
        weight_stats = model.get_adaptive_weight_statistics()
        if weight_stats:
            Logger(f"Epoch {epoch+1} adaptive weight update: {weight_stats}")

def init_model():
    """初始化模型"""
    def count_parameters(model):
        """计算模型参数量"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained('./tokenizer_k/')

    # 创建模型配置，启用自适应加权
    config = ModelConfig(
        **lm_config,
        use_adaptive_weighting=args.use_adaptive_weighting,
        adaptive_weighting_config={
            'weight_momentum': args.weight_momentum,
            'min_weight': args.min_weight,
            'max_weight': args.max_weight,
            'weight_update_freq': args.weight_update_freq,
            'dropout': args.adaptive_dropout
        }
    )
    
    # 初始化模型
    model = Transformer(config)

    # 加载预训练权重
    if args.pretrain_model_path:
        ckp = args.pretrain_model_path
        state_dict = torch.load(ckp, map_location=args.device)
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict, strict=False)
        Logger("Loaded pretrained model")
    
    # 多卡初始化
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        Logger(f"Using {num_gpus} GPUs with DataParallel!")
        model = torch.nn.DataParallel(model)
    
    model.to(args.device)
    
    # 统计参数量
    Logger(f'LLM总参数量：{count_parameters(model) / 1e6:.3f} 百万')
    if args.use_adaptive_weighting:
        # 统计自适应权重模块的参数
        if hasattr(model, 'module'):  # DataParallel情况
            adaptive_params = sum(p.numel() for p in model.module.adaptive_weighting.parameters() if p.requires_grad)
        else:
            adaptive_params = sum(p.numel() for p in model.adaptive_weighting.parameters() if p.requires_grad)
        Logger(f'自适应加权模块参数量：{adaptive_params / 1e3:.3f} 千')
    
    return model, tokenizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # 模型和数据参数
    parser.add_argument("--model_config", type=str, default="./models/7B/params.json", help="模型配置文件路径")
    parser.add_argument("--tokenizer", type=str, default="./tokenizer.model", help="分词器路径") 
    parser.add_argument("--dataset", type=str, default="./sft_data_single.jsonl", help="训练数据路径")
    parser.add_argument("--pretrain_model_path", type=str, default="", help="预训练模型路径")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=64, help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="数据类型")
    
    # 实验跟踪和数据加载参数
    parser.add_argument("--use_swanlab", action="store_true", help="是否使用SwanLab进行实验跟踪")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载的工作进程数")
    
    # 训练优化参数
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--warmup_iters", type=int, default=1000, help="预热迭代次数")
    
    # 自适应加权模块参数
    parser.add_argument("--use_adaptive_weighting", action="store_true", help="是否使用自适应加权模块")
    parser.add_argument("--weight_momentum", type=float, default=0.9, help="权重更新动量")
    parser.add_argument("--min_weight", type=float, default=0.1, help="最小样本权重")
    parser.add_argument("--max_weight", type=float, default=1.0, help="最大样本权重")
    parser.add_argument("--weight_update_freq", type=int, default=1, help="权重更新频率(epoch)")
    parser.add_argument("--adaptive_dropout", type=float, default=0.1, help="自适应模块dropout率")
    
    # 日志和保存参数
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument("--model_save_name", type=str, default="./sft_model", help="模型保存名称")
    
    args = parser.parse_args()

    # 模型配置，这里使用之前章节的配置
    lm_config = {
        "dim": 1024,
        "n_layers": 18,
        "n_heads": 16,
        "n_kv_heads": 8,
        "vocab_size": 6144,
        "multiple_of": 64,
        "dropout": 0.0,
        "flash_attn": True,
        "norm_eps": 1e-5,
        "max_seq_len": 512,
    }

    # 设置数据类型
    dtype_map = {'float32': torch.float32, 'float16': torch.float16, 'bfloat16': torch.bfloat16}
    dtype = dtype_map[args.dtype]
    
    # 设置混合精度训练的上下文
    ctx = torch.amp.autocast(device_type='cuda', dtype=dtype) if 'cuda' in args.device else nullcontext()

    # SwanLab初始化
    if args.use_swanlab:
        swanlab.init(
            project="happy-llm-adaptive-weighting",
            experiment_name=f"sft_adaptive_w{int(args.use_adaptive_weighting)}",
            config={
                "model": "Transformer with Adaptive Weighting",
                "dataset": args.dataset,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "use_adaptive_weighting": args.use_adaptive_weighting,
                "weight_momentum": args.weight_momentum,
                "min_weight": args.min_weight,
                "max_weight": args.max_weight
            }
        )

    # 初始化模型和分词器
    model, tokenizer = init_model()

    # 创建数据集
    train_ds = SFTDataset(args.dataset, tokenizer, max_length=512)
    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=True,
        num_workers=args.num_workers
    )

    # 计算迭代相关信息
    iter_per_epoch = len(train_loader)
    Logger(f'总迭代次数：{iter_per_epoch}')

    # 初始化优化器和混合精度训练的梯度缩放器
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == torch.float16))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), eps=1e-5)

    # 开始训练
    Logger("开始训练...")
    if args.use_adaptive_weighting:
        Logger("使用自适应加权模块进行训练")
    
    for epoch in range(args.epochs):
        model.train()
        train_epoch(epoch)
        
        # 保存每个epoch结束时的模型
        model_save_path = f"{args.model_save_name}_epoch_{epoch+1}_final.pth"
        if hasattr(model, 'module'):  # DataParallel情况
            torch.save(model.module.state_dict(), model_save_path)
        else:
            torch.save(model.state_dict(), model_save_path)
        Logger(f'Epoch {epoch+1} model saved as {model_save_path}')

    Logger("训练完成!")
    
    # 最终权重统计
    if args.use_adaptive_weighting:
        final_stats = model.get_adaptive_weight_statistics()
        if hasattr(model, 'module'):  # DataParallel情况
            final_stats = model.module.get_adaptive_weight_statistics()
        Logger(f"最终自适应权重统计: {final_stats}")

    if args.use_swanlab:
        swanlab.finish()