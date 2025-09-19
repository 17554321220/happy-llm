"""
带自适应加权的监督微调训练脚本
集成了AdaptiveWeightingModule来缓解多组学数据的异质性噪声并提升模型鲁棒性
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
from adaptive_weighting import AdaptiveWeightingModule, create_adaptive_weighting_module

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

def compute_individual_losses(model, X, Y, ctx):
    """
    计算每个样本的个体损失，用于自适应加权
    """
    batch_size = X.size(0)
    individual_losses = []
    
    # 为每个样本单独计算损失
    for i in range(batch_size):
        x_sample = X[i:i+1]  # [1, seq_len]
        y_sample = Y[i:i+1]  # [1, seq_len]
        
        with ctx:
            out = model(x_sample, y_sample)
            # 计算该样本的平均损失
            sample_loss = out.last_loss.mean()
            individual_losses.append(sample_loss.item())
    
    return torch.tensor(individual_losses, device=X.device)

def train_epoch(epoch):
    """训练一个epoch，集成自适应加权"""
    start_time = time.time()
    epoch_weighted_loss = 0.0
    epoch_regular_loss = 0.0
    total_steps = 0
    
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        # 获取学习率并更新优化器
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 计算每个样本的个体损失用于自适应加权
        individual_losses = compute_individual_losses(model, X, Y, ctx)
        
        # 前向传播
        with ctx:
            out = model(X, Y)
            loss = out.last_loss / args.accumulation_steps
            loss_mask = loss_mask.view(-1)
            
            # 计算常规损失（用于对比）
            regular_loss = torch.sum(loss * loss_mask) / loss_mask.sum()
            
            # 使用自适应加权模块计算加权损失
            if args.use_adaptive_weighting:
                # 准备数据用于自适应加权
                logits = out.logits.view(-1, out.logits.size(-1))  # [batch_size * seq_len, vocab_size]
                targets = Y.view(-1)  # [batch_size * seq_len]
                
                # 生成样本ID（在实际应用中，这应该是真实的样本标识符）
                sample_ids = [step * args.batch_size + i for i in range(X.size(0))]
                
                # 计算加权损失
                weighted_loss, sample_weights = adaptive_weighting_module(
                    logits, targets, individual_losses, sample_ids
                )
                
                # 使用加权损失进行反向传播
                final_loss = weighted_loss
                
                # 记录权重统计信息
                weight_stats = adaptive_weighting_module.get_weight_statistics()
                
            else:
                final_loss = regular_loss
                weight_stats = {}
                sample_weights = torch.ones(X.size(0), device=X.device)

        # 反向传播
        scaler.scale(final_loss).backward()

        # 记录损失
        epoch_weighted_loss += final_loss.item()
        epoch_regular_loss += regular_loss.item()
        total_steps += 1

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
            
            if args.use_adaptive_weighting:
                log_message = (
                    'Epoch:[{}/{}]({}/{}) weighted_loss:{:.3f} regular_loss:{:.3f} lr:{:.7f} '
                    'mean_weight:{:.3f} std_weight:{:.3f} epoch_Time:{}min:'
                ).format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    final_loss.item() * args.accumulation_steps,
                    regular_loss.item() * args.accumulation_steps,
                    optimizer.param_groups[-1]['lr'],
                    weight_stats.get('mean_weight', 0.0),
                    weight_stats.get('std_weight', 0.0),
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60
                )
            else:
                log_message = (
                    'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.7f} epoch_Time:{}min:'
                ).format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    final_loss.item() * args.accumulation_steps,
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60
                )
            
            Logger(log_message)
            
            if args.use_swanlab:
                log_dict = {
                    "weighted_loss" if args.use_adaptive_weighting else "loss": 
                        final_loss.item() * args.accumulation_steps,
                    "lr": optimizer.param_groups[-1]['lr']
                }
                
                if args.use_adaptive_weighting:
                    log_dict.update({
                        "regular_loss": regular_loss.item() * args.accumulation_steps,
                        **{f"weight_{k}": v for k, v in weight_stats.items()}
                    })
                
                swanlab.log(log_dict)

        # 保存模型
        if (step + 1) % args.save_interval == 0:
            model.eval()
            ckp = f'{args.save_dir}/sft_weighted_dim{lm_config.dim}_layers{lm_config.n_layers}_vocab_size{lm_config.vocab_size}.pth'

            # 处理多卡保存
            state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
            torch.save(state_dict, ckp)
            model.train()
        
        # 定期保存模型
        if (step + 1) % 20000 == 0:
            model.eval()
            ckp = f'{args.save_dir}/sft_weighted_dim{lm_config.dim}_layers{lm_config.n_layers}_vocab_size{lm_config.vocab_size}_step{step+1}.pth'

            state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
            torch.save(state_dict, ckp)
            model.train()
    
    # 在epoch结束时更新自适应加权模块
    if args.use_adaptive_weighting:
        adaptive_weighting_module.update_epoch()
    
    # 记录epoch统计信息
    avg_weighted_loss = epoch_weighted_loss / total_steps
    avg_regular_loss = epoch_regular_loss / total_steps
    
    Logger(f"Epoch {epoch + 1} completed:")
    Logger(f"  Average weighted loss: {avg_weighted_loss:.4f}")
    Logger(f"  Average regular loss: {avg_regular_loss:.4f}")
    
    if args.use_adaptive_weighting:
        final_weight_stats = adaptive_weighting_module.get_weight_statistics()
        Logger(f"  Final weight statistics: {final_weight_stats}")


def init_model():
    """初始化模型"""
    def count_parameters(model):
        """计算模型参数量"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained('./tokenizer_k/')

    # 初始化模型
    model = Transformer(lm_config)

    # 加载预训练权重
    ckp = './base_model_215M/pretrain_1024_18_6144.pth'
    if os.path.exists(ckp):
        state_dict = torch.load(ckp, map_location=args.device)
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict, strict=False)
        Logger("预训练权重加载成功")
    else:
        Logger("预训练权重文件不存在，使用随机初始化")
    
    # 多卡初始化
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        Logger(f"Using {num_gpus} GPUs with DataParallel!")
        model = torch.nn.DataParallel(model)
    
    model = model.to(args.device)
    Logger(f'LLM总参数量：{count_parameters(model) / 1e6:.3f} 百万')
    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adaptive Weighted SFT Training")
    parser.add_argument("--out_dir", type=str, default="sft_weighted_model_215M", help="输出目录")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="批处理大小（为了演示自适应加权，使用较小批次）")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="使用的设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="数据类型")
    parser.add_argument("--use_swanlab", action="store_true", help="是否使用SwanLab进行实验跟踪")
    parser.add_argument("--num_workers", type=int, default=2, help="数据加载的工作进程数")
    parser.add_argument("--data_path", type=str, default="./BelleGroup_sft.jsonl", help="训练数据路径")
    parser.add_argument("--accumulation_steps", type=int, default=2, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--warmup_iters", type=int, default=0, help="预热迭代次数")
    parser.add_argument("--log_interval", type=int, default=10, help="日志记录间隔")
    parser.add_argument("--save_interval", type=int, default=500, help="模型保存间隔")
    
    # 自适应加权相关参数
    parser.add_argument("--use_adaptive_weighting", action="store_true", help="是否使用自适应加权模块")
    parser.add_argument("--weight_hidden_dim", type=int, default=64, help="权重网络隐藏层维度")
    parser.add_argument("--min_weight", type=float, default=0.1, help="最小样本权重")
    parser.add_argument("--max_weight", type=float, default=1.0, help="最大样本权重")
    parser.add_argument("--weight_update_freq", type=int, default=1, help="权重更新频率")
    
    # 多卡参数
    parser.add_argument("--gpus", type=str, default='0', help="逗号分隔的GPU ID (例如 '0,1,2')")

    args = parser.parse_args()

    # 设置可见GPU
    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        # 自动设置主设备为第一个GPU
        if torch.cuda.is_available():
            args.device = "cuda:0"
        else:
            args.device = "cpu"

    # 初始化swanlab
    if args.use_swanlab:
        run = swanlab.init(
            project="Happy-LLM-Weighted",
            experiment_name="SFT-215M-Adaptive-Weighted",
            config=args,
        )

    # 模型配置
    lm_config = ModelConfig(
        dim=1024,
        n_layers=18,
    )
    max_seq_len = lm_config.max_seq_len
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(42)
    device_type = "cuda" if "cuda" in args.device else "cpu"

    # 上下文管理器
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()

    # 初始化模型和分词器
    model, tokenizer = init_model()
    
    # 初始化自适应加权模块
    if args.use_adaptive_weighting:
        weighting_config = {
            'hidden_dim': args.weight_hidden_dim,
            'update_frequency': args.weight_update_freq,
            'min_weight': args.min_weight,
            'max_weight': args.max_weight,
            'device': args.device
        }
        
        adaptive_weighting_module = create_adaptive_weighting_module(weighting_config)
        Logger("自适应加权模块初始化完成")
        Logger(f"配置: {weighting_config}")
    else:
        adaptive_weighting_module = None
        Logger("使用常规训练模式（不启用自适应加权）")
    
    # 创建数据集和数据加载器
    data_path = args.data_path
    if not os.path.exists(data_path):
        # 如果指定的数据文件不存在，创建一个简单的示例数据
        Logger(f"数据文件 {data_path} 不存在，创建示例数据用于演示")
        import json
        sample_data = [
            {
                "conversations": [
                    {"from": "human", "value": "你好，请介绍一下自适应加权在深度学习中的应用。"},
                    {"from": "assistant", "value": "自适应加权是一种重要的深度学习技术，它可以根据样本的重要性动态调整损失函数中各个样本的权重。这种技术特别适用于处理数据不平衡、噪声样本和提升模型鲁棒性等场景。"}
                ]
            },
            {
                "conversations": [
                    {"from": "human", "value": "什么是多组学数据？"},
                    {"from": "assistant", "value": "多组学数据是指整合了基因组学、转录组学、蛋白质组学、代谢组学等多个生物学层面数据的综合数据集。这种数据具有高维度、异质性强、噪声多等特点，需要特殊的处理方法。"}
                ]
            },
            {
                "conversations": [
                    {"from": "human", "value": "联邦学习中如何保护数据隐私？"},
                    {"from": "assistant", "value": "联邦学习通过多种技术保护数据隐私，包括：1) 数据本地化，不直接共享原始数据；2) 同态加密，对模型参数进行加密传输；3) 差分隐私，在参数中添加噪声；4) 安全多方计算等。"}
                ]
            },
            {
                "conversations": [
                    {"from": "human", "value": "如何评估模型的鲁棒性？"},
                    {"from": "assistant", "value": "模型鲁棒性可以从多个角度评估：1) 对抗鲁棒性，测试模型对对抗样本的抵抗能力；2) 噪声鲁棒性，评估在有噪声数据上的性能；3) 泛化鲁棒性，在不同分布数据上的表现；4) 参数鲁棒性，对参数变化的敏感性。"}
                ]
            }
        ]
        
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        with open(data_path, 'w', encoding='utf-8') as f:
            for item in sample_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        Logger(f"示例数据已创建：{data_path}")
    
    train_ds = SFTDataset(data_path, tokenizer, max_length=max_seq_len)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=True,
        num_workers=args.num_workers
    )

    # 缩放器和优化器
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    
    # 如果使用自适应加权，同时优化主模型和权重网络
    if args.use_adaptive_weighting:
        # 分别为主模型和权重网络设置优化器
        main_params = list(model.parameters())
        weight_params = list(adaptive_weighting_module.parameters())
        all_params = main_params + weight_params
        optimizer = optim.AdamW(all_params, lr=args.learning_rate)
        Logger(f"优化器配置: 主模型参数 {len(main_params)}, 权重网络参数 {len(weight_params)}")
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # 开始训练
    iter_per_epoch = len(train_loader)
    Logger(f"开始训练，每个epoch有 {iter_per_epoch} 个批次")
    Logger(f"自适应加权模式: {'启用' if args.use_adaptive_weighting else '禁用'}")
    
    for epoch in range(args.epochs):
        Logger(f"开始训练 Epoch {epoch + 1}/{args.epochs}")
        train_epoch(epoch)
        Logger(f"Epoch {epoch + 1} 训练完成")
    
    Logger("训练完成!")
    
    # 保存最终模型
    model.eval()
    final_ckp = f'{args.save_dir}/final_sft_weighted_model.pth'
    state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
    torch.save(state_dict, final_ckp)
    Logger(f"最终模型已保存: {final_ckp}")
    
    # 如果使用自适应加权，也保存权重模块
    if args.use_adaptive_weighting:
        weight_module_ckp = f'{args.save_dir}/adaptive_weighting_module.pth'
        torch.save(adaptive_weighting_module.state_dict(), weight_module_ckp)
        Logger(f"自适应加权模块已保存: {weight_module_ckp}")