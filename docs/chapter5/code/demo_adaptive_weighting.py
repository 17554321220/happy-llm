#!/usr/bin/env python3
"""
自适应加权模块集成测试脚本

该脚本演示如何在LLaMA模型训练中使用自适应加权模块。
"""

import os
import sys
import torch
import torch.nn.functional as F
import json
from transformers import AutoTokenizer

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from k_model import ModelConfig, Transformer
from dataset import SFTDataset
from adaptive_weighting import create_adaptive_weighting_module


def create_mock_data(num_samples=20, seq_len=64, vocab_size=1000):
    """创建模拟训练数据"""
    
    # 创建模拟的多轮对话数据
    conversations = []
    for i in range(num_samples):
        # 模拟不同质量的对话数据
        if i < 10:  # 前10个是高质量数据
            conversation = {
                "conversation": [
                    {"from": "human", "value": f"这是一个高质量的问题 {i}"},
                    {"from": "gpt", "value": f"这是一个详细和准确的回答 {i}"}
                ]
            }
        else:  # 后10个是噪声数据
            conversation = {
                "conversation": [
                    {"from": "human", "value": f"噪声问题 {i} ??? @#$%"},
                    {"from": "gpt", "value": f"低质量回答 {i} ..."}
                ]
            }
        conversations.append(conversation)
    
    # 保存为JSONL格式
    data_file = "/tmp/mock_sft_data.jsonl"
    with open(data_file, 'w', encoding='utf-8') as f:
        for conv in conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + '\n')
    
    return data_file


def demo_adaptive_weighting():
    """演示自适应加权模块的使用"""
    print("="*60)
    print("自适应加权模块集成测试演示")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 1. 创建模拟数据
    print("\n1. 创建模拟训练数据...")
    data_file = create_mock_data()
    print(f"   数据文件: {data_file}")
    
    # 2. 初始化模型和分词器
    print("\n2. 初始化模型...")
    lm_config = ModelConfig(
        dim=256,      # 较小的模型用于演示
        n_layers=4,
        n_heads=8,
        vocab_size=1000,
        max_seq_len=64
    )
    
    model = Transformer(lm_config).to(device)
    print(f"   模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # 由于没有预训练的tokenizer，使用简单的mock tokenizer
    # 在实际使用中应该使用AutoTokenizer.from_pretrained()
    class MockTokenizer:
        def __init__(self):
            self.bos_token = "<|im_start|>"
            self.eos_token = "<|im_end|>"
            self.eos_token_id = 4
        
        def __call__(self, text):
            # 简单的mock实现
            class Result:
                def __init__(self):
                    import random
                    self.data = {'input_ids': [random.randint(1, 999) for _ in range(min(32, len(text)//10+5))]}
                    
                def __getitem__(self, key):
                    if key == 'input_ids':
                        return self.data['input_ids']
                    raise KeyError(key)
            return Result()
        
        def apply_chat_template(self, conversation, tokenize=False, add_generation_prompt=False):
            # 简化的模板应用
            text = ""
            for turn in conversation["conversation"]:
                text += f"<|im_start|>{turn['from']}\n{turn['value']}<|im_end|>\n"
            return text
    
    tokenizer = MockTokenizer()
    
    # 3. 初始化自适应加权模块
    print("\n3. 初始化自适应加权模块...")
    adaptive_config = {
        'weight_net_hidden_dim': 32,
        'ema_decay': 0.9,
        'min_weight': 0.1,
        'device': device,
        'update_frequency': 5
    }
    adaptive_module = create_adaptive_weighting_module(adaptive_config)
    print("   自适应加权模块已创建")
    
    # 4. 创建模拟数据
    print("\n4. 创建模拟数据...")
    batch_size = 4
    seq_len = 16
    vocab_size = lm_config.vocab_size
    
    # 创建模拟的输入数据
    X = torch.randint(1, vocab_size, (batch_size, seq_len)).to(device)
    Y = torch.randint(1, vocab_size, (batch_size, seq_len)).to(device) 
    loss_mask = torch.ones(batch_size, seq_len).to(device)
    
    print(f"   输入形状: {X.shape}")
    print(f"   目标形状: {Y.shape}")
    print(f"   掩码形状: {loss_mask.shape}")
    
    # 5. 模拟训练过程
    print("\n5. 模拟训练过程...")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    total_loss_std = 0
    total_loss_adaptive = 0
    
    for epoch in range(3):
        print(f"\nEpoch {epoch+1}:")
        
        for step in range(5):  # 5个训练步骤
            optimizer.zero_grad()
            
            # 前向传播 - 需要传入targets以获得完整的logits
            output = model(X, Y)
            logits = output.logits
            
            if step == 0:  # 第一步显示调试信息
                print(f"    Debug - Step {step}:")
                print(f"      X shape: {X.shape}")
                print(f"      Y shape: {Y.shape}")
                print(f"      logits shape: {logits.shape}")
            
            # 计算标准损失
            std_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1), reduction='mean')
            
            # 计算自适应加权损失
            sample_ids = [step + epoch * 10 + i for i in range(batch_size)]  # 模拟样本ID
            adaptive_loss, weights = adaptive_module.compute_weighted_loss(
                logits, Y, loss_mask, sample_ids
            )
            
            # 使用自适应损失进行反向传播
            adaptive_loss.backward()
            optimizer.step()
            
            total_loss_std += std_loss.item()
            total_loss_adaptive += adaptive_loss.item()
            
            if step < 2:  # 只显示前2步的详细信息
                print(f"  Step {step}: std_loss={std_loss.item():.4f}, "
                      f"adaptive_loss={adaptive_loss.item():.4f}, "
                      f"weights=[{weights.min().item():.3f}, {weights.max().item():.3f}]")
    
    # 6. 显示结果统计
    print(f"\n6. 训练结果统计:")
    print(f"   平均标准损失: {total_loss_std / 15:.4f}")  # 3 epochs * 5 steps
    print(f"   平均自适应损失: {total_loss_adaptive / 15:.4f}")
    print(f"   损失改善比率: {(1 - total_loss_adaptive/total_loss_std)*100:.1f}%")
    
    # 7. 显示样本统计信息
    print(f"\n7. 样本统计信息:")
    sample_stats = adaptive_module.get_sample_statistics()
    if sample_stats:
        sorted_samples = sorted(sample_stats.items(), key=lambda x: x[1]['contribution'], reverse=True)
        print("   样本贡献度排序:")
        for sample_id, stats in sorted_samples:
            print(f"     样本{sample_id}: 贡献度={stats['contribution']:.3f}, "
                  f"平均损失={stats['loss_ema']:.3f}, 训练次数={stats['count']}")
    
    print("\n" + "="*60)
    print("自适应加权模块演示完成！")
    print("="*60)
    
    # 清理临时文件
    if os.path.exists(data_file):
        os.remove(data_file)


if __name__ == "__main__":
    demo_adaptive_weighting()