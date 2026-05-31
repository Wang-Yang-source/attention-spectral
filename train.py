#!/usr/bin/env python3
"""
训练脚本 — 模算术任务

任务: 给定 a, b (0-99), 预测 (a + b) % 100
序列: "<a> <op> <b> = ?" (5个 token)

每 50 步存一个 checkpoint，包括模型权重 + 固定的 probe 输入
"""
import torch
import torch.nn as nn
import argparse
import os
import json
from model import TinyTransformer, TinyTransformerConfig

def make_data(batch_size, seq_len, device):
    """生成模加法数据: a + b = ?"""
    a = torch.randint(0, 100, (batch_size,), device=device)
    b = torch.randint(0, 100, (batch_size,), device=device)
    result = (a + b) % 100

    x = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    y = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)

    x[:, 0] = a + 1        # token 1-100 = 数字 0-99
    x[:, 1] = 101           # op token
    x[:, 2] = b + 1
    x[:, 3] = 102           # = token
    x[:, 4] = 103           # ? token (占位)
    y[:, 4] = result + 1    # 只在 ? 位置监督

    return x, y

def make_probe_inputs(device, seq_len=10):
    """固定 probe 输入集合 — 每个 checkpoint 都用同一批输入提取注意力"""
    probes = {}
    torch.manual_seed(42)

    # 1. 随机 token
    probes["random"] = torch.randint(1, 100, (1, seq_len), device=device)

    # 2. 同一 token 重复
    probes["uniform"] = torch.full((1, seq_len), 50, dtype=torch.long, device=device)

    # 3. 递增长序列
    probes["ascending"] = torch.tensor([[i % 100 + 1 for i in range(seq_len)]], device=device)

    # 4. 标准模加法输入
    a = torch.tensor([[37]], device=device)
    b = torch.tensor([[42]], device=device)
    result = (37 + 42) % 100
    x = torch.zeros(1, 5, dtype=torch.long, device=device)
    x[:, 0] = 38; x[:, 1] = 101; x[:, 2] = 43; x[:, 3] = 102; x[:, 4] = 103
    # pad to seq_len
    padded = 103 * torch.ones(1, seq_len, dtype=torch.long, device=device)
    padded[:, :5] = x
    probes["mod_add"] = padded

    return probes

def train(args):
    device = torch.device("cpu")
    torch.manual_seed(args.seed)

    config = TinyTransformerConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_seq_len=args.seq_len,
        vocab_size=args.vocab_size,
    )

    model = TinyTransformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.steps)

    os.makedirs(args.out_dir, exist_ok=True)

    # 保存 config
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f)

    # 固定 probe 输入
    probes = make_probe_inputs(device, args.seq_len)

    metrics = []
    for step in range(args.steps):
        model.train()
        x, y = make_data(args.batch_size, args.seq_len, device)

        logits, _ = model(x)
        # Only compute loss on the prediction position (index 4 = '?')
        loss = nn.CrossEntropyLoss()(logits[:, 4, :], y[:, 4])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        metrics.append({"step": step, "loss": loss.item()})

        # 每 50 步或最后一步存 checkpoint
        if step % args.ckpt_every == 0 or step == args.steps - 1:
            ckpt = {
                "step": step,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "metrics": metrics,
                "probes": probes,
            }
            path = os.path.join(args.out_dir, f"ckpt_step_{step:06d}.pt")
            torch.save(ckpt, path)

        if step % 200 == 0:
            print(f"  Step {step:5d}/{args.steps} | loss: {loss.item():.4f} | lr: {scheduler.get_last_lr()[0]:.6f}")

    print(f"✅ 训练完成！{len(os.listdir(args.out_dir))} 个 checkpoint 保存在 {args.out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--seq_len", type=int, default=10)
    parser.add_argument("--vocab_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt_every", type=int, default=50)
    parser.add_argument("--out_dir", type=str, default="./checkpoints")
    args = parser.parse_args()
    train(args)
