#!/usr/bin/env python3
"""
微型 Transformer — 专门为注意力机制研究设计

2层, d_model=128, 4头, 约 0.5M 参数
CPU 上训练 10 分钟, checkpoint 每 50 步存一次
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass

@dataclass
class TinyTransformerConfig:
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 512
    max_seq_len: int = 64
    dropout: float = 0.0   # 研究注意力 → 关 dropout
    vocab_size: int = 128  # 模算术任务用的 token 数

class Attention(nn.Module):
    """单头注意力 — 保留 attention_weights 用于后续分析"""
    def __init__(self, d_model, d_head):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.W_q = nn.Linear(d_model, d_head, bias=False)
        self.W_k = nn.Linear(d_model, d_head, bias=False)
        self.W_v = nn.Linear(d_model, d_head, bias=False)
        self.W_o = nn.Linear(d_head, d_model, bias=False)
        self.attn_weights = None  # 每次 forward 时存储

    def forward(self, x):
        B, T, D = x.shape
        q = self.W_q(x)  # (B, T, d_head)
        k = self.W_k(x)
        v = self.W_v(x)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn = F.softmax(scores, dim=-1)
        self.attn_weights = attn.detach().clone()  # 保存

        out = attn @ v
        return self.W_o(out)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        d_head = d_model // n_heads
        self.heads = nn.ModuleList([Attention(d_model, d_head) for _ in range(n_heads)])
        self.n_heads = n_heads
        self.d_model = d_model

    def forward(self, x):
        head_outs = [h(x) for h in self.heads]
        return sum(head_outs), [h.attn_weights for h in self.heads]

class TransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        attn_out, attn_weights = self.attn(self.ln1(x))
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, attn_weights

class TinyTransformer(nn.Module):
    def __init__(self, config: TinyTransformerConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, config.max_seq_len, config.d_model) * 0.02)
        self.layers = nn.ModuleList([
            TransformerLayer(config.d_model, config.n_heads, config.d_ff)
            for _ in range(config.n_layers)
        ])
        self.ln_final = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, x):
        B, T = x.shape
        h = self.embed(x) + self.pos_embed[:, :T, :]
        all_attn_weights = []
        for layer in self.layers:
            h, attn_weights = layer(h)
            all_attn_weights.append(attn_weights)  # list of list: [layer][head] -> (B,T,T)
        h = self.ln_final(h)
        logits = self.head(h)
        return logits, all_attn_weights

# ---- 工具函数 ----
def extract_attention_matrices(model, x):
    """
    给定输入 x (1, T), 返回所有层 x 所有头的注意力矩阵
    形状: [(n_heads, T, T), ...]  每层一组
    """
    model.eval()
    with torch.no_grad():
        _, attn = model(x)
    result = []
    for layer_heads in attn:
        heads_tensor = torch.stack(layer_heads, dim=0).squeeze(1)  # (n_heads, T, T)
        result.append(heads_tensor)
    return result  # list of (n_heads, T, T)
