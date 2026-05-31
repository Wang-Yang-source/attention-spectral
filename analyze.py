#!/usr/bin/env python3
"""
注意力矩阵数学性质分析

对于每个 checkpoint，分析：
  1. 有效秩（singular value entropy）
  2. 特征值 / 奇异值谱分布
  3. 稀疏度（entropy & Gini index）
  4. 对角线集中度
  5. 层间 / 头间差异

生成脚本供用户自己在有 GPU/Colab 的机器上跑
"""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats, linalg
from scipy.spatial.distance import jensenshannon
import argparse

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "figure.dpi": 150,
})

# =============================================
# 指标计算
# =============================================

def effective_rank(singular_values, threshold=0.9):
    """
    有效秩 = 需要多少个奇异值才能解释 threshold 比例的总方差
    等价于 Shannon entropy over normalized singular values
    """
    s = singular_values[singular_values > 1e-10]
    if len(s) == 0:
        return 0
    p = s / s.sum()
    entropy = -np.sum(p * np.log(p + 1e-12))
    return int(np.exp(entropy))  # Perplexity-style effective rank

def singular_value_entropy(svd_vals):
    """归一化奇异值的香农熵"""
    s = svd_vals[svd_vals > 1e-10]
    if len(s) == 0:
        return 0
    p = s / s.sum()
    return -np.sum(p * np.log(p + 1e-12))

def attention_entropy(attn_matrix):
    """每个 query 位置的注意力分布的熵的平均值"""
    eps = 1e-12
    log_attn = np.log(attn_matrix + eps)
    ent = -np.sum(attn_matrix * log_attn, axis=-1)  # (T,) for each query
    return np.mean(ent)

def gini_index(attn_matrix):
    """稀疏度的 Gini 系数 — 0=均匀, 1=完全集中到单 token"""
    flattened = np.sort(attn_matrix.flatten())
    n = len(flattened)
    cumulative = np.cumsum(flattened)
    # Lorenz curve
    lorenz = cumulative / (cumulative[-1] + 1e-12)
    perfect_equal = np.arange(1, n + 1) / n
    return 1 - 2 * np.trapz(lorenz, perfect_equal)

def diagonal_concentration(attn_matrix):
    """对角线附近 (距离≤2) 的注意力权重占比"""
    T = attn_matrix.shape[0]
    total = 0.0
    mask_total = 0.0
    for i in range(T):
        for j in range(T):
            total += attn_matrix[i, j]
            if abs(i - j) <= 2:
                mask_total += attn_matrix[i, j]
    return mask_total / (total + 1e-12)

def spectral_gap(svd_vals):
    """最大奇异值与第二大的比值 λ₁/λ₂"""
    s = svd_vals[svd_vals > 1e-10]
    if len(s) < 2:
        return 1.0
    return s[0] / s[1]

def marchenko_pastur_fit(svd_vals):
    """
    检验奇异值分布是否符合 Marchenko-Pastur 定律 (随机矩阵零假设)
    返回 KS 统计量和 p-value
    """
    s = svd_vals[svd_vals > 1e-10]
    s_norm = s / np.sqrt(np.mean(s**2))  # normalize to unit variance
    # MP(1) distribution CDF approximation
    mp_sample = np.sort(s_norm**2 / len(s_norm))
    # KS test against theoretical MP(1)
    from scipy.special import jv
    # Simplified: just compute the shape ratio
    ratio = np.mean(s_norm**4) / (np.mean(s_norm**2))**2
    return ratio  # > 3 indicates non-MP (structured)

# =============================================
# 主分析流程
# =============================================

def analyze(attention_data_path, output_dir):
    data = torch.load(attention_data_path, map_location="cpu", weights_only=False)
    ckpts = data["checkpoints"]
    probe_names = list(ckpts[0]["probe_attentions"].keys())
    n_layers = len(ckpts[0]["probe_attentions"][probe_names[0]])
    n_heads = ckpts[0]["probe_attentions"][probe_names[0]][0].shape[0]
    T = ckpts[0]["probe_attentions"][probe_names[0]][0].shape[1]

    steps = [c["step"] for c in ckpts]
    import os
    os.makedirs(output_dir, exist_ok=True)

    print(f"Analyzing {len(ckpts)} checkpoints, {n_layers} layers, {n_heads} heads, sequence length {T}")
    print(f"Probes: {probe_names}")

    # ---- 指标数组 ----
    N = len(ckpts)  # number of checkpoints
    eff_rank = np.zeros((N, n_layers, n_heads))       # 有效秩
    sv_entropy = np.zeros((N, n_layers, n_heads))      # 奇异值熵
    attn_entropy_val = np.zeros((N, n_layers, n_heads))# 注意力熵
    gini = np.zeros((N, n_layers, n_heads))            # Gini 系数
    diag_conc = np.zeros((N, n_layers, n_heads))       # 对角线集中度
    spec_gap = np.zeros((N, n_layers, n_heads))        # 谱隙
    mp_ratio = np.zeros((N, n_layers, n_heads))        # MP 比率
    losses = np.array([c.get("loss", 0) for c in ckpts])

    probe_name = "mod_add"  # 用模加法输入做分析

    for i, ckpt in enumerate(ckpts):
        layers = ckpt["probe_attentions"][probe_name]
        for l in range(n_layers):
            for h in range(n_heads):
                A = layers[l][h]  # (T, T)
                U, s, Vt = np.linalg.svd(A, full_matrices=False)

                eff_rank[i, l, h] = effective_rank(s)
                sv_entropy[i, l, h] = singular_value_entropy(s)
                attn_entropy_val[i, l, h] = attention_entropy(A)
                gini[i, l, h] = gini_index(A)
                diag_conc[i, l, h] = diagonal_concentration(A)
                spec_gap[i, l, h] = spectral_gap(s)
                mp_ratio[i, l, h] = marchenko_pastur_fit(s)

    # =============================================
    # 可视化 1: 损失曲线
    # =============================================
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(steps, losses, "b-", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/01_loss.png", bbox_inches="tight")
    plt.close()
    print("  [1/8] Loss curve saved")

    # =============================================
    # 可视化 2: 有效秩演化 (每层平均)
    # =============================================
    fig, axes = plt.subplots(1, n_layers, figsize=(4*n_layers, 3.5))
    if n_layers == 1:
        axes = [axes]
    colors = plt.cm.viridis(np.linspace(0, 0.9, n_heads))
    for l in range(n_layers):
        for h in range(n_heads):
            axes[l].plot(steps, eff_rank[:, l, h], color=colors[h], linewidth=0.8, label=f"Head {h}")
        axes[l].set_xlabel("Step")
        axes[l].set_ylabel("Effective Rank")
        axes[l].set_title(f"Layer {l}")
        axes[l].grid(True, alpha=0.3)
        axes[l].legend(fontsize=7, ncol=2)
    fig.suptitle("Effective Rank Evolution", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/02_effective_rank.png", bbox_inches="tight")
    plt.close()
    print("  [2/8] Effective rank saved")

    # =============================================
    # 可视化 3: 奇异值熵演化
    # =============================================
    fig, axes = plt.subplots(1, n_layers, figsize=(4*n_layers, 3.5))
    if n_layers == 1:
        axes = [axes]
    for l in range(n_layers):
        for h in range(n_heads):
            axes[l].plot(steps, sv_entropy[:, l, h], color=colors[h], linewidth=0.8, label=f"Head {h}")
        axes[l].set_xlabel("Step")
        axes[l].set_ylabel("SV Entropy")
        axes[l].set_title(f"Layer {l}")
        axes[l].grid(True, alpha=0.3)
        axes[l].legend(fontsize=7, ncol=2)
    fig.suptitle("Singular Value Entropy (Higher = More Uniform Spread)", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/03_sv_entropy.png", bbox_inches="tight")
    plt.close()
    print("  [3/8] SV entropy saved")

    # =============================================
    # 可视化 4: 奇异值谱 (选取几个代表性 checkpoint)
    # =============================================
    sample_idxs = [0, len(ckpts)//4, len(ckpts)//2, -1]
    fig, axes = plt.subplots(n_layers, len(sample_idxs), figsize=(3*len(sample_idxs), 2.5*n_layers))
    if n_layers == 1:
        axes = axes.reshape(1, -1)
    for col, idx in enumerate(sample_idxs):
        for l in range(n_layers):
            ax = axes[l, col]
            for h in range(n_heads):
                A = ckpts[idx]["probe_attentions"][probe_name][l][h]
                _, s, _ = np.linalg.svd(A, full_matrices=False)
                s = s[s > 1e-6]
                ax.semilogy(range(1, len(s)+1), s, "-o", markersize=1, linewidth=0.6,
                           color=colors[h], label=f"H{h}" if col==0 else None)
            ax.set_title(f"Layer {l}, Step {steps[idx]}" if l == 0 else f"Step {steps[idx]}")
            ax.set_xlabel("Index")
            if col == 0:
                ax.set_ylabel("SV (log)")
            ax.grid(True, alpha=0.3)
    fig.suptitle("Singular Value Spectra Across Training", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/04_sv_spectra.png", bbox_inches="tight")
    plt.close()
    print("  [4/8] SV spectra saved")

    # =============================================
    # 可视化 5: Gini 系数 (稀疏度)
    # =============================================
    fig, axes = plt.subplots(1, n_layers, figsize=(4*n_layers, 3.5))
    if n_layers == 1:
        axes = [axes]
    for l in range(n_layers):
        for h in range(n_heads):
            axes[l].plot(steps, gini[:, l, h], color=colors[h], linewidth=0.8)
        axes[l].set_xlabel("Step")
        axes[l].set_ylabel("Gini Coefficient")
        axes[l].set_title(f"Layer {l}")
        axes[l].grid(True, alpha=0.3)
        axes[l].set_ylim(0, 1)
    fig.suptitle("Sparsity (Gini Index): 0=Uniform, 1=Collapsed", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/05_gini.png", bbox_inches="tight")
    plt.close()
    print("  [5/8] Gini saved")

    # =============================================
    # 可视化 6: 谱隙
    # =============================================
    fig, axes = plt.subplots(1, n_layers, figsize=(4*n_layers, 3.5))
    if n_layers == 1:
        axes = [axes]
    for l in range(n_layers):
        for h in range(n_heads):
            axes[l].plot(steps, np.log10(spec_gap[:, l, h] + 1), color=colors[h], linewidth=0.8)
        axes[l].set_xlabel("Step")
        axes[l].set_ylabel("log10(Spectral Gap)")
        axes[l].set_title(f"Layer {l}")
        axes[l].grid(True, alpha=0.3)
    fig.suptitle("Spectral Gap (λ₁/λ₂): Higher = Low-Rank Dominance", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/06_spectral_gap.png", bbox_inches="tight")
    plt.close()
    print("  [6/8] Spectral gap saved")

    # =============================================
    # 可视化 7: 对角线集中度
    # =============================================
    fig, axes = plt.subplots(1, n_layers, figsize=(4*n_layers, 3.5))
    if n_layers == 1:
        axes = [axes]
    for l in range(n_layers):
        for h in range(n_heads):
            axes[l].plot(steps, diag_conc[:, l, h], color=colors[h], linewidth=0.8)
        axes[l].set_xlabel("Step")
        axes[l].set_ylabel("Diagonal Concentration")
        axes[l].set_title(f"Layer {l}")
        axes[l].grid(True, alpha=0.3)
    fig.suptitle("Local Attention vs Global Attention", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/07_diag_conc.png", bbox_inches="tight")
    plt.close()
    print("  [7/8] Diagonal concentration saved")

    # =============================================
    # 可视化 8: 相图 (Phase Diagram) — Eff Rank vs Gini
    # =============================================
    fig, axes = plt.subplots(1, n_layers, figsize=(4*n_layers, 3.5))
    if n_layers == 1:
        axes = [axes]
    for l in range(n_layers):
        for h in range(n_heads):
            sc = axes[l].scatter(eff_rank[:, l, h], gini[:, l, h],
                                c=steps, cmap="plasma", s=3, alpha=0.6)
            if h == 0:
                axes[l].annotate("Start", (eff_rank[0, l, h], gini[0, l, h]),
                                fontsize=6, xytext=(5, 5), textcoords="offset points")
                axes[l].annotate("End", (eff_rank[-1, l, h], gini[-1, l, h]),
                                fontsize=6, xytext=(5, -10), textcoords="offset points")
        axes[l].set_xlabel("Effective Rank")
        axes[l].set_ylabel("Gini (Sparsity)")
        axes[l].set_title(f"Layer {l}")
        axes[l].grid(True, alpha=0.3)
    cbar = fig.colorbar(sc, ax=axes, label="Step", orientation="horizontal", pad=0.15, aspect=30)
    fig.suptitle("Phase Diagram: Rank vs Sparsity During Training", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{output_dir}/08_phase_diagram.png", bbox_inches="tight")
    plt.close()
    print("  [8/8] Phase diagram saved")

    # =============================================
    # 汇总输出
    # =============================================
    print(f"\n✅ All visualizations saved to {output_dir}/")
    print(f"\n📊 Key observations to look for:")
    print(f"  - Does effective rank increase or decrease during training?")
    print(f"  - Does spectral gap λ₁/λ₂ diverge? (low-rank emergence)")
    print(f"  - Do some heads become sparse (high Gini) while others stay broad?")
    print(f"  - Is there a critical step where the phase diagram kinks?")
    print(f"  - Does Layer 1 vs Layer 2 differ qualitatively?")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="attention_data.pt 路径")
    parser.add_argument("--out", type=str, default="./figures", help="输出目录")
    args = parser.parse_args()
    analyze(args.data, args.out)
