# Attention Matrix Spectral Evolution During Training

## 研究问题

Transformer 的注意力矩阵 A = softmax(QKᵀ/√d) 在训练过程中，其数学性质如何演变？

- **有效秩**：注意力矩阵的低秩程度随训练如何变化？
- **特征值谱**：奇异值分布是否逼近 Marchenko-Pastur 定律？何时偏离？
- **稀疏度**：注意力是越来越集中还是越来越分散？
- **谱隙**：λ₁/λ₂ 是否发散？何时出现主导方向？
- **层间分工**：不同层的注意力退化机制相同吗？

## 快速开始

```bash
# 1. 训练小模型 (CPU ~10分钟)
python train.py --steps 2000 --out_dir ./checkpoints

# 2. 提取注意力矩阵
python extract.py --ckpt_dir ./checkpoints --output attention_data.pt

# 3. 分析 + 可视化
python analyze.py --data attention_data.pt --out ./figures
```

输出: `figures/` 目录下 8 张图。

## 结果解读

| 图 | 看什么 | 如果发现... |
|---|---|---|
| `02_effective_rank.png` | 有效秩变化 | 下降=低秩涌现，上升=表示泛化 |
| `03_sv_entropy.png` | 奇异值熵 | 骤降=相变信号 |
| `04_sv_spectra.png` | 谱形状 | 阶梯状=结构化，平滑=随机 |
| `05_gini.png` | 稀疏度 | →1 = 注意力坍缩到单 token |
| `06_spectral_gap.png` | λ₁/λ₂ | 暴涨=秩-1 结构形成 |
| `08_phase_diagram.png` | 秩 vs 稀疏度 | 弧形轨迹=平滑演化，拐点=临界相变 |

## 可投方向

1. **如果发现 abrupt phase transition** → 定位临界点，分析 loss landscape 在临界点的尖锐度（Hessian eigenvalues）
   - 投: ICML/NeurIPS workshop → TMLR
   - 卖点: "类似于 grokking，但发生在注意力结构的相空间"

2. **如果发现不同层有不同退化模式** → 系统分类：哪层先收敛、哪层负责局部、哪层负责全局
   - 投: EMNLP / ACL Findings
   - 卖点: "注意力结构的层级分工：从随机到结构化"

3. **如果发现随机矩阵理论 (RMT) 能预测谱形状** → 推导解析公式，用实验验证
   - 投: JMLR / AISTATS
   - 卖点: "Attention matrix as random matrix: a phase transition perspective"

4. **如果发现稀疏化是逐渐的而非突变的** → 与 grokking 对比，提出"slow feature learning" 假说
   - 投: ICLR
   - 卖点: "Emergent sparsity without catastrophic change"

## 扩展方向

- 变化序列长度 T → 观察 scaling behavior
- 变化 head 数量 → 看 heads 间的 specialization
- 因果 mask vs full attention → 差异
- 用不同 probe (random vs structured) → 看输入依赖
- 深模型 (12层 Pythia 160M) → 中间层有什么特殊的？

## 引用

如果你用这个得到结果，请考虑引用：
- Marchenko & Pastur (1967)
- Pennington & Bahri (2017) "Geometry of Neural Network Loss Surfaces"
- Power et al. (2022) "Grokking: Generalization Beyond Overfitting"
