# 图表登记表（Figure Registry）

所有**概念架构与原理图（Fig.1–Fig.6）**由 `scripts/render_diagrams.py` 生成；所有**实证数据图（Fig.7–Fig.16 及补充图）**严格由 `scripts/analyze.py` 从 `results/raw/` 计算并调用 `src/visualization/benchmark.py` 在 300 DPI 下生成；**选型决策树（Fig.17）**动态读取 `results/tables/` 实测统计量。仓库与 PPT 中不出现任何模拟或手填结果数字。

| 编号 | 图表名称 | 文件路径 | 生成来源 | 所在幻灯片 |
| --- | --- | --- | --- | --- |
| Fig.1 | 深度聚类范式演进谱系图 | `presentation/assets/diag_paradigm_evolution.png` | `scripts/render_diagrams.py`（文献综合） | 第 6 页 |
| Fig.2 | 四维方法分类框架图 | `presentation/assets/diag_taxonomy_4d.png` | `scripts/render_diagrams.py`（理论整理） | 第 7 页 |
| Fig.3 | DEC 与 IDEC 原理对照架构图 | `presentation/assets/diag_dec_idec.png` | `scripts/render_diagrams.py`（数学原理） | 第 9 页 |
| Fig.4 | VaDE 深度生成式聚类架构图 | `presentation/assets/diag_vade_arch.png` | `scripts/render_diagrams.py`（数学原理） | 第 10 页 |
| Fig.5 | SimCLR 与 CC 双视图对比原理图 | `presentation/assets/diag_contrastive_arch.png` | `scripts/render_diagrams.py`（数学原理） | 第 11 页 |
| Fig.6 | 预训练特征与聚类器解耦框架图 | `presentation/assets/diag_pretrained_decouple.png` | `scripts/render_diagrams.py`（E04 设计） | 第 12 页 |
| Fig.7 | E01 多数据集 9 方法 ARI 热力图 | `results/figures/fig_e01_ari.png` | `scripts/analyze.py` ← `e01_summary.csv` | 第 15 页 |
| Fig.8 | E01 潜在空间二维 t-SNE 投影图 | `results/figures/fig_e01_embeddings.png` | `scripts/analyze.py` ← `results/raw/embeddings/` | 第 16 页 |
| Fig.9 | E02 重构权重 λ 双指标消融曲线 | `results/figures/fig_e02_lambda.png` | `scripts/analyze.py` ← `e02_ablation.csv` | 第 17 页 |
| Fig.10 | E02 隐空间维度 d 双指标消融曲线 | `results/figures/fig_e02_latent.png` | `scripts/analyze.py` ← `e02_ablation.csv` | 附录 A2（第 26 页） |
| Fig.10b | E02 初始化与微调组件消融条形图 | `results/figures/fig_e02_components.png` | `scripts/analyze.py` ← `e02_ablation.csv` | 附录 A2（第 26 页） |
| Fig.11 | E03 测试期高斯噪声鲁棒性曲线 | `results/figures/fig_e03_noise.png` | `scripts/analyze.py` ← `e03_noise.csv` | 第 18 页 |
| Fig.12 | E03 类别长尾不平衡敏感性曲线 | `results/figures/fig_e03_imbalance.png` | `scripts/analyze.py` ← `e03_sensitivity.csv` | 附录 A3（第 27 页） |
| Fig.13 | E03 训练样本规模效率曲线 | `results/figures/fig_e03_sample_frac.png` | `scripts/analyze.py` ← `e03_sensitivity.csv` | 附录 A3（第 27 页） |
| Fig.14 | E03 预设聚类数 K 错配敏感性曲线 | `results/figures/fig_e03_k.png` | `scripts/analyze.py` ← `e03_sensitivity.csv` | 附录 A4（第 28 页） |
| Fig.14b | E03 异常噪声样本污染鲁棒性曲线 | `results/figures/fig_e03_outlier_frac.png` | `scripts/analyze.py` ← `e03_sensitivity.csv` | 附录 A4（第 28 页） |
| Fig.15 | E04 编码器 × 聚类器 5×3 解耦热力图 | `results/figures/fig_e04_paired.png` | `scripts/analyze.py` ← `e04_paired.csv` | 第 19 页 |
| Fig.16 | E05 性能—计算成本帕累托前沿 | `results/figures/fig_e05_pareto.png` | `scripts/analyze.py` ← `e05_cost_frontiers.csv` | 第 20 页 |
| Fig.16b | E05 无标签训练稳定性 vs. 回顾性 ARI | `results/figures/fig_e05_stability.png` | `scripts/analyze.py` ← `selection.json` + `e01_summary.csv` | 附录 A5（第 29 页） |
| Fig.17 | 实证驱动的方法选型决策树 | `presentation/assets/diag_decision_tree.png` | `scripts/render_diagrams.py` ← `results/tables/` | 第 21 页 |
