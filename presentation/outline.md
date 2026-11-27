# 深度聚类：范式演进与方法选型综述 — 汇报大纲（24 页正文 + 6 页技术附录）

正文严格控制在 **24 页、23–25 分钟**，预留 5 分钟答疑；全部额外实验图表、完整量化汇总与文献溯源置于 **6 页技术附录（A1–A6）**。每一页均分配唯一稳定 Slide ID，由 `scripts/build_pptx.py` 与 `presentation/speaker_notes.md` 构建时强制一一校验。

## 正文结构（第 1–24 页）

| 页码 | Slide ID | 模块 | 时长 | 页标题 | 核心图表 / 视觉元素 |
| --- | --- | --- | --- | --- | --- |
| 1 | `S01_cover` | 封面 | 0.5 min | 深度聚类：范式演进与方法选型综述 | 蓝白学术封面 + 四主线概览 |
| 2 | `S02_core_questions` | 研究背景 | 1.0 min | 研究核心：不做算法排行榜，聚焦三个可证伪命题 | 三命题卡片 + 25 分钟节奏条 |
| 3 | `S03_textbook_foundations` | 统计基础 | 1.0 min | 统计学起点：从多元统计教材到深度表征聚类 | 王学民教材第 2/3/6/7/8 章映射表 |
| 4 | `S04_kmeans_failure` | 统计基础 | 1.0 min | K-means 数学目标与原始像素欧氏距离失效机制 | 可编辑 Cambria Math 公式 + 高维语义鸿沟剖析 |
| 5 | `S05_paradigm_shift` | 统计基础 | 1.0 min | 深度聚类的核心跃迁：不仅学分组，更学表示 | 可编辑 Cambria Math 公式 + 四阶段信息流组件 |
| 6 | `S06_paradigm_evolution` | 范式演进 | 1.5 min | 范式演进全景：Zhou (2025) 四类并存交互与前沿 | **Fig.1** 原生可编辑形状（PNG 预览保留） |
| 7 | `S07_taxonomy_4d` | 范式演进 | 1.0 min | 四维方法分类框架：跨算法解构机制联系 | **Fig.2** 原生可编辑形状（PNG 预览保留） |
| 8 | `S08_dec_math` | 原理推导 | 1.5 min | 联合优化代表（一）：DEC Student-t 软分配与 KL 自训练 | 可编辑 Cambria Math 公式 + 推导要点 |
| 9 | `S09_idec_math` | 原理推导 | 1.5 min | 联合优化代表（二）：IDEC 重构约束与局部结构保护 | **Fig.3** 原生可编辑形状 + 可编辑公式 |
| 10 | `S10_vade_arch` | 原理推导 | 1.0 min | 生成式深度聚类：VaDE 变分自编码与高斯混合先验 | **Fig.4** 原生可编辑形状 + 可编辑公式 |
| 11 | `S11_contrastive_arch` | 原理推导 | 1.0 min | 自监督与对比聚类：SimCLR + K-means 与 CC 双视图机制 | **Fig.5** 原生可编辑形状 |
| 12 | `S12_pretrained_frontier` | 前沿演进 | 1.5 min | 2024–2026 前沿：预训练视觉模型与外部语义引导 | **Fig.6** 原生可编辑形状 |
| 13 | `S13_selection_hypotheses` | 选型假设 | 1.0 min | 实验前方法选型假设矩阵（待验证命题） | 7 场景方法选型先验对照表 |
| 14 | `S14_protocol_isolation` | 实验设计 | 1.0 min | 统一实验协议、五组实验矩阵与标签隔离红线 | E01–E05 五组矩阵 + 809 条记录纪律 |
| 15 | `S15_e01_benchmark` | 实验证据 | 1.5 min | E01 基准对比：9 种从零训练方法在三类图像上的 ARI | **Fig.7** `fig_e01_ari.png`（±SD、近似并列与 CC 偏斜） |
| 16 | `S16_e01_embeddings` | 实验证据 | 1.0 min | E01 潜在空间几何演化与共享初始化对照 | **Fig.8** `fig_e01_embeddings.png`（预测簇 t-SNE） |
| 17 | `S17_e02_lambda_ablation` | 实验证据 | 1.5 min | E02 重构权重 λ 消融：当前早停协议下 λ 与训练轮数共变 | **Fig.9** `fig_e02_lambda.png`（含 `epochs_ran` 与 `λ=0` 不可比注） |
| 18 | `S18_e03_noise_robustness` | 实验证据 | 1.5 min | E03 鲁棒性测试：测试噪声反转与 GMM / VaDE 分化 | **Fig.11** `fig_e03_noise.png`（含 VaDE 与当前 GMM 对比） |
| 19 | `S19_e04_pretrained_decouple` | 实验证据 | 1.5 min | E04 预训练解耦：表征跃升与冻结特征投影头对照 | **Fig.15** `fig_e04_paired.png`（n=1 配对）+ n=3 稳定性表 |
| 20 | `S20_benchmark_pareto_e2e` | 历史成本基准 | 1.0 min | 回顾性历史基准：性能—端到端训练成本帕累托前沿 | **Fig.16** `fig_benchmark_pareto.png` + 可编辑公式（非 E05，4 点端到端前沿） |
| 21 | `S21_selection_decision_tree` | 方法选型 | 1.5 min | 基于历史基准的分场景选型决策树与无标签警示 | **Fig.17** `diag_decision_tree.png` |
| 22 | `S22_verdict_e05_negative` | 方法选型 | 1.0 min | 回到开篇三命题：实证检验结论与 E05 负面诊断（稳定 ≠ 正确） | 三命题结论卡 + E05 三数据集反例卡片 |
| 23 | `S23_limitations_future` | 局限展望 | 1.0 min | 实验边界、实现局限与未来开放研究方向 | 四项硬边界 + 三大开放方向 |
| 24 | `S24_closing_repro` | 总结答疑 | 0.5 min | 总结：先问表示与预算，再区分历史基准与无标签排雷 | 三准则卡片 + GitHub/Drive 可复现声明 |

## 技术附录（第 25–30 页，答疑备查）

| 页码 | Slide ID | 附录编号 | 内容 | 核心图表 |
| --- | --- | --- | --- | --- |
| 25 | `A01_e01_full_table` | A1 | E01 全指标汇总表（含 ±SD、端到端耗时与簇偏斜） | 动态读取 `e01_summary.csv` |
| 26 | `A02_e02_latent_components` | A2 | E02 隐空间维度曲线与初始化组件消融（均 50 ep） | **Fig.10** `fig_e02_latent.png` + `fig_e02_components.png` |
| 27 | `A03_e03_imbalance_samples` | A3 | E03 类别长尾不平衡（含实际 `n_train`）与样本规模效率 | **Fig.12** `fig_e03_imbalance.png` + **Fig.13** `fig_e03_sample_frac.png` |
| 28 | `A04_e03_k_outliers` | A4 | E03 预设聚类数 K 错配与异常点污染敏感性 | **Fig.14** `fig_e03_k.png` + `fig_e03_outlier_frac.png` |
| 29 | `A05_e05_stability_negative` | A5 | E05 负面诊断三数据集对照（训练稳定性 ≠ 语义正确） | **Fig.16b** `fig_e05_stability.png` |
| 30 | `A06_refs_audit` | A6 | 核心文献核验索引与可复现审计清单 | `bibliography.bib` + `failures.csv` + CI |
