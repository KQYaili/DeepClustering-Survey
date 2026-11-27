# Deep Clustering: A Survey of Paradigm Evolution, Empirical Benchmarks, and Method Selection

深度聚类：范式演进、实验基准与方法选型综述。

这是一份面向课程汇报的可复现实验仓库。它回答一个实际问题：当数据结构、计算预算和先验条件不同，应该优先尝试经典聚类、联合优化、生成式聚类、对比聚类，还是外部预训练特征。

## 当前状态

协议先写入 `configs/protocol.yaml`，再读取测试集指标。当前 `results/raw` 有 809 条成功 JSON 记录，包含 E01 基准、E02 消融、E03 敏感性、E03 噪声和 E04 表征/聚类器比较。E05 是训练侧无标签稳定性诊断，结果以 `results/raw/E05_train/*.npz` 保存，不把测试标签送入拟合或选型，且明确呈现为**“稳定 ≠ 正确”的负面诊断结果**（仅用于排雷坍缩风险，未纳入预测耗时）。

本轮协议修复以 `8eb43fc` 为前一状态。受影响的旧文件保存在 `results/history/pre-repair-8eb43fc/`，修复说明、来源哈希和运行日志保存在 `results/provenance/protocol-repair-20261126/`。其中修复了 SimCLR 初始化顺序和小样本 E03 的 epoch 追加逻辑，并按原 seed 重跑受影响记录。一次失败的 CIFAR-10 GMM 下载中断记录已脱敏本地路径后保留在 `results/attempts/` 与 `results/tables/failures.csv`，随后按同一 seed 只重试一次。

汇总表和图全部由 `scripts/analyze.py` 从 `results/raw` 生成。`results/tables/benchmark_cost_frontiers.csv` 与 `results/figures/fig_benchmark_pareto.png` 明确标注为**带测试标签的回顾性端到端成本基准分析**（已补回 DEC/IDEC 的前置 AE 预训练成本），与无标签 E05 `selection.json` 严格分离。模型权重、外部预训练缓存和完整运行归档不进入 Git，另存于 Google Drive 的 `DeepClustering-Survey` 文件夹。

## 怎么跑

在 WSL2 中：

```bash
bash scripts/setup_wsl.sh
bash scripts/launch_wsl.sh
```

`launch_wsl.sh` 会先跑核心测试，再按队列、噪声评估和分析脚本执行。运行完整回归测试与离线复现检查（无需 GPU）：

```bash
python -m unittest discover -s tests -v
```

只查看队列而不运行：

```bash
python scripts/run_queue.py --dry-run
```

当前实验使用本机 WSL2 的 RTX 5070 Ti Laptop GPU。数据缓存在 `~/.cache/dc-survey/data`，不放进 Git。

## 证据目录

| 路径 | 内容 |
| --- | --- |
| `results/raw/**/*.json` | 每次运行的指标、训练轨迹、耗时和状态（809 条 ok 记录） |
| `results/raw/embeddings/*.npz` | Fashion-MNIST seed 0 四个模型的测试嵌入（共 1.6 MB，保留入库以保证无权重离线运行 `analyze.py` 复现 `fig_e01_embeddings.png`） |
| `results/raw/E05_train/*.npz` | 不看测试标签的训练稳定性诊断（从零 9 候选） |
| `results/tables/` | 由原始 JSON 生成的 CSV/JSON 汇总（含 `benchmark_cost_frontiers.csv`、`selection.json`、`benchmark_interpretation.json`） |
| `results/figures/` | 由 `scripts/analyze.py` 自动生成的 13 张 300-DPI 科研数据图 |
| `results/checkpoints/` | 模型权重（217 份，约 1.4 GB），保存于 Google Drive 归档 |
| `results/history/` | 协议修复前的历史结果备份 |
| `results/provenance/` | 协议、代码哈希和修复运行日志 |
| `presentation/DeepClustering-Survey.pptx` | 蓝白学术信息图风 30 页汇报（24 页正文 + 6 页技术附录 A1–A6，全量动态读表 + Slide ID 严格绑定讲稿） |
| `presentation/DeepClustering-Survey.pdf` | 由 PowerPoint 导出的 30 页高清 PDF 交付版 |
| `presentation/speaker_notes.md` / `.docx` | 按稳定 Slide ID 与 30 页幻灯片严格一一绑定的逐页讲稿（含 Word 打印版） |
| `tests/` & `.github/workflows/ci.yml` | 覆盖核心指标、标签隔离、端到端成本、路径脱敏、PPT 动态绑定与讲稿 ID 校验的自动化 CI 套件 |

## 读结果前先知道

- `fit` 不接收标签。测试集标签只在评价步骤加载。类别不平衡子集使用训练集标签构造，这是实验设计，不是损失函数或选种子。
- 早停只观察训练集簇分配变化。追加 epoch 只在训练损失继续下降时触发，两个阈值固定在 `configs/protocol.yaml`。
- **E02 λ 消融早停共变边界**：在当前统一早停协议下，`λ=0..10` 的实际训练轮数为 `20→27→35→50→50 ep`，ARI 变化受 `λ` 与训练时长共同作用，不能单独归因于量纲；`λ=0` 时重构项未入损失，其 `recon_sse` 不可直接横向比较。
- 重构项是每张图的平方误差和再对 batch 取平均。IDEC 写成 `L_cluster + λ L_recon`（E01 主表配置 `λ=1.0`），这里的 λ 不能直接当作 Guo 等人论文中的 γ。
- **端到端成本口径**：DEC/IDEC 命中 AE 预训练缓存时，`time_train_sec` 仅计微调耗时；`e01_summary.csv` 与 `benchmark_cost_frontiers.csv` 额外给出补回同配置 AE 预训练后的 `time_train_e2e_mean_sec` 及共享摊销 `time_train_amortized_mean_sec`。
- SimCLR 和 Contrastive Clustering 使用课程版 128 维小型 CNN，不是原论文的 ResNet。课程版 CC 省略了原论文的边际簇熵项，在 MNIST 与 CIFAR-10 上出现明显簇规模偏斜（`max_fraction_mean` 达 0.42–0.47）。
- VaDE 的重构误差按图计算平方误差，不是 Bernoulli 交叉熵。GMM 使用 `StandardScaler` 标准化和对角协方差，其轻噪坍缩限定于当前实现，不泛化至整个概率模型家族（VaDE 保持约 53% ARI）。
- DINOv2 和 CLIP 是冻结的外部预训练模型，与从零训练模型分栏独立汇报；E04 5×3 交叉矩阵为 `seed 0` 单次配对（AE/SimCLR+K-means 来自同切分 E01），DINOv2/CLIP+K-means 另有 3 种子稳定性汇总。

## 汇报入口与许可

直接打开 `presentation/DeepClustering-Survey.pptx`。图表登记见 `presentation/figure_registry.md`，大纲见 `presentation/outline.md`，方法选型与 E05 诊断边界见 `docs/method_selection.md`，实验协议见 `docs/experiment_protocol.md`，分类框架见 `docs/taxonomy.md`。本仓库遵循 MIT 开源协议（详见 `LICENSE`）。
