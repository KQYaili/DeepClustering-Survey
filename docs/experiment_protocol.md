# 实验协议

数值以 `configs/protocol.yaml` 为准。协议在读取测试集指标前固定。当前结果由 `scripts/analyze.py` 从 `results/raw` 生成，空白表示没有可比测量，不能用预期数字填补。

## 数据与任务

| 数据集 | 训练子集 | 测试子集 | 输入 |
| --- | ---: | ---: | --- |
| MNIST | 20,000 | 10,000 | 1×28×28，像素除以 255 |
| Fashion-MNIST | 20,000 | 10,000 | 1×28×28，像素除以 255 |
| CIFAR-10 | 10,000 | 2,000 | 3×32×32 |

缓存的子集成员由 seed 决定。训练时不把标签传给 `fit`。默认 K=10 是已知类数假设，E03 会显式测试 K 取错的情况。

## 损失、早停与训练规则

聚类项是样本平均的 KL(p‖q)。重构项是每张图的平方误差和，再对 batch 取平均：

`L_IDEC = L_cluster + λ L_recon`

λ 取 `0, 0.01, 0.1, 1, 10`。λ=0 表示去掉重构约束。Guo 等人把系数放在聚类项上，所以这里的 λ 不能直接换成论文中的 γ。

AE 预训练 50 epoch，学习率 0.001。DEC/IDEC 微调 50 epoch，学习率 0.0001。如果最后 5 个 epoch 的训练损失相对下降超过 5%，最多追加 10 个 epoch。DEC 系方法从第 10 个 epoch 后开始检查训练集簇分配，变化比例低于 0.001 时早停。两项规则都只读训练侧信号，不能看测试结果后修改。

梯度裁剪阈值为 5.0，只用于数值稳定。AE + K-means、DEC、IDEC 在同一子集和隐维度上共享一次 AE 预训练。VaDE 单独预训练，因为编码器还要输出均值和方差。

### E02 重构权重 λ 与早停轮数的相互作用（重要解读边界）

在当前统一早停协议下，E02 的 `lambda_recon` 网格中实际训练轮数 `epochs_ran` 与 `λ` 单调共变：`λ=0 (20ep 早停) → 0.01 (27ep 早停) → 0.1 (35ep 早停) → 1 (50ep 跑满) → 10 (50ep 跑满)`。原因在于重构权重越大，解码梯度越持续推动隐坐标微调，从而推迟训练集簇分配变化率低于 `0.001` 的早停触发。相比之下，E02 的 `latent_dim` 五档、`pretrain=False`、`center_init=random` 均完整运行 50 epoch（`freeze_encoder=True` 在第 10 epoch 结构性早停）。

因此：
- 在当前未重跑固定 50 epoch 的协议下，**不得把 `λ≥1` 时 ARI 的下降单独归因于权重或损失量纲**，必须同时披露 `λ` 与实际训练轮数的共变效应。
- `λ=0` 处的 `recon_sse = 46.21` 对应“重构项未进入总损失且训练轮数最短”的条件，属于定义不可比点，不能与 `λ>0` 的重构误差直接作为连续权衡曲线读取。

### 训练成本的端到端核算口径

由于 DEC 与 IDEC 在运行中复用同配置（相同 `dataset, seed, k, latent_dim`）的 AE 预训练缓存，`results/raw` 中的 `time_train_sec` 仅记录缓存命中后的微调阶段耗时。`scripts/analyze.py` 因此区分三套时间列：
- `time_train_e2e_mean_sec`：**独立端到端训练时间**（对 `ae_cache_hit=True` 的 DEC/IDEC 自动补回匹配的非缓存 `ae_kmeans` 训练时间；若无匹配记录则保留 `NaN`，绝不手填估算）；
- `time_train_amortized_mean_sec`：**共享预训练摊销时间**（当 AE 预训练被 `ae_kmeans / dec / idec` 共同复用时按 1/3 分摊前置 AE 耗时）；
- `time_train_mean_sec`：**当次实测记录时间**（缓存命中时仅代表增量微调耗时）。

回顾性基准帕累托图（`fig_benchmark_pareto.png`）与主比较采用**独立端到端训练时间**，并用空心对照点提示缓存增量耗时，不把缓存命中当作免费计算。

## 方法实现边界

- K-means 使用原始像素和 `n_init=20`。PCA 先中心化、不按方差缩放，取 10 维后做 K-means。
- GMM 使用对角协方差、`n_init=1`，并在聚类前对特征做 `StandardScaler` 标准化。高维满协方差不可识别，这是计算约束；其轻噪坍缩现象严格限定于当前标准化对角协方差像素实现，不泛化至整个概率模型家族（同属概率生成的 VaDE 在 Fashion-MNIST `σ=0.3` 下仍保持约 53% ARI）。
- SimCLR 和 Contrastive Clustering 使用 128 维小型 CNN，和 10 维自编码器不是同一个网络。它们是课程实现，不是原论文的 ResNet 版本。课程版 CC 还省略了原论文的边际簇熵正则项，在 MNIST 与 CIFAR-10 上表现出明显的簇规模偏斜（`e01_summary.csv` 中 `max_fraction_mean` 分别达 0.47 与 0.42，`entropy_norm_mean` 降至 0.72–0.75），不能仅用网络规模解释其低分。
- IDEC 在 E01 主表中的默认配置为 `λ=1.0`。
- VaDE 的重构是每图平方误差，不是 Bernoulli 交叉熵。
- E04 的随机 CNN、AE、SimCLR、DINOv2 ViT-S/14、CLIP ViT-B/32 使用同一 CIFAR-10 测试切分。DINOv2 和 CLIP 冻结，输入缩放到 224 并使用各自归一化。特征随后 L2 归一化。DEC head 只训练 10 维投影和聚类中心 20 epoch，不更新主干。E04 5×3 交叉矩阵为 `seed 0` 单次配对（其中 AE/SimCLR + K-means 取自同切分 E01 `seed 0`，其余 13 格来自 E04 `seed 0`），外部 DINOv2/CLIP 的 K-means 另有 3 种子稳定性表；“复杂聚类器无正收益”仅限定于本次冻结特征、10 维投影头与单 seed 设置。

## 运行覆盖

- E01 基准：81 条 JSON，MNIST、Fashion-MNIST、CIFAR-10 各 9 个方法配置，使用 seed 0/1/2。
- E02 消融：Fashion-MNIST seed 0（`n=1`），改变 λ 和隐维度，另外记录初始化对照。
- E03 敏感性：Fashion-MNIST seed 0（`n=1`），测不平衡（实际 `n_train` 随比例由 20,000 降至 5,724 / 3,879 / 2,412）、样本比例、K 和异常点。E01 中已经测过的全量、平衡、无异常、K=10 行直接复用。
- E03 噪声：624 条 JSON，噪声标准差为 0、0.1、0.2、0.3（3 种子）。噪声评估不重训。
- E04 表征与聚类器：CIFAR-10，同一特征空间比较 K-means、GMM、DEC head（`n=1, seed 0`），外部 DINOv2/CLIP 的 K-means 另有 3 种子稳定性汇总（`n=3`）。
- E05 训练侧稳定性：使用 `results/raw/E05_train/*.npz`，仅覆盖 E01 从零 9 候选，计算同一训练样本在干净和扰动视图下的簇分配一致性，不加载测试标签，且当前未启用预测耗时阈值（`max_predict_sec=None`）。

## 标签、评价与噪声

ACC 使用匈牙利匹配，NMI 使用算术平均，ARI 使用成对计数。轮廓系数只在各自嵌入空间内解释，不跨 10、128、784 维直接排名。

测试集标签只在 `src/evaluation/metrics.py`，以及 `scripts/run_one.py` 和 `scripts/eval_noise.py` 的 `fit` 返回后评价步骤出现。可视化脚本 `src/visualization/benchmark.py` 不读取 `y_test`，因此 `fig_e01_embeddings.png` 按预测簇上色。不平衡比例使用训练集标签构造子集，拟合函数仍看不到标签。噪声只加在测试图上，标准差相对 `[0,1]` 像素并截断回合法范围。

异常点实验把一部分训练图换成均匀噪声像素。样本量实验是无标签随机子集，epoch 预算不随样本量改变，所以小样本会看到更多遍数据。

## 选型与证据范围

详见 `docs/method_selection.md`：`results/tables/selection.json` 是无标签训练稳定性与坍缩风险诊断（负面结果：高稳定性不等于语义正确，未纳入预测耗时）；`results/tables/benchmark_interpretation.json` 与 `results/tables/benchmark_cost_frontiers.csv` 是带测试标签的历史基准回顾性分析，不得当作新无标签数据上的自动选型器。

## 协议修复、路径脱敏与可追溯性

本轮修复前的受影响文件保留在 `results/history/before-protocol-repair/`。`results/provenance/protocol-repair-2026-11-26/manifest.json` 记录修复原因、环境、替换文件与运行日志。一次基础设施失败保留在 `results/attempts/` 与 `results/tables/failures.csv`（已清除个人目录与机器绝对路径，替换为 `<REPO>` / `<VENV>`），只按原配置和 seed 重试一次，绝不因准确率低换 seed。
