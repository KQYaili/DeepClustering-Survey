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

### λ 与训练轮数

早停只看训练集簇分配是否稳定。λ 越大，重构梯度越持续地推动隐坐标，簇分配越晚稳定：按原协议，种子 0 上 λ = 0、0.01、0.1 分别在第 20、27、35 轮早停，λ = 1 和 10 跑满 50 轮。因此早停版的 λ 曲线同时改变了 λ 和训练轮数。`E02_fixed` 关闭早停和追加轮次，让每个 λ 都从同一个预训练自编码器出发训练 50 轮，结果在 `results/tables/e02_fixed_epochs.csv`。λ = 0 时重构项不进入损失，它的重构误差不能和 λ > 0 的点连成一条权衡曲线。

### 训练成本的端到端核算口径

由于 DEC 与 IDEC 在运行中复用同配置（相同 `dataset, seed, k, latent_dim`）的 AE 预训练缓存，`results/raw` 中的 `time_train_sec` 仅记录缓存命中后的微调阶段耗时。`scripts/analyze.py` 因此区分三套时间列：
- `time_train_e2e_mean_sec`：**独立端到端训练时间**（对 `ae_cache_hit=True` 的 DEC/IDEC 自动补回匹配的非缓存 `ae_kmeans` 训练时间；若无匹配记录则保留 `NaN`，绝不手填估算）；
- `time_train_amortized_mean_sec`：**共享预训练摊销时间**（当 AE 预训练被 `ae_kmeans / dec / idec` 共同复用时按 1/3 分摊前置 AE 耗时）；
- `time_train_mean_sec`：**当次实测记录时间**（缓存命中时仅代表增量微调耗时）。

帕累托前沿图（`results/figures/cost_frontier.png`）使用独立端到端训练时间，不把缓存命中当作免费计算。

## 方法实现边界

- K-means 使用原始像素和 `n_init=20`。PCA 先中心化、不按方差缩放，取 10 维后做 K-means。
- GMM 使用对角协方差、`n_init=1`，并在聚类前对特征做 `StandardScaler` 标准化。高维满协方差不可识别，这是计算约束；其轻噪坍缩现象严格限定于当前标准化对角协方差像素实现，不泛化至整个概率模型家族（同属概率生成的 VaDE 在 Fashion-MNIST `σ=0.3` 下仍保持约 53% ARI）。
- SimCLR 和 Contrastive Clustering 使用 128 维小型 CNN，和 10 维自编码器不是同一个网络。它们是课程实现，不是原论文的 ResNet 版本。课程版 CC 还省略了原论文的边际簇熵正则项，在 MNIST 与 CIFAR-10 上表现出明显的簇规模偏斜（`e01_summary.csv` 中 `max_fraction_mean` 分别达 0.47 与 0.42，`entropy_norm_mean` 降至 0.72–0.75），不能仅用网络规模解释其低分。
- IDEC 在 E01 主表中的默认配置为 `λ=1.0`。
- VaDE 的重构是每图平方误差，不是 Bernoulli 交叉熵。
- E04 在三个数据集上比较 5 种表示（随机 CNN、AE、SimCLR、DINOv2 ViT-S/14、CLIP ViT-B/32）与 3 种聚类器（K-means、GMM、DEC 聚类头），每个组合 3 个种子。DINOv2 和 CLIP 冻结，输入缩放到 224 并使用各自的归一化，特征随后 L2 归一化。DEC 聚类头只训练 10 维投影和簇中心 20 轮，不更新主干。AE / SimCLR + K-means 两格即 E01 中同一切分、同一种子的模型。

## 运行覆盖

- E01 基准：MNIST、Fashion-MNIST、CIFAR-10 各 9 个方法，种子 0、1、2。
- E02 消融（Fashion-MNIST，IDEC）：λ、隐空间维度、AE 预训练、簇中心初始化、冻结编码器，种子 0、1、2。
- `E02_fixed`：同一组 λ，关闭早停与追加轮次、每个 λ 训练 50 轮，种子 0、1、2。
- E03 敏感性（Fashion-MNIST）：类别不平衡（实际训练图数随比例由 20,000 降至 5,724 / 3,879 / 2,412）、样本比例、K 和异常点。K-means、AE + K-means、DEC、IDEC 跑种子 0、1、2，CC 只跑种子 0。E01 中已经测过的全量、平衡、无异常、K = 10 行直接复用。
- E03 测试噪声：噪声标准差 0、0.1、0.2、0.3，作用于 E01 的全部拟合（3 个种子）和 E03 种子 0 的变体，不重训。
- E04 表示与聚类器：三个数据集 × 5 种表示 × 3 种聚类器 × 3 个种子。
- E05 训练侧稳定性：`results/raw/E05_train/*.npz`，覆盖 E01 种子 0 的 9 个从零训练方法，计算同一批训练图在干净和加噪视图下的簇分配一致性，不加载测试标签，未使用预测耗时。

原始运行（E01、E02/E03/E04 的种子 0、E05）在笔记本 GPU（RTX 5070 Ti）上完成；补充的种子与 `E02_fixed` 在 Colab T4 和 CPU 上完成，每条记录的 `device` 字段注明硬件。时间比较只使用 E01 的记录。

## 标签、评价与噪声

ACC 使用匈牙利匹配，NMI 使用算术平均，ARI 使用成对计数。轮廓系数只在各自嵌入空间内解释，不跨 10、128、784 维直接排名。

测试集标签只在 `src/evaluation/metrics.py`，以及 `scripts/run_one.py` 和 `scripts/eval_noise.py` 的 `fit` 返回后评价步骤出现。可视化代码（`src/visualization/figures.py`、`scripts/render_illustrations.py`）不读取测试标签：`embeddings_tsne.png` 按各方法自己的预测簇上色，像素与 DINOv2 的图像马赛克只按表示排列。不平衡比例使用训练集标签构造子集，拟合函数仍看不到标签。噪声只加在测试图上，标准差相对 `[0,1]` 像素并截断回合法范围。

异常点实验把一部分训练图换成均匀噪声像素。样本量实验是无标签随机子集，epoch 预算不随样本量改变，所以小样本会看到更多遍数据。

## 选型与证据范围

详见 `docs/method_selection.md`：`results/tables/selection.json` 是无标签训练稳定性与坍缩风险诊断（负面结果：高稳定性不等于语义正确，未纳入预测耗时）；`results/tables/benchmark_interpretation.json` 与 `results/tables/benchmark_cost_frontiers.csv` 是带测试标签的历史基准回顾性分析，不得当作新无标签数据上的自动选型器。

## 协议修复、路径脱敏与可追溯性

本轮修复前的受影响文件保留在 `results/history/before-protocol-repair/`。`results/provenance/protocol-repair-2026-11-26/manifest.json` 记录修复原因、环境、替换文件与运行日志。一次基础设施失败保留在 `results/attempts/` 与 `results/tables/failures.csv`（已清除个人目录与机器绝对路径，替换为 `<REPO>` / `<VENV>`），只按原配置和 seed 重试一次，绝不因准确率低换 seed。
