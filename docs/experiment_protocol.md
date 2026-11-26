# 实验协议

数值以 `configs/protocol.yaml` 为准。协议在读取测试集指标前固定。当前结果由 `scripts/analyze.py` 从 `results/raw` 生成，空白表示没有可比测量，不能用预期数字填补。

## 数据与任务

| 数据集 | 训练子集 | 测试子集 | 输入 |
| --- | ---: | ---: | --- |
| MNIST | 20,000 | 10,000 | 1×28×28，像素除以 255 |
| Fashion-MNIST | 20,000 | 10,000 | 1×28×28，像素除以 255 |
| CIFAR-10 | 10,000 | 2,000 | 3×32×32 |

缓存的子集成员由 seed 决定。训练时不把标签传给 `fit`。默认 K=10 是已知类数假设，E03 会显式测试 K 取错的情况。

## 损失与训练规则

聚类项是样本平均的 KL(p‖q)。重构项是每张图的平方误差和，再对 batch 取平均：

`L_IDEC = L_cluster + λ L_recon`

λ 取 `0, 0.01, 0.1, 1, 10`。λ=0 表示去掉重构约束。Guo 等人把系数放在聚类项上，所以这里的 λ 不能直接换成论文中的 γ。

AE 预训练 50 epoch，学习率 0.001。DEC/IDEC 微调 50 epoch，学习率 0.0001。如果最后 5 个 epoch 的训练损失相对下降超过 5%，最多追加 10 个 epoch。DEC 系方法从第 10 个 epoch 后开始检查训练集簇分配，变化比例低于 0.001 时早停。两项规则都只读训练侧信号，不能看测试结果后修改。

梯度裁剪阈值为 5.0，只用于数值稳定。AE + K-means、DEC、IDEC 在同一子集和隐维度上共享一次 AE 预训练。VaDE 单独预训练，因为编码器还要输出均值和方差。

## 方法实现边界

- K-means 使用原始像素和 `n_init=20`。PCA 先中心化、不按方差缩放，取 10 维后做 K-means。
- GMM 使用对角协方差、`n_init=1`，并在聚类前对特征做标准化。高维满协方差不可识别，这是计算约束。
- SimCLR 和 Contrastive Clustering 使用 128 维小型 CNN，和 10 维自编码器不是同一个网络。它们是课程实现，不是原论文的 ResNet 版本。课程版 CC 省略原论文的熵项。
- VaDE 的重构是每图平方误差，不是 Bernoulli 交叉熵。
- E04 的随机 CNN、AE、SimCLR、DINOv2 ViT-S/14、CLIP ViT-B/32 使用同一 CIFAR-10 测试切分。DINOv2 和 CLIP 冻结，输入缩放到 224 并使用各自归一化。特征随后 L2 归一化。DEC head 只训练 10 维投影和聚类中心 20 epoch，不更新主干。

## 运行覆盖

- E01 基准：81 条 JSON，MNIST、Fashion-MNIST、CIFAR-10 各 9 个方法配置，使用 seed 0/1/2。
- E02 消融：Fashion-MNIST seed 0，改变 λ 和隐维度，另外记录初始化对照。
- E03 敏感性：Fashion-MNIST seed 0，测不平衡、样本比例、K 和异常点。E01 中已经测过的全量、平衡、无异常、K=10 行直接复用。
- E03 噪声：624 条 JSON，噪声标准差为 0、0.1、0.2、0.3。噪声评估不重训。
- E04 表征与聚类器：CIFAR-10，同一特征空间比较 K-means、GMM、DEC head，外部 DINOv2/CLIP 的 K-means 另有三 seed 稳定性汇总。
- E05 训练侧稳定性：使用 `results/raw/E05_train/*.npz`，计算同一训练样本在干净和扰动视图下的簇分配一致性，不加载测试标签。它是在基准完成后增加的诊断，不能与 E01 的带标签回顾性比较混写。

## 标签、评价与噪声

ACC 使用匈牙利匹配，NMI 使用算术平均，ARI 使用成对计数。轮廓系数只在各自嵌入空间内解释，不跨 10、128、784 维直接排名。

测试集标签只在 `src/evaluation/metrics.py`，以及 `scripts/run_one.py` 和 `scripts/eval_noise.py` 的 `fit` 返回后评价步骤出现。不平衡比例使用训练集标签构造子集，拟合函数仍看不到标签。噪声只加在测试图上，标准差相对 `[0,1]` 像素并截断回合法范围。

异常点实验把一部分训练图换成均匀噪声像素。样本量实验是无标签随机子集，epoch 预算不随样本量改变，所以小样本会看到更多遍数据。

## 选型与证据范围

`src/evaluation/selection.py` 的训练侧规则只使用拟合训练子集上的预测、簇稳定性、非空簇和预测耗时。缺失测量值时规则不启用。`results/tables/selection.json` 是无标签训练稳定性选择结果。

`results/tables/benchmark_interpretation.json` 是带测试标签的回顾性解读，只能说明历史基准中各方法的表现，不能当作新无标签数据上的自动选型器。E05 诊断与该回顾性表分开保存。

## 协议修复与可追溯性

本轮修复前的受影响文件保留在 `results/history/pre-repair-8eb43fc/`。`results/provenance/protocol-repair-20261126/manifest.json` 记录前一 commit、协议哈希、代码哈希、修复原因和替换文件。一次基础设施失败保留在 `results/attempts/`，只按原配置和 seed 重试一次。不能因为准确率低而换 seed。
