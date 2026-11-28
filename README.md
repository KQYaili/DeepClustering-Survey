# 深度聚类：范式演进、对照实验与方法选择

![同一批 CIFAR-10 测试图在像素空间（左）与 DINOv2 表示空间（右）中的排列](presentation/assets/readme_mosaics.png)

面对一批没有标签的图像，应该先改进“看图的方式（表示）”，还是先换更复杂的“分组规则（聚类器）”？在统一协议下系统比较了经典统计聚类、自编码器系深度聚类（AE + K-means、DEC、IDEC）、生成式（VaDE）、对比学习（SimCLR + K-means、CC）以及冻结外部预训练模型（DINOv2、CLIP）共 **5 大家族 11 种方法**。

## 方法谱系、模型架构与原始数学推导

- **五大家族与 11 种模型**：
  - **经典统计**：K-means、PCA + K-means、标准化对角协方差 GMM；
  - **自编码器系**：AE + K-means、DEC（自训练 KL 锐化）、IDEC（KL + 重构双目标）；
  - **生成式**：VaDE（变分自编码器 + 隐空间 GMM 先验，ELBO 闭式后验）；
  - **对比学习**：SimCLR + K-means、Contrastive Clustering（CC，实例行对比 + 簇列对比）；
  - **外部预训练（冻结）**：DINOv2（ViT-S/14，384 维）、CLIP（ViT-B/32，512 维）+ L2 归一化 + 下游聚类器。
- **核心理论与文档**：
  - [数学基础与原始推导](docs/mathematical_foundations.md)：K-means 各向同性噪声保序证明、PCA 最大方差/最小 L2 重构等价性、GMM EM 与像素对角协方差轻噪坍缩机理、DEC 原始梯度（Xie et al. Eq. 4–5）逐行求导、IDEC 权重换算证明（$\lambda = 1/(\gamma D)$）、VaDE 完整 ELBO 与最优簇后验 $q^*(c\mid x)=p(c\mid z)$ 推导、SimCLR InfoNCE/NT-Xent 与 CC 行列双对比及簇熵 $H(Y)$、L2 归一化余弦等价性，以及 ACC / NMI / ARI / Silhouette / 宏观 Jaccard 精确公式。
  - [文献综述](docs/literature_review.md)：DEC、IDEC、VaDE、SimCLR、CC、DINOv2、CLIP、TAC 与预印本 VLM4Cluster 的原始定义与脉络。
  - [分类框架](docs/taxonomy.md)：依据 Zhou et al. (2025) Figure 1 的四种交互范式（multi-stage、iterative、generative、simultaneously）与外部预训练分支。

## 数据集介绍与实验设计

### 三个基准数据集（均含 10 个均衡类别，固定 seed=42 分层划分）

| 数据集 | 规格与维度 | 训练 / 测试规模 | 难度特征与角色 |
| --- | --- | --- | --- |
| **MNIST** | 28×28×1 灰度（784 维） | 20,000 / 10,000 | 难度 1：前景/背景对比清晰，基础校验 |
| **Fashion-MNIST** | 28×28×1 灰度（784 维） | 20,000 / 10,000 | 难度 2：平移失配且上装四类高度重叠，承担主消融（E02）与敏感性（E03） |
| **CIFAR-10** | 32×32×3 彩色（3,072 维） | 10,000 / 2,000 | 难度 3：背景颜色与姿态变化主导像素距离，检验从零训练语义极限 |

### 七组对照实验（E01–E05）与防泄密

全套实验共完成 **1,068 条有效评估记录**（E01 主基准 81、E02 消融 36、E02_fixed 固定轮次对照 15、E03 敏感性 195、E03_noise 测试噪声 624、E04 编码器×聚类器 117）及 **27 份 E05 嵌入快照与无标签稳定性诊断**，详见 [数据集介绍与实验协议](docs/experiment_protocol.md)：

- **标签严格隔离**：`fit` 接口不接收标签；测试标签仅在最终评分步骤读取；类别不平衡子集仅用训练集标签构造采样比例。
- **规则事前锁定**：早停仅看训练集簇分配变化率（$\Delta < 0.1\%$ 连续 2 次），追加轮次只看训练损失是否仍在下降，全部参数写入 `configs/protocol.yaml`。
- **端到端耗时记账与自动出图**：DEC / IDEC 耗时包含 AE 预训练全链路；所有表和图由 `scripts/analyze.py` 直接从 `results/raw/` 生成，不手填结果，失败运行原样保留。
- **3 种子重复与预训练分表**：主要对比均跑 3 个随机种子报均值±标准差；DINOv2 与 CLIP 借助外部数据，与从零训练严格分表报告。

## 各实验结果对比与主要结论

### 1. 从零训练主基准（E01）

![从零训练基准 ARI 对比](results/figures/benchmark_ari.png)

> 灰度图（MNIST / Fashion-MNIST）上自编码器系（DEC、IDEC）与生成式（VaDE）明显领先；彩色自然图 CIFAR-10 上九种从零训练方法全部 ≤ 0.054，接近随机乱分。

### 2. 隐空间几何演化（E05）

![PCA、AE、DEC、IDEC 隐空间 t-SNE 演化](results/figures/embeddings_tsne.png)

> Fashion-MNIST 2,000 张测试图 t-SNE：非线性自编码器先把大类拉开，DEC 自训练进一步把每个簇压成紧凑孤岛，IDEC 因保留重构约束簇边界相对松散。

### 3. 改表示 vs 换聚类器（E04）

![5 种编码器 × 3 种聚类器 ARI 对比](results/figures/representation_vs_clusterer.png)

> 上下跨行（换表示）ARI 最多变化 0.60，同行横向（换聚类器）平均仅变化约 0.10；冻结 DINOv2 + K-means 在 CIFAR-10 上达 0.62，但在手写数字 MNIST 上仅 0.32，不及本地自编码器（0.59）。

### 4. 测试高斯噪声鲁棒性（E03_noise）

![测试高斯噪声鲁棒性对比](results/figures/noise_robustness.png)

> 测试集加噪后排名反转：DEC 仅保留 47% 的干净图 ARI，像素 K-means 保留 93%；标准化像素对角 GMM 在 $\sigma = 0.1$ 即塌缩。

### 5. 训练样本量敏感性（E03）

![训练样本量梯度对比](results/figures/sample_size.png)

> 仅用 1%（200 张）训练图时 K-means 超过 DEC；样本量增至千张以上，深度编码器预训练充分后联合优化才稳定反超。

### 6. IDEC 组件消融与重构权重 $\lambda$（E02 / E02_fixed）

![IDEC 组件消融与重构权重 λ 对比](results/figures/components_lambda.png)

> 去掉 AE 预训练或随机初始化中心导致最大跌幅；固定 50 轮排除早停干扰后，$\lambda \le 0.1$ 与 DEC 持平，$\lambda \ge 1$ 在三个种子上全部低于 DEC。

### 7. 端到端训练成本前沿

![端到端训练时间与 ARI 帕累托前沿](results/figures/cost_frontier.png)

> 横轴计入完整端到端耗时（DEC / IDEC 含 AE 预训练）：`PCA + K-means → K-means → AE + K-means → DEC` 构成性价比帕累托前沿。

### 8. 无标签稳定性诊断（E05）

![无标签稳定性与真实 ARI 对比](results/figures/stability_vs_ari.png)

> 加噪前后簇成员一致度（宏观 Jaccard）在三个数据集上选出的都是经典方法，而真实 ARI 最高的均为深度方法：无标签稳定性可用于排雷塌缩，但不能替代人工抽检。

<details>
<summary>展开更多消融与敏感性对比图（隐空间维度、簇数 K、训练离群点与长尾不平衡）</summary>

![隐空间维度消融](results/figures/latent_dim.png)

![簇数 K、训练离群点与类别不平衡敏感性](results/figures/sensitivity_k_outliers_imbalance.png)

</details>

### 核心实证发现

- **表示比聚类器更重要，但最好的表示取决于领域匹配。** 在三个数据集上，换表示让 ARI 最多变化 0.60，同一表示上换聚类器平均只变化约 0.10。冻结 DINOv2 + K-means 在 CIFAR-10 上达 0.62（九种从零训练方法均不超过 0.06），在 Fashion-MNIST 上达 0.53（DEC 为 0.44）；但在手写数字 MNIST 上只有 0.32，不如在 MNIST 本身训练的自编码器（0.59）。
- **联合优化有条件地有效。** 干净、样本充足时 DEC 领先（MNIST ARI 0.669）；但测试图加噪后 DEC 只保留 47% 的 ARI，像素 K-means 保留 93%；1% 极小样本时 K-means 也超过 DEC；固定 50 轮对照表明过大的重构权重（$\lambda \ge 1$）会系统压低聚类精度。
- **有了预训练表示，简单聚类器就够了。** 在冻结的 DINOv2 / CLIP 特征上，GMM 平均比 K-means 低 0.04，DEC 聚类头在 6 组中全部更低。
- **没有标签时，稳定不等于正确。** 按加噪前后簇成员一致度（宏观 Jaccard）挑模型，三个数据集上选中的都是经典方法，而 ARI 最高的都是深度方法；无标签稳定性适合排雷塌缩模型（如像素 GMM），但不能代替人工抽检（详见 [方法选择指南](docs/method_selection.md)）。

## 局限

实验固定簇数 $K = 10$。SimCLR 和 CC 使用课程规模 128 维小型 CNN、训练 100 轮，且 CC 省略了原论文的簇熵正则 $-H(Y)$；VaDE 的重构项为每图平方误差（SSE）而非 Bernoulli 交叉熵；GMM 使用标准化后的对角协方差。验证集仅覆盖 3 个小尺寸图像数据集；DINOv2 与 CLIP 依赖海量外部预训练数据，训练成本不可与从零训练直接对比。

本仓库使用 MIT License。
