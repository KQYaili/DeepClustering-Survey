# 数据集介绍与实验协议

本文完整说明本项目使用的**三个基准图像数据集**、**数据切分与扰动构造规则**、**七组对照实验矩阵（E01–E05）**、**11 种模型的统一超参数与实现边界**，以及**端到端成本核算与防标签泄漏红线**。所有超参数与运行覆盖均以 `configs/protocol.yaml` 为准，在读取测试集指标前预先固定；所有结果表与图均由 `scripts/analyze.py` 从 `results/raw/` 自动汇总生成。

---

## 1. 三个基准数据集详细介绍

为系统检验不同深度聚类范式在**“简单结构 → 细粒度纹理 → 复杂自然语义”**难度梯度上的表现，本实验选取三个官方标准图像数据集（通过 `torchvision.datasets` 获取，见 `src/datasets/data.py`）：

### 1.1 基本规格与切分对照

| 数据集 | 官方全量（训练 / 测试） | 本项目单种子切分（训练 / 测试） | 图像尺寸与通道 | 拉平维度 $D$ | 像素预处理 | 真实类数 $K$ | 核心聚类难点 |
| --- | ---: | ---: | --- | ---: | --- | ---: | --- |
| **MNIST** | 60,000 / 10,000 | **20,000 / 10,000** | `1 × 28 × 28` 灰度 | 784 | 除以 255 归一至 `[0, 1]` | 10 | 居中单色手写笔画、纯黑背景；类内方差低，非线性流形较清晰 |
| **Fashion-MNIST** | 60,000 / 10,000 | **20,000 / 10,000** | `1 × 28 × 28` 灰度 | 784 | 除以 255 归一至 `[0, 1]` | 10 | 服饰轮廓高度重叠（如 T 恤 / 套头衫 / 衬衫 / 外套）、对像素平移极敏感 |
| **CIFAR-10** | 50,000 / 10,000 | **10,000 / 2,000** | `3 × 32 × 32` RGB 彩色 | 3,072 | 除以 255 归一至 `[0, 1]` | 10 | 背景颜色与光照主导像素距离，同物体视角/姿态/颜色变化极大，需要高层物体语义 |

> **为什么从全量做确定性抽样？**
> 在课程单卡算力约束下，为确保 11 种方法在 3 个随机种子及消融/敏感性/交叉实验中保持完全一致的样本切分，同时避免过长单次运行阻塞全矩阵覆盖，协议预先固定 MNIST / Fashion-MNIST 为 `20,000 / 10,000`、CIFAR-10 为 `10,000 / 2,000`。每个种子对应的索引由 `src/datasets/data.py::base_indices()` 用 `np.random.default_rng(seed)` 无放回抽样并持久化为 `.npz` 缓存，同一种子下所有方法看到的图像完全逐张一致。

### 1.2 各数据集的 10 个真实类别与视觉统计特征

1. **MNIST（手写数字，LeCun et al.）**
   - **10 个类别**：`0, 1, 2, 3, 4, 5, 6, 7, 8, 9`。
   - **统计特征**：字符统一居中归一化、背景几乎全黑（边缘像素方差近 0），低维非线性编码器（如 10 维 MLP AE）即可有效拉开各数字簇；但正因与自然彩图域差异巨大，在自然图像上预训练的 DINOv2 / CLIP 反而表现逊于本域自编码器（域不匹配现象）。

2. **Fashion-MNIST（电商服饰，Xiao et al., 2017）**
   - **10 个类别**：`0: T 恤 (T-shirt/top)`、`1: 裤子 (Trouser)`、`2: 套头衫 (Pullover)`、`3: 连衣裙 (Dress)`、`4: 外套 (Coat)`、`5: 凉鞋 (Sandal)`、`6: 衬衫 (Shirt)`、`7: 运动鞋 (Sneaker)`、`8: 包 (Bag)`、`9: 短靴 (Ankle boot)`。
   - **统计特征**：与 MNIST 尺寸完全相同（`28×28` 灰度），但属于“上衣 / 鞋靴 / 裤包”多层级细粒度语义，类间轮廓易混淆。实测统计（`presentation/assets/pixel_distance.yaml`，随机抽 1,000 张 T 恤）：将 T 恤右移仅 4 像素后，**99.9%** 的样本离原图的像素欧氏距离（中位 $11.3$）反而大于其离最近一件异类服饰（如衬衫，中位 $4.8$）的距离；即便仅右移 2 像素，该比例仍高达 **98.5%**。因此本项目选 Fashion-MNIST 作为 **E02 消融**与 **E03 鲁棒性敏感性**的核心测试床。

3. **CIFAR-10（自然彩色图像，Krizhevsky, 2009）**
   - **10 个类别**：`0: 飞机 (airplane)`、`1: 汽车 (automobile)`、`2: 鸟 (bird)`、`3: 猫 (cat)`、`4: 鹿 (deer)`、`5: 狗 (dog)`、`6: 青蛙 (frog)`、`7: 马 (horse)`、`8: 船 (ship)`、`9: 卡车 (truck)`。
   - **统计特征**：包含真实天空、海洋、草地、道路等复杂背景。在原始像素空间中，蓝天下的飞机与蓝海上的轮船距离极近，深色背景的多种动物混作一团，所有从零训练小网络（像素 K-means、ConvAE、DEC、IDEC、VaDE、小型 CNN 对比学习）的 ARI 全部停留在 `0.03–0.054`（接近随机分组）；必须借助大规模外部预训练语义表示（DINOv2 / CLIP）才能实现物体级聚拢（ARI 跃升至 `0.62`）。

---

## 2. 数据扰动构造与严格标签隔离规则

### 2.1 标签隔离红线（硬约束）

- **训练接口零标签**：所有算法的 `fit(x_train)` 接口仅接收图像数组，不接收也不读取任何标签；
- **测试标签仅用于最终评分**：测试集标签只允许出现在 `src/evaluation/metrics.py` 以及 `scripts/run_one.py`、`scripts/eval_noise.py` 在 `fit` 返回之后的评价步骤；
- **可视化零测试标签**：`embeddings_tsne.png` 按各方法自己的预测簇上色；封面与第 17 页的 CIFAR-10 图像马赛克仅按表示空间的 t-SNE 网格排列，不读取测试标签；
- **类别不平衡的实验设计说明**：E03 不平衡实验使用**训练集标签**按既定比例筛选训练子集，这是构造不平衡实验条件的数据采样步骤，拟合函数、损失函数、早停与选种过程仍完全不可见标签，且评价在均衡测试集上进行。

### 2.2 四类受控数据变换的精确实现（`src/datasets/data.py`）

1. **样本量缩放（`sample_frac`）**：从 20,000 张 Fashion-MNIST 训练图中无放回随机抽取 `{1%, 5%, 10%, 25%, 100%}`，对应精确样本数 **`200 / 1,000 / 2,000 / 5,000 / 20,000`**；训练轮数上限保持不变，测试集固定为 10,000 张。
2. **类别不平衡（`imbalance`）**：以训练子集中样本最多的单个类别为多数类（数量 $n_{\max}$），其余 9 个少数类各随机下采样至 $\mathrm{round}(n_{\max} / r)$，比例 $r \in \{1, 5, 10, 50\}$。因此总训练样本数随不平衡加剧而同步减少——3 个随机种子上的实测均值训练图数分别为 **`20,000 / 5,749.3 / 3,901.3 / 2,422.3`**（种子 0 为 `20,000 / 5,724 / 3,879 / 2,412`）。解读不平衡曲线时需注意样本量同步下降的伴随效应。
3. **训练集异常图（`outlier_frac`）**：按比例 `{0%, 5%, 10%, 20%}`（`rng = default_rng(seed + 17)`）随机选取训练图，将整张图替换为 $\mathrm{Uniform}(0, 1)$ 均匀随机噪声像素。
4. **测试集高斯加噪（`noise_std`）**：在模型**完全不重新训练**的条件下，向测试图加入像素级零均值高斯噪声 $\mathcal{N}(0, \sigma^2)$（`rng = default_rng(seed + 101)`），$\sigma \in \{0, 0.1, 0.2, 0.3\}$，并截断回合法区间 `[0, 1]`。

---

## 3. 七组对照实验矩阵总览（E01–E05）

全部计划实验共包含 **`1,068` 条成功运行的 JSON 记录**（`results/tables/coverage.csv` 状态 100% `ok`），外加 **`27` 组训练侧稳定性诊断 `.npz` 文件**（`results/raw/E05_train/`）：

| 实验编号 | 核心检验问题 | 数据集 | 覆盖方法与因子 | 随机种子 | 成功记录数 | 核心结果表与图表 |
| --- | --- | --- | --- | --- | ---: | --- |
| **E01** | 从零训练全谱系基准对比与端到端成本前沿 | MNIST、Fashion-MNIST、CIFAR-10 | 9 种从零训练方法（K-means / PCA+KM / GMM / AE+KM / DEC / IDEC / VaDE / SimCLR+KM / CC） | `0, 1, 2` | **81** | `e01_summary.csv`、`benchmark_cost_frontiers.csv`、`benchmark_ari.png`、`embeddings_tsne.png`、`cost_frontier.png` |
| **E02** | IDEC 组件消融（预训练 / 初始化 / 冻结编码器 / 隐维度 $d$ / $\lambda$） | Fashion-MNIST | 单因素变动：$\lambda \in \{0, 0.01, 0.1, 1, 10\}$、$d \in \{2, 10, 32, 64, 128\}$、关预训练、随机中心、冻结编码器 | `0, 1, 2` | **36** | `e02_summary.csv`、`components_lambda.png`、`latent_dim.png` |
| **E02_fixed** | 解耦重构权重 $\lambda$ 与早停训练轮数的混淆效应 | Fashion-MNIST | 同一预训练 AE 出发，关闭早停与追加轮次，每个 $\lambda \in \{0, 0.01, 0.1, 1, 10\}$ 严格训满 50 轮 | `0, 1, 2` | **15** | `e02_fixed_epochs.csv`、`components_lambda.png` |
| **E03** | 样本量、类别不平衡、异常图、预设簇数 $K$ 敏感性 | Fashion-MNIST | K-means / AE+KM / DEC / IDEC（3 种子）与 CC（单种子）× 4 因素（`sample_frac` / `imbalance` / `outlier_frac` / `k∈{5,8,10,12,15,20}`），全量基准行复用 E01 | `0, 1, 2`（CC 仅 `0`） | **195** | `e03_sensitivity_summary.csv`、`sample_size.png`、`sensitivity_k_outliers_imbalance.png` |
| **E03_noise** | 不重训条件下测试图加噪鲁棒性 | 3 数据集（E01 全量）+ Fashion（E03 种子 0） | $\sigma \in \{0, 0.1, 0.2, 0.3\}$ 测试图加噪评估 | `0, 1, 2`（E01） / `0`（E03） | **624** | `e03_noise.csv`、`noise_robustness.png` |
| **E04** | 表示（编码器）vs. 聚类器因子交叉实验 & 预训练优势与领域边界 | MNIST、Fashion-MNIST、CIFAR-10 | 5 种编码器（随机 CNN / AE / SimCLR / DINOv2 / CLIP）× 3 种聚类器（K-means / GMM / DEC 聚类头）；AE/SimCLR+KM 复用 E01 同源模型 | `0, 1, 2` | **117**（含复用共 135 格） | `e04_summary.csv`、`e04_pretrained.csv`、`representation_vs_clusterer.png` |
| **E05** | 无标签训练侧扰动稳定性诊断（验证“稳定是否等于正确”） | MNIST、Fashion-MNIST、CIFAR-10 | E01 种子 0 的 9 种从零训练方法，在 2,000 张训练子集上比较干净与 $\sigma=0.1$ 预测的 Macro Cluster Jaccard，**全程零测试标签** | `0` | **27 组 `.npz`** | `selection.json`、`stability_vs_ari.png` |

> **为什么需要补做 `E02_fixed`？**
> 按原协议早停规则（训练集簇分配变化 $<0.1\%$ 时停），$\lambda$ 越大，重构梯度越持续扰动隐坐标，簇分配越晚稳定：3 种子平均实际训练轮数随 $\lambda=0, 0.01, 0.1, 1, 10$ 依次为 **`26.0 / 29.3 / 45.0 / 50.0 / 50.0` 轮**（种子 0 为 `20 / 27 / 35 / 50 / 50`）。早停版曲线把“重构权重 $\lambda$”和“训练时长”混在了一起；`E02_fixed` 强制所有 $\lambda$ 从同一个预训练 AE 出发严格微调 50 轮，从而干净分离出 $\lambda$ 的净效应（$\lambda \le 0.1$ 与 DEC 持平，$\lambda=1, 10$ 在 3 个种子上全部低于 DEC）。

---

## 4. 11 种模型的网络配置、超参数与实现边界

### 4.1 统一训练控制规则（`configs/protocol.yaml`）

- **批大小与数值稳定**：`batch_size = 256`，梯度裁剪阈值 `grad_clip = 5.0`，微调阶段随机种子偏移 `finetune_seed_offset = 1009`。
- **自适应轮数追加**：计划轮数结束时，若最近 5 轮（`window=5`）训练损失相对下降超过 `5%`（`relative_drop=0.05`），自动追加一次 `10` 轮（`extra_epochs=10`），仅看训练损失。
- **早停规则**：DEC / IDEC / VaDE（ELBO 阶段）从第 10 轮起（`min_epoch=10`）检查训练集预测簇分配变化比例，若低于 `0.001`（0.1%）则早停，仅看训练侧信号。
- **预训练缓存共享**：同 `(dataset, seed, k, latent_dim)` 的 `ae_kmeans / dec / idec` 共享一次 AE 预训练权重；VaDE 因额外输出方差头，独立进行 50 轮重构预训练。

### 4.2 各模型具体架构、超参数与边界对照

| 家族 | 方法 | 核心网络结构（按数据集） | 隐维度 | 优化器与轮数 / 关键超参 | 实现边界与注意事项 |
| --- | --- | --- | ---: | --- | --- |
| 经典 | `kmeans` | 无神经网络，直接拉平像素 | 784 / 3072 | `n_init=20` | 像素欧氏距离对位置/亮度敏感，但对各向同性噪声保序 |
| 经典 | `pca_kmeans` | 中心化 PCA（不缩放方差）+ K-means | 10 | `n_components=10`, `n_init=20` | 线性降维基线，与 10 维 AE 直接对齐维度 |
| 经典 | `gmm` | `StandardScaler` + 对角协方差 GMM | 784 / 3072 | `covariance="diag"`, `n_init=1`, `max_iter=100`, `reg_covar=1e-6` | 轻噪坍缩仅限于高维像素标准化对角协方差实现，不泛化至全部概率模型 |
| 自编码器系 | `ae_kmeans` | 灰度：`MLP 784→500→500→2000→10`（对称解码 + Sigmoid）；CIFAR：`Conv2d(3→32→64→128)→FC(10)`（对称反卷积） | 10 | Adam, `lr=1e-3`, 50 轮预训练 + K-means (`n_init=20`) | 预训练采用每图像素平方和误差（SSE） |
| 自编码器系 | `dec` | 复用同配置预训练 AE 编码器，阶段②**丢弃解码器**，接 $K\times 10$ 簇中心 | 10 | Adam, `lr=1e-4`, 50 轮, $\alpha=1.0$, 早停 `<0.1%` | 等价于 $\lambda=0$ 的 IDEC 特例 |
| 自编码器系 | `idec` | 复用预训练 AE 完整编解码器 + $K\times 10$ 簇中心 | 10 | Adam, `lr=1e-4`, 50 轮, E01 默认 $\lambda=1.0$ | $L_{\mathrm{IDEC}} = L_{\mathrm{KL}} + \lambda L_{\mathrm{SSE}}$，与 Guo 等人 $\gamma$ 换算为 $\lambda = 1/(\gamma D)$ |
| 生成式 | `vade` | 同上编码器接 $(\mu, \log\sigma^2)$ 双头 + 对称解码器 + GMM 先验 $(\pi_c, \mu_c, \sigma_c^2)$ | 10 | Adam, `lr=1e-3`, 50 轮预训练 + 50 轮 ELBO, `logvar_bias=-2.0` | **重构项采用每图 SSE，非原论文 Bernoulli 交叉熵** |
| 对比学习 | `simclr_kmeans` | 共享 `SmallCNN`（`Conv 3×3: 32→64→128 → GAP → FC(128)`）+ 实例投影头 `128→128→128` + 离线 K-means | 128 | Adam, `lr=3e-4`, `wd=1e-4`, 100 轮, $\tau=0.5$ | **课程小型 CNN（非原论文 ResNet），仅训 100 轮**，路线性能被低估 |
| 对比学习 | `cc` | 同上共享 `SmallCNN` + 实例头 `128→128→128` + 簇头 `Linear(128→10)+Softmax` | 128 / 10 | Adam, `lr=3e-4`, `wd=1e-4`, 100 轮, $\tau_I=0.5, \tau_C=1.0$ | **小型 CNN + 省略原论文边际簇熵正则项**，存在明显簇偏斜（`max_fraction` 达 0.42–0.47） |
| 外部预训练 | `dinov2_*` | 灰度转 3 通道、双线性缩放 `224×224` + ImageNet 归一 → 冻结 `ViT-S/14` → L2 归一 | 384 | 冻结无训练；外接 K-means / Diag-GMM / `DecHead(D→64→10, 20轮, lr=1e-3)` | LVD-142M 外部数据自监督预训练，训练成本不可与从零训练直比，单独分组报告 |
| 外部预训练 | `clip_*` | OpenAI CLIP 预处理 `224×224` → 冻结 `ViT-B/32` 图像编码器 → L2 归一 | 512 | 同上 | OpenAI WIT 4 亿图文对预训练，单独分组报告 |

---

## 5. 训练成本核算口径、硬件分布与可追溯性

### 5.1 训练时间三套核算口径

因 DEC 与 IDEC 在运行中复用同配置 `(dataset, seed, k, latent_dim)` 的 AE 预训练缓存，`results/raw` 中的 `time_train_sec` 在缓存命中时只包含微调阶段耗时。为避免把预训练缓存当成“免费计算”，`scripts/analyze.py` 显式输出三列时间：

- **`time_train_e2e_mean_sec`（独立端到端训练时间）**：当 `ae_cache_hit=True` 时，自动补回同配置非缓存 `ae_kmeans` 的完整预训练耗时（无匹配记录时保留 `NaN`，绝不手填）。帕累托前沿表（`benchmark_cost_frontiers.csv`）与图（`cost_frontier.png`）**严格采用此端到端时间**；
- **`time_train_amortized_mean_sec`（共享预训练摊销时间）**：当 AE 预训练在 `ae_kmeans / dec / idec` 三个模型间共享时按 1/3 分摊前置耗时；
- **`time_train_mean_sec`（当次实测记录时间）**：原样保留当次脚本实测秒数。

### 5.2 运行硬件与可比性范围

- 原始运行（E01 全量 3 种子、E02/E03/E04 种子 0、E05）在同一台笔记本 GPU（RTX 5070 Ti）上完成；
- 补充的多种子运行（E02/E03/E04 种子 1–2）与 `E02_fixed` 在 Colab `Tesla T4`（213 条）与本机 `cpu`（46 条）上完成，每条原始记录的 `device` 字段均如实标注硬件；
- **时间对比严格限定于同一硬件完成的 E01 记录**，不跨硬件混排耗时。

### 5.3 选型与证据范围、协议修复追溯

- **两类证据严格分离**（详见 [方法选择](method_selection.md)）：`results/tables/selection.json` 是纯训练侧、零测试标签的坍缩与稳定性诊断（负面结论：稳定 $\ne$ 正确，`max_predict_sec=None` 未纳入预测耗时）；`results/tables/benchmark_interpretation.json` 与 `results/tables/benchmark_cost_frontiers.csv` 是使用测试标签的历史基准回顾性分析，不得当作新无标签数据的自动选型器。
- **审计与脱敏追溯**：协议修复前文件归档于 `results/history/before-protocol-repair/`，修复清单与日志记录在 `results/provenance/protocol-repair-2026-11-26/manifest.json`；一次基础设施错误记录保留在 `results/attempts/` 与 `results/tables/failures.csv`（个人路径已脱敏为 `<REPO>` / `<VENV>`），仅按原配置和原 seed 重试一次，绝不因准确率低更换种子。
