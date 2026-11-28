# 数学基础与原始推导

本文系统整理本综述与实验所涉及的全部**经典统计聚类方法、深度聚类代表算法、对比与预训练表示机制、评价指标与无标签稳定性诊断**的原始数学推导，并在每一节明确标注**“原论文设定”与“本项目课程实现”之间的对应与差异**，便于结合 [实验协议](experiment_protocol.md)、[代码实现](../src/) 与 [参考文献库](bibliography.bib) 对照阅读。

---

## 0. 统一符号约定与统计教材对应

| 符号 | 含义 | 对应多元统计教材（王学民第六版） |
| --- | --- | --- |
| $x_i \in [0,1]^D$ | 第 $i$ 张原始图像（拉平为 $D$ 维向量；MNIST/Fashion $D=784$，CIFAR-10 $D=3072$） | 第 2 章：多元观测样本向量 |
| $z_i = f_\theta(x_i) \in \mathbb{R}^d$ | 编码器 $f_\theta$ 输出的低维隐表示（AE/DEC/IDEC/VaDE 默认 $d=10$，SimCLR/CC 为 $128$，DINOv2/CLIP 为 $384/512$） | 第 7、8 章：主成分得分 / 潜因子向量 |
| $\hat{x}_i = g_\phi(z_i)$ | 解码器 $g_\phi$ 由隐表示 $z_i$ 重构出的图像 | 第 7、8 章：降维重构观测 |
| $K$ | 预设簇数（本实验基准默认 $K=10$） | 第 6 章：聚类的类数 $G$ 或 $K$ |
| $C_k,\;\mu_k,\;\Sigma_k,\;\pi_k$ | 第 $k$ 簇的样本集合、簇中心（均值）、协方差矩阵与混合先验比例（$\sum_{k=1}^K \pi_k = 1$） | 第 3、6 章：多元正态参数与组中心 |
| $q_{ij},\;p_{ij}$ | DEC / IDEC 中样本 $i$ 归属第 $j$ 簇的当前软分配与锐化目标分布 | 第 6 章：动态聚类与隶属度推广 |
| $\gamma_{ic} = q(c \mid x_i)$ | GMM / VaDE 中样本 $i$ 属于第 $c$ 簇的后验责任概率 | 第 3 章：多元混合分布后验 |
| $N,\;B$ | 训练集总样本数、随机梯度下降的批大小（本协议 $B=256$） | — |

---

## 1. 经典聚类基线与几何失效机理

### 1.1 K-means 交替极小化推导与噪声保序性

#### （1）目标函数与两步交替证明

给定特征坐标 $\{z_i\}_{i=1}^N \subset \mathbb{R}^d$ 与簇数 $K$，K-means 极小化所有样本到其所属簇中心的平方欧氏距离之和（即组内离差平方和，WCSS）：

$$J\big(\{C_k\}_{k=1}^K,\;\{\mu_k\}_{k=1}^K\big) = \sum_{k=1}^{K}\sum_{i \in C_k} \|z_i - \mu_k\|_2^2.$$

该问题关于离散划分 $\{C_k\}$ 与连续中心 $\{\mu_k\}$ 联合求解是 NP 难的，Lloyd 算法采用坐标下降交替优化：

1. **固定中心 $\{\mu_k\}$，更新划分 $\{C_k\}$（分配步）**：
   目标对每个样本 $i$ 完全可分，为使总和最小，必须把 $z_i$ 分配给欧氏距离最近的中心：
   $$\hat{c}_i = \arg\min_{k \in \{1,\dots,K\}} \|z_i - \mu_k\|_2^2.$$
   因此 $J(\{C_k^{(t+1)}\}, \{\mu_k^{(t)}\}) \le J(\{C_k^{(t)}\}, \{\mu_k^{(t)}\})$。

2. **固定划分 $\{C_k\}$，更新中心 $\{\mu_k\}$（更新步）**：
   对每个非空簇 $C_k$，将目标对 $\mu_k$ 求梯度并令其为零：
   $$\nabla_{\mu_k} J = \sum_{i \in C_k} 2(\mu_k - z_i) = 0 \implies \mu_k^* = \frac{1}{|C_k|}\sum_{i \in C_k} z_i.$$
   由于海森矩阵 $\nabla^2_{\mu_k} J = 2|C_k|I \succ 0$，样本均值是严格全局极小值，故 $J(\{C_k^{(t+1)}\}, \{\mu_k^{(t+1)}\}) \le J(\{C_k^{(t+1)}\}, \{\mu_k^{(t)}\})$。

因为 $J \ge 0$ 单调非增，且有限样本集的划分总数有限（不超过 $K^N$），算法必在有限步内收敛至局部极小值（本项目设 `n_init=20` 缓解局部最优敏感性）。

#### （2）为什么 K-means 在像素空间失效，却在各向同性噪声下极其抗噪？

- **平移敏感（像素距离量错语义）**：两张图像拉成向量后，$\|x_a - x_b\|_2^2 = \sum_{d=1}^D (x_{ad} - x_{bd})^2$ 逐像素做差。同一物体仅平移几个像素，前景边缘与黑色背景错位叠加，平方差急剧膨胀。在 Fashion-MNIST 训练集随机抽取 1,000 张 T 恤实测（见 `presentation/assets/pixel_distance.yaml`）：右移 4 像素后，**99.9%** 的 T 恤与原图的像素距离（中位 $11.3$）反而大于它与最近一件“非 T 恤”图像的距离（中位 $4.8$）。
- **测试加噪保序性推导（解释 E03 测试噪声排名反转）**：假设测试图加入零均值独立各向同性高斯噪声 $\varepsilon \sim \mathcal{N}(0, \sigma^2 I_D)$，噪声图像 $\tilde{x} = x + \varepsilon$ 到任意固定中心 $\mu_k$ 的期望平方距离为：
  $$\mathbb{E}_{\varepsilon}\!\left[\|\tilde{x} - \mu_k\|_2^2\right] = \|x - \mu_k\|_2^2 + 2(x - \mu_k)^\top \underbrace{\mathbb{E}[\varepsilon]}_{0} + \mathbb{E}\!\left[\|\varepsilon\|_2^2\right] = \|x - \mu_k\|_2^2 + D\sigma^2.$$
  噪声项 $D\sigma^2$ **与簇编号 $k$ 无关**；当 $D$ 较大时，由大数定律 $\|\varepsilon\|_2^2 \approx D\sigma^2$ 且交叉项 $2(x-\mu_k)^\top\varepsilon$ 方差远小于距离差，因此对所有簇中心近似加上同一常数，最近中心 $\arg\min_k \|\tilde{x}-\mu_k\|_2^2$ 基本不变。这在数学上解释了为何 Fashion-MNIST 加噪至 $\sigma=0.3$ 时，像素 K-means 仍保留 **93%** 的 ARI，而深层非线性网络特征因像素扰动逐层放大，DEC 仅剩 **47%**。

---

### 1.2 主成分分析（PCA）的最大方差与最小重构等价推导

设去中心化后的样本矩阵为 $\tilde{X} \in \mathbb{R}^{N \times D}$（$\frac{1}{N}\sum_i \tilde{x}_i = 0$），样本协方差矩阵 $S = \frac{1}{N}\tilde{X}^\top \tilde{X}$。寻求单位正交投影矩阵 $W = [w_1,\dots,w_d] \in \mathbb{R}^{D \times d}$（$W^\top W = I_d$）：

1. **最大投影方差视角**：对第一主成分 $w_1$，构造拉格朗日函数：
   $$\mathcal{L}(w_1, \lambda_1) = w_1^\top S w_1 - \lambda_1(w_1^\top w_1 - 1).$$
   令 $\nabla_{w_1}\mathcal{L} = 2Sw_1 - 2\lambda_1 w_1 = 0 \implies Sw_1 = \lambda_1 w_1$。代入目标得投影方差 $w_1^\top S w_1 = \lambda_1$，故最优 $w_1$ 为 $S$ 最大特征值对应的特征向量；依次施以正交约束，即得前 $d$ 个最大特征值对应的正交特征向量。

2. **最小线性重构误差视角**：将 $\tilde{x}_i$ 投影到 $W$ 张成的 $d$ 维子空间并线性重构为 $\hat{x}_i = W W^\top \tilde{x}_i$，由勾股定理（$W^\top W = I_d$）：
   $$\frac{1}{N}\sum_{i=1}^N \|\tilde{x}_i - W W^\top \tilde{x}_i\|_2^2 = \frac{1}{N}\sum_{i=1}^N \|\tilde{x}_i\|_2^2 - \frac{1}{N}\sum_{i=1}^N \|W^\top \tilde{x}_i\|_2^2 = \mathrm{tr}(S) - \mathrm{tr}(W^\top S W).$$
   因此，**极小化线性重构平方误差与极大化投影方差完全等价**。若把线性编码 $W^\top x$ 与线性解码 $W z$ 替换为多层非线性神经网络 $f_\theta(x)$ 与 $g_\phi(z)$，即得到深度聚类中的**自编码器（Autoencoder, AE）**。

> **实现说明**：本项目 `PCA + K-means` 仅按训练均值中心化、不做方差缩放，取 $d=10$ 维主成分后运行 `KMeans(n_init=20)`，与 10 维 AE 直接对齐维度。

---

### 1.3 高斯混合模型（GMM）的 EM 算法推导与轻噪坍缩机理

#### （1）生成模型与 EM 算法完整推导

GMM 假设观测（或特征）$z_i \in \mathbb{R}^d$ 由 $K$ 个多元正态成分混合生成，参数记为 $\Theta = \{\pi_k, \mu_k, \Sigma_k\}_{k=1}^K$：

$$p(z_i \mid \Theta) = \sum_{k=1}^{K} \pi_k\,\mathcal{N}(z_i \mid \mu_k, \Sigma_k), \qquad \sum_{k=1}^K \pi_k = 1,\;\pi_k \ge 0.$$

对数边缘似然 $\ell(\Theta) = \sum_{i=1}^N \log \sum_{k=1}^K \pi_k \mathcal{N}(z_i \mid \mu_k, \Sigma_k)$ 在对数内部含求和，无闭式解析解。对每个样本引入任意辅助概率分布 $q_i(k)$（$\sum_k q_i(k)=1$），由对数函数的凹性与 **Jensen 不等式**：

$$\ell(\Theta) = \sum_{i=1}^N \log \sum_{k=1}^K q_i(k) \frac{\pi_k \mathcal{N}(z_i \mid \mu_k, \Sigma_k)}{q_i(k)} \ge \sum_{i=1}^N \sum_{k=1}^K q_i(k) \log \frac{\pi_k \mathcal{N}(z_i \mid \mu_k, \Sigma_k)}{q_i(k)} \triangleq \mathcal{F}(q, \Theta).$$

- **E 步（紧化下界）**：固定 $\Theta^{(t)}$，将上式与原对数似然做差可得 $\ell(\Theta^{(t)}) - \mathcal{F}(q, \Theta^{(t)}) = \sum_{i=1}^N \mathrm{KL}\!\big(q_i(k)\,\|\,p(c_i=k \mid z_i, \Theta^{(t)})\big) \ge 0$。当且仅当 $q_i(k)$ 等于真实后验责任概率 $\gamma_{ik}$ 时等号成立：
  $$\gamma_{ik} = p(c_i=k \mid z_i, \Theta^{(t)}) = \frac{\pi_k^{(t)}\,\mathcal{N}(z_i \mid \mu_k^{(t)}, \Sigma_k^{(t)})}{\sum_{l=1}^K \pi_l^{(t)}\,\mathcal{N}(z_i \mid \mu_l^{(t)}, \Sigma_l^{(t)})}.$$

- **M 步（极大化完全数据期望对数似然 $\mathcal{Q}(\Theta \mid \Theta^{(t)})$）**：固定 $\gamma_{ik}$，分别对 $\pi_k$（加拉格朗日项 $\eta(\sum_k \pi_k - 1)$）、$\mu_k$ 与 $\Sigma_k^{-1}$ 求偏导并令其为零，记有效簇样本数 $N_k = \sum_{i=1}^N \gamma_{ik}$，得闭式更新：
  $$\pi_k^{(t+1)} = \frac{N_k}{N}, \qquad \mu_k^{(t+1)} = \frac{1}{N_k}\sum_{i=1}^N \gamma_{ik} z_i, \qquad \Sigma_k^{(t+1)} = \frac{1}{N_k}\sum_{i=1}^N \gamma_{ik}(z_i - \mu_k^{(t+1)})(z_i - \mu_k^{(t+1)})^\top.$$

#### （2）标准化对角协方差像素 GMM 为何一遇轻噪即塌缩为单簇？

本项目基线像素 GMM 受限于 784/3072 维满协方差不可逆，采用 `StandardScaler` 逐维标准化 + 对角协方差（`covariance_type="diag"`，`reg_covar=1e-6`）。其对数后验得分对第 $k$ 簇为：

$$\log \pi_k - \frac{1}{2}\sum_{d=1}^D \left[\log(2\pi \sigma_{kd}^2) + \frac{(\tilde{x}_{id} - \mu_{kd})^2}{\sigma_{kd}^2}\right], \qquad \tilde{x}_{id} = \frac{x_{id} - \bar{x}_d}{s_d}.$$

在 MNIST / Fashion-MNIST 上，大量边框背景像素在训练集中几乎恒为 $0$，导致训练标准差 $s_d \approx 0$ 且所有簇在该维的方差均退化至正则下界 $\sigma_{kd}^2 \approx 10^{-6}$。一旦测试图加入极小噪声 $\varepsilon_{id} \sim \mathcal{N}(0, 0.1^2)$，经标准化放大为 $\tilde{x}_{id} \approx \varepsilon_{id}/s_d \gg 1$，平方项 $(\tilde{x}_{id} - \mu_{kd})^2 / \sigma_{kd}^2$ 达到 $10^6$ 量级，彻底淹没中心前景像素的判别信号；哪一簇的 $\sigma_{kd}^2$ 哪怕略大一点点（微小数值差异），就以压倒性对数似然吸走全部测试图，导致在 $\sigma=0.1$ 时即塌缩为 **1 个非空簇**。

> **实现边界**：该塌缩严格限定于“高维原始像素 + 逐维标准化 + 对角协方差”这一经典实现，不代表所有概率混合模型都怕噪——同属高斯混合生成路线的 VaDE 先将图像压缩至 10 维平滑隐空间再建模，在 $\sigma=0.3$ 下仍保留约 **53%** ARI。

---

## 2. 联合优化深度聚类：DEC 与 IDEC 原始推导

### 2.1 DEC（Xie, Girshick & Farhadi, ICML 2016）

DEC 从预训练好的自编码器编码器 $f_\theta$ 出发，在隐空间 $z_i = f_\theta(x_i) \in \mathbb{R}^d$ 上用 K-means 初始化 $K$ 个可学习簇中心 $\{\mu_j\}_{j=1}^K$，之后**丢弃解码器**，仅用自训练散度联合微调 $\theta$ 与 $\{\mu_j\}$。

#### （1）Student-t 软分配与平方锐化目标分布

借用 t-SNE 的自由度为 $\alpha$（默认 $\alpha=1$）的 Student-t 核度量点 $z_i$ 与中心 $\mu_j$ 的相似度，定义软分配矩阵 $Q = [q_{ij}]$：

$$q_{ij} = \frac{\big(1 + \|z_i - \mu_j\|_2^2 / \alpha\big)^{-\frac{\alpha+1}{2}}}{\sum_{j'=1}^K \big(1 + \|z_i - \mu_{j'}\|_2^2 / \alpha\big)^{-\frac{\alpha+1}{2}}}.$$

接着构造更锐化的辅助目标分布 $P = [p_{ij}]$：

$$p_{ij} = \frac{q_{ij}^2 / f_j}{\sum_{j'=1}^K q_{ij'}^2 / f_{j'}}, \qquad f_j = \sum_{i=1}^N q_{ij}.$$

- **平方项 $q_{ij}^2$**：放大高置信预测（例如两簇等大时，$q=[0.70, 0.30] \implies p=[0.84, 0.16]$），形成“强者愈强”的向心拉力。
- **除以簇软频率 $f_j$**：对大簇降权、小簇加权，防止少数大簇吞并小簇导致隐空间塌缩。

#### （2）KL 散度目标与原始论文精确梯度推导（Xie et al. Eq. 4–5）

每轮固定 $P$，最小化 $P$ 与 $Q$ 之间的 KL 散度：

$$L_{\mathrm{DEC}} = \mathrm{KL}(P \,\|\, Q) = \sum_{i=1}^N \sum_{j=1}^K p_{ij} \log \frac{p_{ij}}{q_{ij}}.$$

下面给出原论文中 $L_{\mathrm{DEC}}$ 对隐特征 $z_i$ 与簇中心 $\mu_j$ 的逐步求导过程。令 $u_{ij} = \left(1 + \frac{\|z_i - \mu_j\|^2}{\alpha}\right)^{-\frac{\alpha+1}{2}}$，$Z_i = \sum_{j'} u_{ij'}$，则 $q_{ij} = u_{ij}/Z_i$，且：

$$\log q_{ij} = -\frac{\alpha+1}{2}\log\!\left(1 + \frac{\|z_i - \mu_j\|^2}{\alpha}\right) - \log Z_i.$$

注意到 $\frac{\partial \log u_{ij}}{\partial z_i} = -\frac{\alpha+1}{\alpha}\left(1 + \frac{\|z_i - \mu_j\|^2}{\alpha}\right)^{-1}(z_i - \mu_j)$，而对分母 $\log Z_i$ 求导得：

$$\frac{\partial \log Z_i}{\partial z_i} = \frac{1}{Z_i}\sum_{j'=1}^K u_{ij'} \frac{\partial \log u_{ij'}}{\partial z_i} = \sum_{j'=1}^K q_{ij'} \frac{\partial \log u_{ij'}}{\partial z_i}.$$

固定 $p_{ij}$（且 $\sum_{j=1}^K p_{ij} = 1$），对 $L_i = \sum_j p_{ij}(\log p_{ij} - \log q_{ij})$ 求梯度：

$$\begin{aligned}
\frac{\partial L_{\mathrm{DEC}}}{\partial z_i} &= -\sum_{j=1}^K p_{ij}\frac{\partial \log q_{ij}}{\partial z_i} = -\sum_{j=1}^K p_{ij}\frac{\partial \log u_{ij}}{\partial z_i} + \underbrace{\left(\sum_{j=1}^K p_{ij}\right)}_{1}\sum_{j'=1}^K q_{ij'}\frac{\partial \log u_{ij'}}{\partial z_i} \\
&= -\sum_{j=1}^K (p_{ij} - q_{ij})\frac{\partial \log u_{ij}}{\partial z_i} \\
&= \frac{\alpha+1}{\alpha}\sum_{j=1}^K \left(1 + \frac{\|z_i - \mu_j\|_2^2}{\alpha}\right)^{-1} (p_{ij} - q_{ij})(z_i - \mu_j).
\end{aligned}$$

由对称性（$\|z_i - \mu_j\|_2^2$ 对 $\mu_j$ 求导只多一负号），对簇中心 $\mu_j$ 的梯度为：

$$\frac{\partial L_{\mathrm{DEC}}}{\partial \mu_j} = -\frac{\alpha+1}{\alpha}\sum_{i=1}^N \left(1 + \frac{\|z_i - \mu_j\|_2^2}{\alpha}\right)^{-1} (p_{ij} - q_{ij})(z_i - \mu_j).$$

**几何含义与失效隐患**：
- 当 $p_{ij} > q_{ij}$ 时，负梯度 $-\frac{\partial L}{\partial z_i}$ 沿 $\mu_j - z_i$ 方向把高置信样本拉向第 $j$ 簇中心；
- 权重因子 $(1 + \|z_i - \mu_j\|^2/\alpha)^{-1}$ 在距离极大时趋近于 $0$，防止极端离群样本产生过大梯度；
- **风险**：如果预训练表示不够好（如 200 张小样本或 CIFAR-10），初始误分样本若碰巧获得较高 $q_{ij}$，会被 $p_{ij}$ 正反馈越拉越偏，永久破坏局部结构。

> **实现说明**：本项目 `src/deep/common.py` 中 `cluster_kl` 对 batch 取样本均值 $\frac{1}{B}\sum_{i=1}^B \sum_j p_{ij}\log(p_{ij}/q_{ij})$；每轮开始用全体训练样本重算一次目标 $P$，并使用 Adam 配合反向传播自动求导更新编码器与簇中心。

---

### 2.2 IDEC（Guo et al., IJCAI 2017）与权重系数换算证明

为避免 DEC 丢弃解码器后扭曲隐空间局部流形结构，IDEC 在微调阶段**保留解码器 $g_\phi$**，同时优化聚类 KL 散度与像素重构误差：

- **原论文 Guo et al. 写法**（重构误差按像素取平均 MSE，系数 $\gamma$ 乘在聚类项上）：
  $$L_{\mathrm{Guo}} = \frac{1}{B D}\sum_{i=1}^B \|x_i - g_\phi(f_\theta(x_i))\|_2^2 + \gamma\, L_{\mathrm{cluster}}.$$
- **本项目实现写法**（重构误差按每张图求像素平方和 SSE，系数 $\lambda$ 乘在重构项上）：
  $$L_{\mathrm{IDEC}} = L_{\mathrm{cluster}} + \lambda \cdot \frac{1}{B}\sum_{i=1}^B \|x_i - g_\phi(f_\theta(x_i))\|_2^2.$$

**严格等价换算证明**：将 $L_{\mathrm{Guo}}$ 整体除以正常数 $\gamma > 0$（不改变最优解及梯度方向）：

$$\frac{1}{\gamma} L_{\mathrm{Guo}} = L_{\mathrm{cluster}} + \frac{1}{\gamma D}\left(\frac{1}{B}\sum_{i=1}^B \|x_i - g_\phi(f_\theta(x_i))\|_2^2\right) \implies \lambda = \frac{1}{\gamma D}.$$

例如在 MNIST / Fashion-MNIST（$D=784$）上，原论文默认 $\gamma=0.1$ 对应 $\lambda = \frac{1}{0.1 \times 784} \approx 0.0128$。这也从量纲上解释了为什么在本项目 E02 与 `E02_fixed` 中：$\lambda \in \{0.01, 0.1\}$ 与 DEC ($\lambda=0$) 几乎持平，而当 $\lambda \in \{1, 10\}$ 时，每图平方和量级极大，重构梯度压倒聚类梯度，反而令 Fashion-MNIST 的 ARI 明显下降。

---

## 3. 生成式深度聚类：VaDE（Jiang et al., IJCAI 2017）完整 ELBO 推导

### 3.1 生成过程与联合概率分解

VaDE 将高斯混合模型搬入变分自编码器（VAE）的隐空间，假设单张图像 $x$ 按如下三步生成：

1. 按类别先验抽簇编号：$c \sim \mathrm{Cat}(\pi)$，$p(c) = \pi_c$；
2. 从第 $c$ 簇的对角协方差高斯分布抽隐变量：$z \mid c \sim \mathcal{N}(\mu_c, \mathrm{diag}(\sigma_c^2))$；
3. 经非线性解码网络 $g_\phi$ 生成观测图像：$x \mid z \sim p_\phi(x \mid z)$。

联合分布分解为 $p(x, z, c) = p_\phi(x \mid z)\,p(z \mid c)\,p(c)$。

### 3.2 证据下界（ELBO）推导

由对数边缘似然与 Jensen 不等式，引入变分近似后验 $q(z, c \mid x)$：

$$\begin{aligned}
\log p(x) &= \log \sum_{c=1}^{K} \int p(x, z, c)\,dz = \log \sum_{c=1}^{K} \int q(z, c \mid x)\frac{p(x, z, c)}{q(z, c \mid x)}\,dz \\
&\ge \mathbb{E}_{q(z, c \mid x)}\!\left[\log \frac{p(x, z, c)}{q(z, c \mid x)}\right] \triangleq \mathcal{L}_{\mathrm{ELBO}}(x).
\end{aligned}$$

VaDE 采用平均场（mean-field）分解 $q(z, c \mid x) = q(z \mid x)\,q(c \mid x)$，其中编码器输出对角高斯参数 $q(z \mid x) = \mathcal{N}\big(\tilde{\mu}(x), \mathrm{diag}(\tilde{\sigma}^2(x))\big)$，则 ELBO 展开为：

$$\begin{aligned}
\mathcal{L}_{\mathrm{ELBO}}(x) &= \mathbb{E}_{q(z, c \mid x)}\!\left[\log p_\phi(x \mid z) + \log p(z \mid c) + \log p(c) - \log q(z \mid x) - \log q(c \mid x)\right] \\
&= \mathbb{E}_{q(z \mid x)}[\log p_\phi(x \mid z)] - \mathrm{KL}\big(q(c \mid x)\,\|\,p(c)\big) - \sum_{c=1}^{K} q(c \mid x)\,\mathrm{KL}\big(q(z \mid x)\,\|\,p(z \mid c)\big).
\end{aligned}$$

### 3.3 最优簇后验 $q^*(c \mid x) = p(c \mid z)$ 的严格证明

将上述 $\mathcal{L}_{\mathrm{ELBO}}(x)$ 按 $p(z, c) = p(c \mid z)p(z)$ 重新配对整理：

$$\begin{aligned}
\mathcal{L}_{\mathrm{ELBO}}(x) &= \mathbb{E}_{q(z \mid x)}\!\left[\log \frac{p_\phi(x \mid z)\,p(z)}{q(z \mid x)} + \sum_{c=1}^K q(c \mid x)\log \frac{p(c \mid z)}{q(c \mid x)}\right] \\
&= \mathbb{E}_{q(z \mid x)}\!\left[\log \frac{p_\phi(x \mid z)\,p(z)}{q(z \mid x)}\right] - \mathbb{E}_{q(z \mid x)}\!\Big[\mathrm{KL}\big(q(c \mid x)\,\|\,p(c \mid z)\big)\Big].
\end{aligned}$$

第一项与 $q(c \mid x)$ 无关；第二项中 $\mathrm{KL}(\cdot\|\cdot) \ge 0$。当单样本蒙特卡洛采样隐变量 $z^{(l)}$ 后，为使下界取极大值，唯一最优闭式解即为使该 KL 散度为零的贝叶斯后验：

$$\gamma_c \triangleq q^*(c \mid x) = p(c \mid z) = \frac{p(c)\,p(z \mid c)}{\sum_{c'=1}^K p(c')\,p(z \mid c')} = \frac{\pi_c\,\mathcal{N}\big(z \mid \mu_c, \mathrm{diag}(\sigma_c^2)\big)}{\sum_{c'=1}^K \pi_{c'}\,\mathcal{N}\big(z \mid \mu_{c'}, \mathrm{diag}(\sigma_{c'}^2)\big)}.$$

这一优雅的解析推导使 VaDE **无需单独训练分类器网络**，仅凭编码器与 GMM 先验参数即可直接算出簇归属概率 $\gamma_c$。

### 3.4 重参数化与闭式解析 KL 展开（与 `src/deep/vade.py` 逐项对应）

使用 SGVB 重参数化技巧采样 $z = \tilde{\mu} + \tilde{\sigma} \odot \varepsilon$（$\varepsilon \sim \mathcal{N}(0, I_d)$），利用两个 $d$ 维对角高斯分布 KL 散度的闭式解：

$$\mathrm{KL}\!\big(\mathcal{N}(\tilde{\mu}, \mathrm{diag}(\tilde{\sigma}^2)) \,\|\, \mathcal{N}(\mu_c, \mathrm{diag}(\sigma_c^2))\big) = \frac{1}{2}\sum_{j=1}^{d}\left(\log \sigma_{cj}^2 + \frac{\tilde{\sigma}_j^2 + (\tilde{\mu}_j - \mu_{cj})^2}{\sigma_{cj}^2} - 1 - \log \tilde{\sigma}_j^2\right),$$

以及离散分类分布 KL 散度 $\mathrm{KL}(q(c \mid x)\,\|\,p(c)) = \sum_{c=1}^K \gamma_c \log \frac{\gamma_c}{\pi_c}$，极大化 ELBO 等价于极小化损失函数：

$$L_{\mathrm{VaDE}}(x) = L_{\mathrm{recon}}(x, \hat{x}) + \sum_{c=1}^{K} \gamma_c \log \frac{\gamma_c}{\pi_c} + \frac{1}{2}\sum_{c=1}^{K}\gamma_c \sum_{j=1}^{d}\left(\log \sigma_{cj}^2 + \frac{\tilde{\sigma}_j^2 + (\tilde{\mu}_j - \mu_{cj})^2}{\sigma_{cj}^2} - 1 - \log \tilde{\sigma}_j^2\right).$$

> **原论文与本项目实现差异**：Jiang 等人原文对二值化灰度图采用 Bernoulli 解码似然（即逐像素二元交叉熵 BCE）；本项目 `src/deep/vade.py` 为在 MNIST、Fashion-MNIST 与彩色自然图 CIFAR-10 上统一重构口径，采用**每图平方误差（SSE）** $L_{\mathrm{recon}}(x, \hat{x}) = \sum_{j=1}^D (x_j - \hat{x}_j)^2$（等价于固定方差 $\sigma_x^2 = \frac{1}{2}$ 的各向同性高斯观测似然，略去常数项）。预测阶段确定性地使用编码器均值 $\tilde{\mu}(x)$ 计算 $\gamma_c$ 并取 $\arg\max_c \gamma_c$。

---

## 4. 对比学习路线：SimCLR 与对比聚类（CC）数学推导

### 4.1 SimCLR（Chen et al., ICML 2020）NT-Xent 损失与互信息下界

对批内 $N$ 张图像，每张图独立施加两次随机数据增强（随机裁剪缩放、水平翻转、亮度扰动），得到 $2N$ 个增强视图 $\{x_1^a, \dots, x_N^a,\; x_1^b, \dots, x_N^b\}$。经共享骨干网络 $f_\theta$ 得到表征 $h_i$，再经两层非线性投影头 $g_\psi$ 得到投影向量 $u_i = g_\psi(h_i)$。

定义余弦相似度 $\mathrm{sim}(u, v) = \frac{u^\top v}{\|u\|_2 \|v\|_2}$。对互为正样本对的 $(i, i^+)$，归一化温度缩放交叉熵损失（NT-Xent / InfoNCE）为：

$$\ell_i = -\log \frac{\exp\!\big(\mathrm{sim}(u_i, u_i^+) / \tau\big)}{\sum_{k=1,\,k\neq i}^{2N} \exp\!\big(\mathrm{sim}(u_i, u_k) / \tau\big)}, \qquad \mathcal{L}_{\mathrm{inst}} = \frac{1}{2N}\sum_{i=1}^{2N} \ell_i.$$

- **互信息下界意义**：按 Oord 等人（CPC / InfoNCE）的经典证明，若令 $s(u_i, u_k) = \exp(\mathrm{sim}(u_i, u_k)/\tau)$ 逼近密度比 $\frac{p(x^a, x^b)}{p(x^a)p(x^b)}$，则最小化 $\mathcal{L}_{\mathrm{inst}}$ 等价于极大化两视图之间互信息 $I(x^a; x^b)$ 的变分下界：
  $$I(x^a; x^b) \ge \log(2N - 1) - \mathcal{L}_{\mathrm{inst}}.$$
- **温度系数 $\tau$ 的作用**：对 $\ell_i$ 关于负样本相似度 $s_{ik} = \mathrm{sim}(u_i, u_k)$ 求梯度得 $\frac{\partial \ell_i}{\partial s_{ik}} = \frac{1}{\tau}\frac{\exp(s_{ik}/\tau)}{\sum_{m \neq i}\exp(s_{im}/\tau)}$；$\tau$ 越小，梯度越集中在与 $u_i$ 最相似的“难负样本”上。

训练结束后，SimCLR 丢弃投影头 $g_\psi$，在骨干特征 $h_i = f_\theta(x_i) \in \mathbb{R}^{128}$ 上离线运行 K-means。

### 4.2 Contrastive Clustering（Li et al., AAAI 2021）行/列双对比与簇熵正则

CC 在共享骨干之后并行接出两个头：
1. **实例头（行对比）**：输出 $Z^a, Z^b \in \mathbb{R}^{N \times d_p}$，沿**行**（样本维度）计算实例级 NT-Xent 损失 $\mathcal{L}_{\mathrm{inst}}$（温度 $\tau_I = 0.5$）。
2. **簇头（列对比）**：输出经 Softmax 归一化的软簇分配矩阵 $Y^a, Y^b \in [0,1]^{N \times K}$。取矩阵的**第 $k$ 列** $y_k^a, y_k^b \in [0,1]^N$——它刻画了“批内 $N$ 张图分别有多大概率属于第 $k$ 簇”。因为两份增强视图应把同一批图分进同一个簇，所以 $(y_k^a, y_k^b)$ 构成正簇对，与其他 $2K-2$ 个列向量构成负簇对，簇级对比损失为：
   $$\ell_k^{\mathrm{clus}} = -\log \frac{\exp\!\big(\mathrm{sim}(y_k^a, y_k^b) / \tau_C\big)}{\sum_{j=1}^{K}\exp\!\big(\mathrm{sim}(y_k^a, y_j^b)/\tau_C\big) + \sum_{j \neq k}\exp\!\big(\mathrm{sim}(y_k^a, y_j^a)/\tau_C\big)}, \qquad \mathcal{L}_{\mathrm{clus}} = \frac{1}{2K}\sum_{k=1}^{K}\big(\ell_k^{a} + \ell_k^{b}\big).$$
3. **原论文边际簇熵正则项**：为避免所有样本落入同一簇的平凡解，Li 等人在原论文总损失中减去了簇分布香农熵（等价于鼓励边缘概率均匀分布）：
   $$H(Y) = -\sum_{k=1}^K \Big[P(y_k^a)\log P(y_k^a) + P(y_k^b)\log P(y_k^b)\Big], \qquad P(y_k^a) = \frac{1}{N}\sum_{i=1}^N Y_{ik}^a,$$
   原论文总目标为 $\mathcal{L}_{\mathrm{CC}}^{\mathrm{orig}} = \mathcal{L}_{\mathrm{inst}} + \mathcal{L}_{\mathrm{clus}} - H(Y)$。

> **课程实现边界（重要）**：受单卡/课程算力限制，本项目 `src/contrastive/train.py` 中的 SimCLR 与 CC 均使用 3 层 128 维 `SmallCNN`（而非原论文 ResNet-18/34/50），仅训练 100 轮；且课程版 CC 优化 $\mathcal{L}_{\mathrm{inst}} + \mathcal{L}_{\mathrm{clus}}$，**省略了原论文的边际簇熵项 $-H(Y)$**，因此实验中 CC 出现簇偏斜（MNIST 与 CIFAR-10 的 `max_fraction_mean` 分别达 0.47 与 0.42）。所有结论均严格限定在此课程配置下，不直接否定大网络完整版对比聚类路线。

---

## 5. 外部预训练表示与 L2 归一化的几何等价性

在 E04 中，我们对冻结的 DINOv2（ViT-S/14，384 维）与 CLIP（ViT-B/32，512 维）输出特征做逐样本 L2 归一化：

$$\hat{u}_i = \frac{u_i}{\|u_i\|_2} \in \mathbb{S}^{d-1}.$$

任意两个归一化向量之间的平方欧氏距离严格满足恒等式：

$$\|\hat{u}_i - \hat{u}_j\|_2^2 = \|\hat{u}_i\|_2^2 + \|\hat{u}_j\|_2^2 - 2\hat{u}_i^\top \hat{u}_j = 2\big(1 - \cos(u_i, u_j)\big).$$

这意味着：**在 L2 归一化后的预训练表示上运行最简单的欧氏 K-means，本质上就是直接在单位超球面上做余弦语义聚类**。当外部预训练已经把同语义物体的特征拉到相近方向上（如 CIFAR-10 上 DINOv2 特征），线性可分的球面划分已足够接近最优，再额外叠加未经预训练对齐的随机初始化两层 `DecHead`（`D→64→ReLU→10`）自训练反而会引入噪声、扭曲原有高质量语义几何结构。

---

## 6. 聚类评价指标与无标签稳定性诊断公式

由于聚类是无监督任务，预测簇编号 $\hat{c}_i \in \{0,\dots,K-1\}$ 与真实类别 $y_i \in \{0,\dots,K-1\}$ 之间只差一个任意置换，所有有监督评价指标必须具备**置换不变性**（且真实标签仅允许出现在 `src/evaluation/metrics.py` 的最终评分阶段）。

### 6.1 聚类准确率（ACC）

遍历所有 $K!$ 种一一对应置换映射 $m: \{0,\dots,K-1\} \to \{0,\dots,K-1\}$，取命中率最高者：

$$\mathrm{ACC} = \max_{m \in \mathcal{P}_K}\;\frac{1}{N}\sum_{i=1}^{N}\mathbb{1}\!\big[y_i = m(\hat{c}_i)\big].$$

实际计算不暴力枚举 $K!$，而是先构造 $K \times K$ 混淆列联表 $M_{jk} = \sum_i \mathbb{1}[\hat{c}_i = j,\; y_i = k]$，再通过 **Kuhn-Munkres（匈牙利算法）** 在 $O(K^3)$ 时间内求最大权匹配（`scipy.optimize.linear_sum_assignment`）。

### 6.2 归一化互信息（NMI）

将真实标签 $Y$ 与预测簇 $\hat{C}$ 视为两个离散随机变量，互信息 $I(Y; \hat{C}) = \sum_{j,k} P(j, k)\log \frac{P(j,k)}{P_Y(j)P_{\hat{C}}(k)}$，香农熵 $H(Y) = -\sum_j P_Y(j)\log P_Y(j)$。本项目采用与 `scikit-learn` 默认一致的**算术平均归一化**：

$$\mathrm{NMI}(Y, \hat{C}) = \frac{2\,I(Y;\hat{C})}{H(Y) + H(\hat{C})} \in [0, 1].$$

### 6.3 调整兰德指数（ARI，本项目核心指标）

设 $n_{jk}$ 为真实属第 $j$ 类且被分进第 $k$ 簇的样本数，行和 $a_j = \sum_k n_{jk}$，列和 $b_k = \sum_j n_{jk}$。兰德指数考察所有 $\binom{N}{2}$ 对样本“是否同组”的一致比例；ARI 在其基础上减去固定边缘分布下随机划分的期望一致数（Hubert & Arabie, 1985）：

$$\mathrm{ARI} = \frac{\displaystyle\sum_{j,k}\binom{n_{jk}}{2} - \frac{\left[\sum_j \binom{a_j}{2}\right]\left[\sum_k \binom{b_k}{2}\right]}{\binom{N}{2}}}{\displaystyle\frac{1}{2}\left[\sum_j \binom{a_j}{2} + \sum_k \binom{b_k}{2}\right] - \frac{\left[\sum_j \binom{a_j}{2}\right]\left[\sum_k \binom{b_k}{2}\right]}{\binom{N}{2}}}.$$

- **解释尺度**：完全一致时 $\mathrm{ARI} = 1$；独立随机划分的期望为 $0$；比随机还差时可取负值。注意 ARI 是**成对一致度**的期望校正值，不能直接等价于“逐张分类正确率”。

### 6.4 轮廓系数与簇均衡度诊断

- **轮廓系数（Silhouette）**：对样本 $i$，记 $a(i)$ 为其到同簇其余点的平均欧氏距离，$b(i)$ 为其到最近其他簇所有点的平均距离：
  $$s(i) = \frac{b(i) - a(i)}{\max\{a(i),\, b(i)\}} \in [-1, 1].$$
  因为高维欧氏距离随维度膨胀呈现集中现象，本项目协议明确规定**轮廓系数只在各方法自身隐空间内解释，严禁跨 10 / 128 / 784 维直接横向排名**。
- **簇均衡度**：记各簇占比 $r_k = |C_k|/N$，记录非空簇数 `n_nonempty` $= \sum_k \mathbb{1}[r_k > 0]$、最大簇占比 `max_fraction` $= \max_k r_k$、以及归一化簇熵 `entropy_norm` $= \frac{-\sum_{r_k>0} r_k \log r_k}{\log K} \in [0,1]$，用于检测空簇塌缩与严重偏斜。

### 6.5 无标签扰动稳定性（Macro Cluster Jaccard，E05）

在完全不读取任何标签的前提下，对同一已拟合模型输入同一批训练子集（2,000 张）的干净视图与加噪视图（$\sigma = 0.1$），分别得到簇集合 $A_k = \{i : \hat{c}_i^{\mathrm{clean}} = k\}$ 与 $B_k = \{i : \hat{c}_i^{\mathrm{noisy}} = k\}$（因属同一固定模型，簇号 $k$ 天然对齐）。宏平均簇 Jaccard 一致度定义为（见 `src/evaluation/selection.py::stability_evidence`）：

$$J = \frac{1}{K}\sum_{k=0}^{K-1}\frac{|A_k \cap B_k|}{|A_k \cup B_k|} \in [0, 1],$$

并附加硬门槛：干净视图与加噪视图均须保留全部 $K$ 个非空簇（剔除单簇塌缩模型）。如第 1.1 节所示，像素 K-means 在各向同性噪声下天然几乎不改边界，因此 $J$ 极高（0.94–0.96），但这仅证明“每次分得一样”，并不证明“分得符合语义”——这从数学上解释了 E05 观察到的**“最稳定 ≠ 最准确”**反例。
