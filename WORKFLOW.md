# 执行顺序

| 阶段 | 命令 | 通过条件 |
| --- | --- | --- |
| 0 | `python -m unittest discover -s tests -v` | 指标、损失、队列路径、标签隔离、端到端成本、路径脱敏、PPT 数据绑定与讲稿绑定全部通过 |
| 1 | `python scripts/run_queue.py` | 每个作业写出 `ok` 或 `failed` JSON，失败保留 traceback |
| 2 | `python scripts/eval_noise.py` | 只对已有权重做测试期加噪，不重训模型 |
| 3 | `python scripts/eval_train_stability.py` | 仅用训练侧预测生成无标签稳定性诊断的 NPZ |
| 4 | `python scripts/analyze.py` | 汇总表与结果图只从 `results/raw` 生成；图按幻灯片上的尺寸绘制 |
| 5 | `python scripts/render_illustrations.py`、`python scripts/render_equations.py` | 生成封面马赛克、像素距离示例、数据集样例、噪声示例和 LaTeX 公式（需要数据缓存、CIFAR-10 特征缓存和 pdflatex） |
| 6 | `python scripts/build_pptx.py`、`python scripts/export_notes_docx.py` | 30 页 PPT（24 页正文 + 6 页附录）与 Word 讲稿；所有数字从结果表读取，讲稿按页面 ID 绑定 |

协议见 `configs/protocol.yaml`。E01 使用 3 个种子。E02、E03、E04 最初只跑了种子 0；种子 1、2，E04 在 MNIST 与 Fashion-MNIST 上的扩展，以及关闭早停、固定 50 轮的 λ 扫描（`E02_fixed`）在运行前登记到同一份协议中。E03 中全量、平衡、无异常、K = 10 的情况复用 E01 对应的拟合。噪声只加在测试图上并截断回 `[0, 1]`。无标签随机子集的训练轮数上限不变，样本越少，每个样本被看到的次数越多。类别不平衡子集按训练集标签构造比例，评价仍在干净、均衡的测试集上。

外部模型权重和完整运行归档保存在 Google Drive，用 `rclone copy` 追加，不用删除式同步覆盖历史归档。GitHub 只保存代码、协议、结果表、图、轻量嵌入文件和汇报文件；`results/checkpoints/` 与 `results/cache/` 留在 Drive。
