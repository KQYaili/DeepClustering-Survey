# 执行顺序

| 阶段 | 命令 | 通过条件 |
| --- | --- | --- |
| 0 | `python -m unittest discover -s tests -v` | 指标、损失、队列路径、标签隔离、端到端成本、路径脱敏与 PPT 讲稿绑定全部通过 |
| 1 | `python scripts/run_queue.py` | 每个作业写出 `ok` 或 `failed` JSON，失败保留 traceback |
| 2 | `python scripts/eval_noise.py` | 只对已有权重做测试期加噪，不重训模型 |
| 3 | `python scripts/eval_train_stability.py` | 仅用训练侧预测生成 E05 的 NPZ 诊断 |
| 4 | `python scripts/analyze.py` | 汇总表（含 `benchmark_cost_frontiers.csv`、`selection.json`）与 13 张 300-DPI 科研图只从 `results/raw` 生成 |
| 5 | `python scripts/render_equations.py`、`python scripts/render_diagrams.py` | 生成 300-DPI 数学公式、架构原理图与动态读取实测数据的选型决策树 |
| 6 | `python scripts/build_pptx.py`、`python scripts/export_notes_docx.py` | 结果页全部从 CSV/JSON 动态读取，按稳定 Slide ID 严格校验 30 页讲稿绑定，输出 PPTX 与 Word 讲稿 |
| 7 | 通过 PowerPoint MCP / `scripts/ppt_com.py` 在打开状态下实时检查、渲染预览并导出 PDF | 避免覆盖锁定文件，确保无拉伸、无溢出、无错位 |

当前 JSON 队列共有 809 条成功记录。E01 使用 3 个 seed。E02 和 E03 的协议配置固定为 seed 0。E04 的同一特征空间聚类器比较按 seed 0 运行（AE/SimCLR+K-means 复用同切分 E01 seed 0），预训练 DINOv2/CLIP 的 K-means 另有 3-seed 稳定性表。E05 是在基准完成后增加的训练侧稳定性诊断（仅覆盖从零 9 候选且未纳入预测耗时），明确作为“稳定 ≠ 正确”的负面诊断结果，不能当作独立测试集验证或自动最佳方法推荐器。

E03 的全量、平衡、无异常、K=10 情况复用 E01 对应行，不重复训练。噪声实验把噪声加到测试图并截断回 `[0, 1]`，不在噪声上重新拟合。无标签随机子集的 epoch 预算保持不变，因此样本更少时会看到更多遍数据。类别不平衡子集使用训练集标签构造，评价仍在干净测试集上。

外部模型权重和完整运行归档用 `rclone copy`（或非破坏性上传脚本）追加到 Google Drive，禁止用删除式同步覆盖历史归档。GitHub 只保存代码、协议、表、图、轻量化嵌入文件（1.6 MB，用于离线复现 `fig_e01_embeddings.png`）和汇报文件，`results/checkpoints/` 与外部模型归档留在 Drive。
