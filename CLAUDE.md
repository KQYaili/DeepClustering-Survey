# 会话入口

先读 `WORKFLOW.md` 和 `results/tables/failures.csv`（如果存在），再读 `configs/protocol.yaml`。

实验在 WSL2 里跑，解释器是 `/home/lgd/anaconda3/bin/python`。`PYTHONPATH` 需要包含仓库的 `src` 和 `/home/lgd/venvs/dc-extra`。数据目录是 `DATA_ROOT`，默认 `~/.cache/dc-survey/data`。

Windows 上的 PowerPoint 用 COM 改已经打开的 `presentation/deep-clustering-evolution-selection-guide.pptx`。不要在文件被 PowerPoint 锁住时用 python-pptx 覆盖保存。

汇报页数是 24 页正文。计划表里的模块加总曾是 22，多出来的两页是开场问题页和局限页。

完成一组原始结果后运行 `python scripts/analyze.py`，再重建 PPT。不要在表还是空的时候写具体的 ACC、NMI 或 ARI。
