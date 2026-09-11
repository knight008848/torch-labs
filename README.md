# torch-labs — PyTorch 30 天：从 Tensor 到具身数据管道

> 建立日期：2026-09-10
> 周期：30 天（Day 1-30），Day 7/14/21/28 为周测日，Day 29-30 为冲刺与验收
> 时间预算：20 小时/周（≈2.8 小时/天）

## 这个仓库是什么

一份**全新的 PyTorch 学习计划**。以 `docs/source_brief.md` 的《Embodied AI Data Pipeline: Month 1 Sprint》为主线，
把它的四周数据管道冲刺，扩展成"先打地基、再上工程"的 30 天路线。

**OpenCV 与 pandas 在本计划中是辅助角色**，不是学习目标：

| 角色 | 用途 | 出现位置 |
|:---|:---|:---|
| **PyTorch** | 主线，唯一的学习目标 | 全程 |
| OpenCV | 只用来 `imread` 读图、`cvtColor` 做 BGR→RGB 对照 | Day 13 |
| pandas | 只用来把压测结果整理成表、算中位数 | Day 26-27 |
| matplotlib | headless 环境下的可视化出口（`savefig`） | 全程 |

## 最终目标

一个可展示的 PyTorch 数据管道：HDF5（robomimic Lift）→ 多模态批量张量，
在 8GB 显存约束下跑出可复现的 FPS 基准，并产出压测报告。

对应 `source_brief.md` 的四个交付物，全部收敛到 Day 30 的验收。

## 目录结构

```
torch-labs/
├── README.md                  # 本文件
├── pyproject.toml             # ★ 依赖清单与最低版本（唯一出处）
├── AGENTS.md                  # ★ agent 指令入口（跨工具约定，原 CLAUDE.md 的 PyTorch 改写版）
├── CLAUDE.md                  # 一行指针，@ 导入 AGENTS.md
├── docs/
│   ├── 00_LEARNING_PLAN.md    # ★ 30 天主计划（唯一权威路线）
│   ├── knowledge_graph.md     # ★ PyTorch 枢纽知识图谱（骨架）
│   ├── domain_map.md          # ★ 概念编号地图（引用用）
│   ├── dataset_spec.md        # ★ 数据集规格与准备（robomimic / HDF5 结构）
│   ├── source_brief.md        # 原始任务书存档（10 个 Task 原文）
│   ├── progress.md            # 进度追踪（每天更新）
│   ├── error_log.md           # 错题四分类表（C/A/E/K）
│   ├── daily_template.md      # 每日交付文件头模板
│   ├── figs/                  # （运行期产出）matplotlib savefig 落图处
│   ├── BENCHMARK.md           # （Day 27 产出）压测报告
│   └── REPORT.md              # （Day 29 定稿）项目验收报告
├── data/
│   ├── raw/                   # teleop_demo.hdf5 或 Mock 数据
│   └── processed/             # 归一化统计、sweep.csv
├── experiments/               # day_NN_主题.py 每日交付
├── notebooks/                 # 探索性 notebook
├── src/                       # 可复用代码（embodied_dataset.py 等）
└── models/                    # checkpoint
```

> 注：`data/ experiments/ notebooks/ src/ models/` 已于 Day 1（2026-09-11）建立。
> 当前进度见 `docs/progress.md`。

## 环境

WSL的 conda环境可以更好的发挥torch的全部性能，依赖清单与最低版本
**以 `pyproject.toml` 为唯一出处**，本文件不再复述版本号，免得两处漂移。

```bash
source ~/miniforge3/bin/activate embodied_ai
python -m pip install -e .     # 装齐依赖；本机已满足时是空操作
python src/env_check.py        # 实测校验，任一项不过即退出码 1
```

清单里是**下限不是锁版**：本机跑第二行是空操作，但换台机器重建时 pip 会挑满足下限的
最新版。三条硬约束不在这份清单的保证范围内，重建后**必须**靠 `env_check.py` 复核：

- **torch 必须是 CUDA 12.x 构建**——`env_check.py` 会拦下别的版本。同时新版 torch
  可能不再为 sm_75 出构建，而本机是 RTX 2070 Max-Q。
- **显存 8GB**——`batch_size` / `prefetch_factor` / `num_workers` 的上限由它定，
  任何 DataLoader 配置变更都要在报告里记录显存占用。
- **headless**——无 GUI，可视化一律 `matplotlib` `savefig`，禁止 `cv2.imshow`
  系列；只有真调用才会抛错，别用 `hasattr` 判断。

环境细则见 `AGENTS.md` 的「环境」一节。

## 怎么用这份计划

1. 先读 `docs/00_LEARNING_PLAN.md` 的"总览"与"学习机制"两节，建立全局地图。
2. 每天开工前读当天的**枢纽定位**，收工后勾 `docs/progress.md`。
3. 每天必须产出一个可运行的 `.py`，验收不通过 = 当天不算完成。
4. 做错的题记进 `docs/error_log.md`，按 C/A/E/K 分类。
5. Day 7 / 14 / 21 / 28 为阶段测试日。

## 相关文档

- `AGENTS.md` — 领域学习工程师指令（教学法、编码规范、headless 与 git 约束）
- `docs/source_brief.md` — 原始《Embodied AI Data Pipeline: Month 1 Sprint》四周任务书存档
- `docs/dataset_spec.md` — 数据集规格与准备（robomimic 结构、下载、探针工具）
