# 原始任务书存档

> 来源：原 `D:\Program\git\readme.md`《Embodied AI Data Pipeline: Month 1 Sprint》（已归档删除）。
> 本文保留**原文的四周任务分解**，用于溯源与验收比对。
> **日常执行请以 `00_LEARNING_PLAN.md` 为准**——本计划的 30 天排期是它的重排与扩展。

---

## 一、项目目标（原文）

构建一个面向具身智能（Embodied AI）的**高吞吐量、低内存占用的 PyTorch 多模态数据加载管道**。
解决 HDF5/RLDS 等海量大文件在多进程读取时的 I/O 瓶颈、内存泄漏以及张量转换问题。

## 二、硬件与环境约束（原文）

| 约束 | 值 |
|:---|:---|
| GPU 显存上限 | **8GB**（需严格控制 `batch_size` 与预取队列，防止 DataLoader 导致 OOM） |
| 时间预算 | **20 小时 / 周** |
| 核心依赖 | `torch`, `h5py`, `numpy`, `opencv-python` |

---

## 三、原始四周任务分解（10 个 Task）

### 📌 Week 1：视觉流与 PyTorch 缝合（Vision-Tensor Bridge）

**核心聚焦**：自定义 Dataset，打通从底层的 OpenCV BGR 矩阵到模型所需的 RGB 张量流。

- [ ] **Task 1.1：Dataset 骨架搭建。** 创建 `EmbodiedDataset` 类，继承自 `torch.utils.data.Dataset`，实现 `__len__`。
- [ ] **Task 1.2：图像通道张量化。** 在 `__getitem__` 中，将读取的 BGR 图像转换至 RGB，利用 `np.transpose` 完成 `[H, W, C]` 到 `[C, H, W]` 的维度转换。
- [ ] **Task 1.3：数据归一化。** 将 `uint8` 类型的像素值，安全地除以 255.0，转换为 `float32` 类型的 PyTorch Tensor。
- **交付物**：一个可以通过 `dataset[0]` 正确返回 `{ "image": Tensor, "state": Tensor, "action": Tensor }` 字典的 Python 脚本。

### 📌 Week 2：HDF5 大文件与 I/O 破局（I/O Optimization）

**核心聚焦**：彻底解决具身智能中数十 GB 轨迹文件无法装入内存，且多进程读取容易死锁的问题。

- [ ] **Task 2.1：懒加载（Lazy Loading）实现。** 确保文件读取逻辑中只有在被索引切片时才将数据从硬盘拉取到内存。
- [ ] **Task 2.2：攻克 Fork 死锁危机。** **（极易踩坑）** 严禁在 `__init__` 中实例化 `h5py.File`。使用 `worker_init_fn` 或在首次调用 `__getitem__` 时按进程独立打开文件句柄，确保 `num_workers > 0` 时不引发进程崩溃。
- **交付物**：修改后的 Dataset 支持被传入 `DataLoader`，并在开启 `num_workers=4` 的情况下平稳运行，内存占用维持在 2GB 以下。

### 📌 Week 3：物理状态规范化与动作分块（Physics & Action Chunking）

**核心聚焦**：融入机器人学特性，处理高维度的位姿与动作序列。

- [ ] **Task 3.1：物理张量化。** 将机械臂 6-DoF 位姿与夹爪开合度合并为单一的一维 `state` Tensor。
- [ ] **Task 3.2：极值归一化（Min-Max Normalization）。** 编写预处理脚本，统计整个 HDF5 文件的动作极值，并在读取时将所有 `action` Tensor 严格缩放至 `[-1, 1]` 区间。
- [ ] **Task 3.3：动作分块（Action Chunking）。** 重写 `__getitem__` 逻辑：不再返回单一帧的动作，而是利用滑动窗口，返回未来 $K$ 步的连续动作序列矩阵（例如 `[K, 7]`），为解决策略发散做准备。
- **交付物**：能够输出含有时间序列动作特征（Temporal Action Sequence）的批量数据结构。

### 📌 Week 4：极限吞吐量压测（Performance Profiling）

**核心聚焦**：在显存有限的环境下，压榨 DataLoader 的极限性能，确保 GPU 计算无阻塞。

- [ ] **Task 4.1：内存锁页注入。** 在 DataLoader 中开启 `pin_memory=True`，实现从 CPU 到 GPU 的无缓冲快速内存拷贝。
- [ ] **Task 4.2：寻找 I/O 甜点。** 针对当前的硬件磁盘与 8GB 显存，反复调整 `batch_size`、`num_workers` 与 `prefetch_factor` 参数组合，记录数据。
- [ ] **Task 4.3：FPS 基准测试。** 编写一个纯遍历数据的 Profiler 循环，不进行模型推理，仅计算纯数据加载的 FPS（Frames Per Second）。
- **交付物**：一份 Markdown 格式的压测报告，记录不同参数组合下的吞吐率，并得出当前硬件下的最优 DataLoader 配置参数。

---

## 四、Task → 30 天计划映射

原文按"周"组织，本计划按"天"重排，并把原文省略的 PyTorch 地基补在 Day 1-11。

| 原 Task | 本计划落点 | 里程碑 |
|:---|:---|:---|
| Task 1.1 | Day 11-12 | M3 |
| Task 1.2 + 1.3 | Day 13 | M3 |
| Task 2.1 | Day 15-16 | M4 |
| Task 2.2 | Day 17-19 | M4 |
| Task 3.1 | Day 22 | M5 |
| Task 3.2 | Day 23 | M5 |
| Task 3.3 | Day 24 | M5 |
| Task 4.1 | Day 25 | M6 |
| Task 4.2 | Day 26 | M6 |
| Task 4.3 | Day 27 | M6 |

**新增（原文没有、本计划补充的部分）**：Day 1-11 的 PyTorch 地基
（Tensor / autograd / nn.Module / 训练循环 / Dataset 契约），
Day 7 / 14 / 21 / 28 四次周测，以及 Day 29-30 的端到端集成与验收展示。

> 注：原文 Week 1 Task 1.2 写的是 `np.transpose`；本项目统一用 PyTorch 的 `permute`（`T04`），
> 两者语义等价，但 `permute` 与后续 `unsqueeze`/`contiguous` 的衔接更自然。
