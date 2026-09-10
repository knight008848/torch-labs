# 概念编号地图

> 编号规则：`<类别前缀><两位序号>`。所有文档、对话、错题记录统一引用此编号。
> 骨架见 `knowledge_graph.md`。

---

## T — Tensor 基础（枢纽 ①）

| 编号 | 概念 | 一句话 | 首现 |
|:---|:---|:---|:---|
| T01 | Tensor 创建 | `tensor`/`zeros`/`ones`/`arange`/`randn` | Day 2 |
| T02 | dtype | `uint8`/`float32`/`float64`/`long`，图像与模型的桥 | Day 2 |
| T03 | device | CPU ↔ CUDA，`.to()` 必须显式 | Day 2 |
| T04 | shape 手术 | `view`/`reshape`/`permute`/`transpose`/`squeeze` | Day 3 |
| T05 | 索引与掩膜 | 基础切片、高级索引、布尔掩膜 | Day 4 |
| T06 | 广播 | `[N,7]` op `[7]` 的隐式扩展规则 | Day 5 |
| T07 | in-place 陷阱 | `add_`/`mul_` 等破坏 autograd 的原因 | Day 5 |
| T08 | 拼接 | `cat`/`stack` 在 batch 维的应用 | Day 20 |

## G — 自动微分（枢纽 ②）

| 编号 | 概念 | 一句话 | 首现 |
|:---|:---|:---|:---|
| G01 | 计算图 | 动态图随前向传播实时构建 | Day 6 |
| G02 | requires_grad | 标记哪些张量需要求导 | Day 6 |
| G03 | backward | 反向传播，梯度写进 `.grad` | Day 6 |
| G04 | 梯度累积 | 不清零就累加，`zero_grad()` 的存在理由 | Day 6 |
| G05 | no_grad / detach | 切断计算图，省显存 | Day 6 |

## M — 模块化与训练（枢纽 ③）

| 编号 | 概念 | 一句话 | 首现 |
|:---|:---|:---|:---|
| M01 | nn.Module | 参数容器 + `forward` 契约 | Day 8 |
| M02 | parameters / buffers | 可学习参数 vs 不可学习状态 | Day 8 |
| M03 | state_dict | 模型序列化的标准格式 | Day 8、10 |
| M04 | 损失函数 | `MSELoss`/`CrossEntropyLoss` | Day 9 |
| M05 | 优化器 | `SGD`/`Adam`，`zero_grad→forward→backward→step` | Day 9 |
| M06 | 训练循环 | epoch/batch 双层循环 + train/eval 切换 | Day 10 |
| M07 | 保存与加载 | `torch.save` / `load_state_dict` | Day 10 |

## D — 数据接口（枢纽 ④，本仓库主角）

| 编号 | 概念 | 一句话 | 首现 |
|:---|:---|:---|:---|
| D01 | Dataset 契约 | `__len__` + `__getitem__` 两方法 | Day 11 |
| D02 | DataLoader | 批处理 / 打乱 / 多进程 / 预取的装配器 | Day 11 |
| D03 | collate_fn | 把样本列表拼成 batch 张量的策略 | Day 11、20 |
| D04 | num_workers | 子进程数，I/O 与计算的权衡旋钮 | Day 12、18 |
| D05 | worker_init_fn | 每进程独立初始化（本仓库用来开文件句柄） | Day 18 |
| D06 | drop_last / shuffle | 训练期语义，管道压测时要关掉 | Day 11 |

## P — 具身数据管道（本仓库主线，挂在枢纽 ④下）

| 编号 | 概念 | 一句话 | 首现 |
|:---|:---|:---|:---|
| P01 | BGR→RGB | OpenCV 读进来是 BGR，模型要 RGB | Day 13 |
| P02 | HWC→CHW | `permute(2,0,1)`，通道前置 | Day 13 |
| P03 | 归一化 | `uint8 / 255.0 → float32 [0,1]` | Day 13 |
| P04 | HDF5 懒加载 | 只索引才读盘，数十 GB 不进内存 | Day 15 |
| P05 | fork 死锁 | 父进程的 HDF5 句柄被 fork 继承后失效 | Day 17 |
| P06 | 逐进程句柄 | `worker_init_fn` 里按 pid 各自开文件 | Day 18 |
| P07 | 内存压测 | RSS 峰值监控，2GB 上限验证 | Day 19 |
| P08 | 动作分块 | 滑窗返回未来 K 步 `[K,7]` | Day 24 |
| P09 | Min-Max 归一化 | action 全量统计后缩放到 `[-1,1]` | Day 23 |
| P10 | 预取 | `prefetch_factor`，I/O 与计算重叠 | Day 26 |
| P11 | FPS Profiler | 纯遍历不推理，测纯数据加载吞吐 | Day 27 |

## X — 辅助工具（非学习目标，只解决具体问题）

| 编号 | 概念 | 用途 | 首现 |
|:---|:---|:---|:---|
| X01 | OpenCV 读图 | `imread` + `cvtColor`，仅作 BGR/RGB 对照 | Day 13 |
| X02 | pandas 记录 | 压测结果存表、算中位数 | Day 26 |
| X03 | matplotlib headless | `savefig` 替代 `imshow`（本环境无 GUI） | Day 9 |
| X04 | pathlib | 所有路径用 `Path` 拼接 | Day 1 |

---

## 概念类型索引

**枢纽（4 个）**：T（Tensor）、G（autograd）、M（nn.Module）、D（Dataset/DataLoader）

**附属 A（操作手段）**：T01-T08、G02、G05
**附属 B（桥）**：T02(dtype 桥)、P01、D03(collate_fn)、D05(worker_init_fn)、B05(pin_memory→Day 25)
**附属 C（时间序列层）**：P04、P08、P10

---

## 术语中英对照

| 中文 | 英文 | 编号 |
|:---|:---|:---|
| 张量 | Tensor | T |
| 自动微分 | autograd | G |
| 计算图 | computation graph | G01 |
| 数据加载器 | DataLoader | D02 |
| 拼接函数 | collate_fn | D03 |
| 工作进程初始化 | worker_init_fn | D05 |
| 锁页内存 | pin_memory | Day 25 |
| 动作分块 | action chunking | P08 |
| 懒加载 | lazy loading | P04 |
