# 数据集规格与准备

> 来源：原 `D:\Program\git\readme.md` 的「前期环境与数据准备」三节（已归档删除，内容全文迁移至此）。
> 本文是编写 `__getitem__` 切片逻辑的**唯一权威依据**——不要凭记忆切片。

---

## 1. 核心实战数据集：Robomimic（Lift 任务）

选型理由：体量适中、结构标准、含完整多模态图像与物理位姿，是具身智能的业界黄金标准测试集。

- **任务**：Lift（机械臂抓取方块）
- **体量**：约 **1.5GB**——适合本地高吞吐压测，且能有效控制系统内存与显存开销，避免 OOM

**下载指令**（在项目根目录执行）：

```bash
mkdir -p data && cd data
wget https://url.robomimic.github.io/datasets/lift/ph/image.hdf5 -O teleop_demo.hdf5
```

**备用方案（网络受限时）**：编写纯 NumPy/h5py 脚本生成 **100MB** 的结构化仿真 Mock 数据，
**键名、dtype、shape 必须与第 3 节完全一致**，否则后续真机数据接入时切片逻辑要重写。

> ⚠️ Day 12 的 `embodied_dataset.py` v0.1 就建立在 Mock 数据上；Day 16 再替换为真实数据。
> Mock 生成器必须保证结构等价，这是整条链路能否顺利切换的前提。

---

## 2. 数据视察工具（Data Viewer）

HDF5 是二进制层级格式，无法用文本编辑器查看。**在编写多进程管道之前，必须先用探针确认内部键值特征，坚决避免盲目切片。**

### 图形化方案（推荐）

下载并安装官方 **HDFView**。可像资源管理器一样直观展开 `teleop_demo.hdf5` 的内部目录树。

### 脚本化方案

利用 HDF5 的**懒加载（Lazy Loading）**机制快速打印全盘结构——此操作极度轻量，不会占用过多系统内存：

```python
import h5py


def print_hdf5_structure(name, obj):
    print(f"{name} - {type(obj)}")


with h5py.File("data/teleop_demo.hdf5", "r") as f:
    f.visititems(print_hdf5_structure)
```

> 这段脚本是 Day 15 `experiments/day_15_hdf5_probe.py` 的起点。

---

## 3. 核心目录结构规格

数据集解析后呈现如下高度嵌套结构。这是切片逻辑的核心依据，
也是提取多模态状态、动作与图像观察流的基础映射表。

| 路径 | 类型 / 形状 | 业务说明 |
|:---|:---|:---|
| `data/demo_0/obs/agentview_image` | `uint8 [N, 84, 84, 3]` | 遥操作录制的全局第三人称相机 RGB 视频帧序列 |
| `data/demo_0/obs/robot0_joint_pos` | `float64 [N, 7]` | 机械臂在任务执行周期内的 7 自由度关节物理状态序列（Proprioception） |
| `data/demo_0/actions` | `float64 [N, 7]` | 专家操作的目标动作序列，**后续训练 VLA 策略模型的最核心监督标签** |

> ⚠️ **路径含顶层 `data/` 组**（Day 1 踩过）。表里的 `data/` 是 HDF5 内部的一个**组**，
> 不是文件名、也不是相对目录。Mock 生成器与校验器都从 `experiments/day_01_env_and_data.py`
> 的 `DATASET_ROOT = "data"` 拼路径，别再手写 `demo_0/...`。
>
> 📌 **探针可能看到本表之外的东西**（待 Day 15 实测确认，勿提前写死）：
> 真实 robomimic 文件据其官方文档还带一个平级的 `mask/` 组，以及 `data` 组上的
> `env_args` / `total` 属性。本项目**不使用** `mask/`，Day 15 探针看到它属正常，
> 不要据此判定"文件损坏"；也**不要**在 Day 15 之前按它的存在写任何逻辑。

### 切片时要留意的三点

1. **`agentview_image` 是 `uint8` + HWC 轴序**——到模型输入需要 `T02`（dtype 转换）、
   `P02`（`permute` → CHW）、`P03`（`/255.0`）三步，见 Day 13。
2. **`robot0_joint_pos` 与 `actions` 都是 `float64`**——PyTorch 默认 `float32`，
   需显式 `.float()`，否则会和模型权重 dtype 不匹配（`T02`）。
3. **三个键的 N 必须对齐**——`state` 与 `image` 的帧数不一致是最隐蔽的 bug，
   Day 22 的验收标准专门有一条断言检查。
4. **`demo_0` 只是示例**——真实数据集含 `demo_0 ... demo_N`，遍历时要按 demo 分组再拼。

---

## 4. 与后续任务的对应

| 本文章节 | 用在哪一天 | 涉及编号 |
|:---|:---|:---|
| 下载 / Mock 生成 | Day 1、Day 12、Day 16 | P04 |
| visititems 探针 | Day 15 | P04 |
| 结构规格表 | Day 15-16、Day 22-24 | P04 P08 P09 |
| dtype / 轴序注意点 | Day 13、Day 22 | T02 P01 P02 P03 |
