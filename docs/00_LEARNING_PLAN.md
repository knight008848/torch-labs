# PyTorch 28 天学习计划

> 建立日期：2026-09-10
> 起点：Day 1 = 2026-09-11（顺延一天用于环境准备确认）
> 时间预算：20 小时/周 ≈ 2.8 小时/天
> 主线来源：`source_brief.md`《Embodied AI Data Pipeline: Month 1 Sprint》（原 readme.md 存档）
> 骨架来源：`knowledge_graph.md`（枢纽图谱）、`domain_map.md`（概念编号）

---

## 〇、一句话总纲

**先把 PyTorch 的四个枢纽打通（Day 1-14），再把 `source_brief.md` 的四周数据管道冲刺压缩进 Day 15-27（Day 10 起已开始），Day 28 验收展示。**

OpenCV 与 pandas 全程只做辅助，不在学习目标内。

---

## 一、总览：四周地图

| 周 | 天数 | 主题 | 枢纽定位 | 周测 |
|:---|:---|:---|:---|:---|
| **W1** | Day 1-7 | 地基 I：Tensor 与 autograd | 建立枢纽 ① ② | Day 7 |
| **W2** | Day 8-14 | 地基 II：nn.Module、训练循环、Dataset 契约 | 建立枢纽 ③ ④，管道模块 A 起步 | Day 14 |
| **W3** | Day 15-21 | 管道：真实 HDF5 与 I/O 破局 | 深化枢纽 ④（`readme` W2） | Day 21 |
| **W4** | Day 22-28 | 管道：状态规范化、动作分块、吞吐压测 | 深化枢纽 ④（`readme` W3+W4） | Day 28 |

**依赖链**（详见 `knowledge_graph.md`）：

```
Tensor ──> autograd ──> nn.Module ──┐
   │                                 ├──> 训练循环
   └──────> Dataset/DataLoader ──────┘
                    │
                    └──> 具身数据管道（本仓库主线，Day 10-27）
```

> **注意**：本计划对枢纽③（nn.Module）只做"链路验证"级别的学习（Day 8-10），
> 火力集中在枢纽①（Tensor）和枢纽④（Dataset/DataLoader）。

---

## 二、学习机制

### 2.1 五步教学法（每个模块必走）

```
地图 ──> 概念 ──> 练习 ──> 检查 ──> 复盘
 │
 └─ 每一步都先回答"当前内容在谱系中的位置"
```

1. **地图**：先说清本模块在依赖链的哪一环，引用 `domain_map.md` 编号
2. **概念**：每个概念讲足五要素
   - 一句话解释（≤30 字）
   - 生活类比（中国人熟悉的日常场景）
   - 技术解释（≤200 字）
   - 真实案例（工业/科研场景）
   - 一个 3 分钟内能做完的小练习
3. **练习**：当天产出一个可运行 `.py`
4. **检查**：对照当天的"验收标准"逐条勾
5. **复盘**：写 `progress.md` + `error_log.md`

### 2.2 概念类型标注（每个概念讲完必标）

- **枢纽概念** → 强调"它能用来解释什么"（被依赖面）
- **附属概念** → 指明"它挂在哪个枢纽下"

### 2.3 复述检验（费曼检查）

每个模块结束时**不查资料**复述三点：
① 本模块枢纽是什么、为什么是枢纽；② 它与其他枢纽的依赖链方向；③ 一个生活类比。

复述题见 `knowledge_graph.md` 第五节。

### 2.4 每日交付（规则 3）

- 文件名：`experiments/day_NN_主题.py`
- 文件头必须含：`Day NN / Date / Goal / Runtime`（模板见 `daily_template.md`）
- **验收标准不通过 = 当天不算完成**

### 2.5 每周阶段测试（规则 5）

Day 7 / 14 / 21 / 28 为测试日，题型固定：

| 题型 | 数量 | 说明 |
|:---|:---|:---|
| 选择题 | 4 道 | 覆盖本周概念，含至少 1 道易混对照题 |
| 代码补全 | 2 道 | 留空关键 API |
| 实战题 | 1 道 | 独立写一个 30 行内的小脚本 |

### 2.6 薄弱点调整（规则 7）

- 某模块测验正确率 **< 60%** → 后续安排补习日
- 同一概念反复出错 → 下一模块安排变体练习
- 错题归类见 `error_log.md`，某一代号累计 ≥3 次触发预警

---

## 三、逐日计划

### 📌 W1（Day 1-7）｜地基 I：Tensor 与 autograd

> **枢纽定位**：引入枢纽 ①（Tensor）与枢纽 ②（autograd）。
> 本模块**不引入新枢纽之外的东西**，目的是把后面一切的地基砸实。
> 依赖前置：Python 基础 + NumPy 基本用法。

| Day | 主题 | 涉及编号 | 交付物 | 验收标准 |
|:---|:---|:---|:---|:---|
| **1** | 环境奠基与数据落地 | X04 | `experiments/day_01_env_and_data.py`、`src/env_check.py` | ① WSL pydata 环境装好 torch（CUDA 12.x 版）+ h5py + pandas；② 脚本打印 torch 版本 / `cuda.is_available()` / GPU 名 / 显存总量；③ 建好目录骨架；④ `data/raw/teleop_demo.hdf5` 存在且 h5py 能打开，**或** Mock 生成器产出 ≥100MB 结构化 HDF5（下载指令与规格见 `dataset_spec.md`） |
| **2** | Tensor 三要素：dtype / shape / device | T01 T02 T03 B01 | `experiments/day_02_tensor_basics.py` | ① 手写创建 5 种 dtype 张量；② CPU→CUDA→CPU 往返并断言值不变；③ 解释清"图像用 uint8、模型用 float32"的原因；④ **演示 `torch.from_numpy` 共享内存**：改 numpy 原数组，张量跟着变 |
| **3** | 形状手术：view / reshape / permute | T04 | `experiments/day_03_shape_ops.py` | ① 把真实 84×84×3 图像 permute 到 3×84×84；② 证明 permute 后 `.is_contiguous() == False`，并说明为何 `view` 会在此报错；③ 用 matplotlib 存一张"轴序错乱"的图作为反例 |
| **4** | 索引、切片与布尔掩膜 | T05 | `experiments/day_04_indexing.py` | ① 从 `[N,7]` 中取出指定轨迹段；② 用布尔掩膜筛出夹爪开合 > 0.5 的帧；③ 说明 `[1,7]` 与 `[7]` 的区别（保留维度） |
| **5** | 广播与 in-place 陷阱 | T06 T07 | `experiments/day_05_broadcast.py` | ① 用广播把 `[N,7]` 归一化，不用 for 循环；② **故意**写一个 in-place 操作让反向传播报错，抄下报错原文；③ 说明 `a += b` 与 `a.add_(b)` 在 autograd 下的差异 |
| **6** | autograd 动态计算图 | G01 G02 G03 G04 G05 | `experiments/day_06_autograd.py` | ① 手算 y=x² 在 x=3 处梯度（=6），用 `backward()` 验证；② 演示不调 `zero_grad()` 的梯度累积现象；③ 用 `torch.no_grad()` 对比显存/耗时 |
| **7** | **周测 1 + 复盘** | T01-T07 G01-G05 | `experiments/day_07_quiz_1.py` | 选择 4 + 补全 2 + 实战 1；正确率 ≥60%；完成复述检验① |

**W1 生活类比锚点**：Tensor 就是"多维 Excel 表格"，dtype 是单元格格式，device 是"存在本地还是云盘"。

---

### 📌 W2（Day 8-14）｜地基 II：模块化、训练循环与数据接口

> **枢纽定位**：引入枢纽 ③（nn.Module）与枢纽 ④（Dataset/DataLoader）。
> 枢纽③只学到"能验证链路通"；枢纽④是本仓库主角，Day 11 建立契约后一路深化到 Day 27。
> 本周末尾开始接入 `source_brief.md` 的 **Week 1：视觉流与 PyTorch 缝合**。

| Day | 主题 | 涉及编号 | 交付物 | 验收标准 |
|:---|:---|:---|:---|:---|
| **8** | nn.Module 与参数容器 | M01 M02 M03 | `experiments/day_08_nn_module.py` | ① 自定义一个极简 MLP；② 打印 `state_dict()` 的参数名与形状；③ 说清 `parameters()` 与 `buffers()` 的区别 |
| **9** | 损失函数与优化器 | M04 M05 X03 | `experiments/day_09_loss_optim.py` | ① 用 `MSELoss` 拟合 y=2x+1；② 对比 SGD 与 Adam 的收敛曲线，用 matplotlib `savefig` 存图（**禁止 imshow**）；③ 默写 `zero_grad→forward→backward→step` 四步 |
| **10** | 完整训练循环 + 保存加载 | M06 M07 | `experiments/day_10_train_loop.py`、`models/` | ① 跑满 5 个 epoch；② 正确切换 `train()`/`eval()`；③ `torch.save` → `load_state_dict` 往返后预测值完全一致；④ 链路闭环：数据→张量→训练→保存 |
| **11** | Dataset / DataLoader 接口契约 | D01 D02 D03 D06 | `experiments/day_11_dataset_contract.py` | ① 实现 `__len__`/`__getitem__`；② `DataLoader(batch_size=4)` 取出一个 batch；③ 验证 `dataset[0]` 与 `next(iter(loader))` 第 0 个样本形状一致；④ 说清 `shuffle`/`drop_last` 在压测时为何要关 |
| **12** | 管道模块 A 起步：EmbodiedDataset 骨架 | D04 P04 | `src/embodied_dataset.py` v0.1 | `dataset[0]` 返回 `{"image": Tensor, "state": Tensor, "action": Tensor}` 字典；**此版先用 Mock 数据跑通** |
| **13** | 视觉张量缝合：BGR→RGB / HWC→CHW / 归一化 | P01 P02 P03 X01 | `experiments/day_13_vision_bridge.py` | ① 断言输出 `shape==(3,84,84)`、`dtype==float32`、`0<=min<=max<=1`；② 用 OpenCV `imread`+`cvtColor` 做 BGR/RGB 对照，matplotlib 并排 `savefig`；③ 处理 BGR/RGB 搞反导致的"蓝脸"反例并截图 |
| **14** | **周测 2 + 复盘** | M01-M07 D01-D06 | `experiments/day_14_quiz_2.py` | 覆盖 W1+W2；正确率 ≥60%；完成复述检验② |

**交付物对照**（`source_brief.md` Week 1 Task 1.1-1.3）：Day 11 完成 Task 1.1，Day 13 完成 Task 1.2 + 1.3。

---

### 📌 W3（Day 15-21）｜管道：真实 HDF5 与 I/O 破局

> **枢纽定位**：全部挂在枢纽 ④（Dataset/DataLoader）下，是附属 C（时间序列层）与附属 B（桥）的密集区。
> 本模块对应 `source_brief.md` 的 **Week 2：HDF5 大文件与 I/O 破局**，是全计划**最容易踩坑**的一周。

| Day | 主题 | 涉及编号 | 交付物 | 验收标准 |
|:---|:---|:---|:---|:---|
| **15** | HDF5 结构探针与懒加载机制 | P04 | `experiments/day_15_hdf5_probe.py` | ① 用 `visititems` 打印 robomimic 全树；② 对照 `dataset_spec.md` 第 3 节的结构规格表核对 `agentview_image`/`robot0_joint_pos`/`actions` 的 dtype 与 shape；③ 用内存对比证明"切片只读一帧" |
| **16** | 用真实 hdf5 替换 Mock | P04 | `src/embodied_dataset.py` v0.2 | 真实数据下 `dataset[0]` 通过；demo_0 的全部 N 帧均可索引且无越界 |
| **17** | fork 死锁原理与**复现** | P05 | `experiments/day_17_fork_deadlock.py` | ① 按 `source_brief.md` Task 2.2 **故意**在 `__init__` 里实例化 `h5py.File`；② 在 `num_workers=4` 下复现报错/挂起；③ 抄下报错原文，写清根因（句柄被 fork 继承） |
| **18** | worker_init_fn 修复 | P06 D05 | `src/embodied_dataset.py` v0.3 | ① `num_workers=4` 平稳跑完 100 个 batch 无报错无挂起；② 与 Day 17 的失败版做对照记录 |
| **19** | 内存占用实测与泄漏排查 | P07 | `experiments/day_19_memory_profile.py` | ① 记录 `num_workers=0/2/4` 下 RSS 峰值；② 峰值 < 2GB（`source_brief.md` Task 2.1 硬指标）；③ 说清句柄必须在何时关闭 |
| **20** | 多模态字典契约与批量化 | T08 D03 | `src/embodied_dataset.py` v0.4、契约文档 | ① batch 形状：image `[B,3,84,84]`、state `[B,7]`、action `[B,7]`；② 覆盖 3 个边界用例：index 越界、N < K、文件缺失；③ 错误处理和边界情况必须有示范 |
| **21** | **周测 3 + 复盘** | P04-P07 D03-D05 | `experiments/day_21_quiz_3.py` | 覆盖 W3；正确率 ≥60%；完成复述检验③（"为什么不能在 `__init__` 开文件"，用生活类比） |

**交付物对照**（`source_brief.md` Week 2 Task 2.1-2.2）：Day 15-16 完成 Task 2.1，Day 17-19 完成 Task 2.2。

---

### 📌 W4（Day 22-28）｜管道：状态规范化、动作分块与吞吐压测

> **枢纽定位**：仍挂在枢纽 ④ 下。前半是附属 C（时间序列层，动作分块），后半是附属 B（桥，pin_memory）。
> 对应 `source_brief.md` 的 **Week 3（物理状态与动作分块）** 与 **Week 4（性能压测）**。

| Day | 主题 | 涉及编号 | 交付物 | 验收标准 |
|:---|:---|:---|:---|:---|
| **22** | 物理状态张量化（6-DoF + 夹爪） | P09 | `experiments/day_22_state_tensor.py` | ① 把 `robot0_joint_pos [N,7]` 与图像观测帧对齐；② 逐维注释 state 向量的物理含义；③ 断言 state 与 image 的 N 一致 |
| **23** | Min-Max 归一化到 `[-1,1]` | P09 | `src/normalize.py`、`data/processed/action_stats.json` | ① 全量统计 action 极值并落盘；② 归一化后实测 `min>=-1` 且 `max<=1`；③ 处理 `max==min` 的除零边界 |
| **24** | 动作分块 Action Chunking（滑窗 `[K,7]`） | P08 | `experiments/day_24_action_chunking.py` | ① 返回未来 K 步动作矩阵 `[K,7]`（`source_brief.md` Task 3.3）；② 末帧不足 K 的边界策略（pad 或 repeat 二选一）写进注释并实现；③ K=8 与 K=16 各跑一遍 |
| **25** | pin_memory 与 8GB 显存约束 | B05 | `experiments/day_25_pin_memory.py` | ① 对比 `pin_memory=True/False` 的首 batch 拷贝耗时；② 在 8GB 显存下确认不 OOM；③ 说清 `pin_memory` 与 `.to('cuda')` 的分工 |
| **26** | I/O 甜点扫描 | P10 X02 | `experiments/day_26_param_sweep.py`、`data/processed/sweep.csv` | ① 网格扫描 `batch_size × num_workers × prefetch_factor`（`source_brief.md` Task 4.2）；② 每格跑 3 次取中位数，用 pandas 整理成 CSV；③ 画出热力图 `savefig` |
| **27** | FPS 基准 Profiler 与压测报告 | P11 X02 X03 | `experiments/day_27_fps_bench.py`、`docs/BENCHMARK.md` | ① 纯遍历数据、**不做模型推理**，输出 FPS（`source_brief.md` Task 4.3）；② 报告含参数对照表 + 最优配置结论 + 硬件环境说明 |
| **28** | **期末验收 + 项目展示** | 全部 | `docs/REPORT.md`、`src/pipeline_demo.py` | ① 一键端到端：HDF5 → 批量张量；② 复现 Day 27 的最优配置；③ 完成复述检验④；④ 通过全部四次周测的错题不再犯；⑤ 展示时长 ≤10 分钟 |

**交付物对照**（`source_brief.md` Week 3/4）：Day 22-24 完成 Task 3.1-3.3，Day 25-27 完成 Task 4.1-4.3。

---

## 四、里程碑与验收

| 里程碑 | 日期 | 交付物 | 状态 |
|:---|:---|:---|:---|
| M1 环境就绪 + 数据落地 | Day 1 | `env_check.py` 通过 + HDF5 可读 | ☐ |
| M2 Tensor/autograd 打通 | Day 7 | 周测 1 ≥60% | ☐ |
| M3 `dataset[0]` 返回三件套字典 | Day 13 | `embodied_dataset.py` v0.1 | ☐ |
| M4 `num_workers=4` 平稳运行 + 内存 <2GB | Day 19 | v0.3 + 内存记录 | ☐ |
| M5 动作分块 `[K,7]` 输出 | Day 24 | v0.5 | ☐ |
| M6 压测报告出炉 | Day 27 | `BENCHMARK.md` | ☐ |
| M7 项目验收展示 | Day 28 | `REPORT.md` + demo | ☐ |

---

## 五、硬约束（完整版见 `AGENTS.md`）

1. **Headless**：禁止 `cv2.imshow` / `cv2.waitKey` / `cv2.createTrackbar` / `cv2.setMouseCallback`。
   一律用 matplotlib `savefig`。
2. **显存**：8GB 上限，`batch_size` 与 `prefetch_factor` 必须控制（`source_brief.md` 约束）。
3. **编码规范**：注释用英文；优先函数式风格；路径用 `pathlib.Path`；文件头含 Day/Date/Goal/Runtime。
4. **错误处理**：每个脚本必须示范边界情况处理，禁止省略。
5. **单日新概念 ≤3 个**。
6. **禁止直接给练习完整答案**（只给提示）。
7. **代码提交前必须过 code-reviewer agent 审核**。
8. **范围**：不引入本计划外的深度学习内容（不做模型架构、不做训练调参、不做多模态对齐）。
   本仓库只到"数据管道"为止。

---

## 六、进度追踪

- 每日勾选：`progress.md`
- 错题记录：`error_log.md`
- 周测成绩：`progress.md` 的周测表
