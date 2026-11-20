# 进度追踪

> **规则 4**：每次学习后必须更新本文件。
> 每天收工前勾选完成项、记录薄弱点、写下下一步。
> 计划版本：30 天制（Day 29-30 为冲刺日，Day 29 兼作缓冲）

---

## 当前状态

| 项 | 值 |
|:---|:---|
| 当前 Day | 1（2026-09-11） |
| 当前周 | W1 地基 I：Tensor 与 autograd |
| 当前里程碑 | M1 已达成，M2 待办（Day 7 周测） |
| 累计投入 | 0.5 h |
| 本周投入 | 0.5 h / 20 h |

> 维护与复核不计入学习时长。下方「复工复核」记录了 09-11 之后的这次环境重验。

---

## 复工复核（2026-11-20）

> 距 Day 1（09-11）**68 天**。正式恢复 Day 2 之前重跑环境闸门，确认这 68 天里没有
> 静默腐坏——版本不会自己变好，但会自己变坏。

**结论：环境完整可用，Day 2 可以开始。** 细节见 `docs/env_notes.md`。

- `src/env_check.py` 全绿 `exit=0`：torch 2.10.0+cu128 / CUDA 12.8 / sm_75 /
  8192 MiB（空闲 7144）/ Python 3.10.19 / 依赖齐备。空闲显存与 9 月记录一致。
- **加固三项**：
  1. 必需清单补 `torchvision`——它此前从未被校验，而它和 torch 是硬绑定的
     （元数据精确钉 `torch (==2.10.0)`），错一个 minor 就报 `operator torchvision::nms
     does not exist`，且要到 Day 13 才会发作。
  2. 新增 GPU matmul 探针——`is_available()` 返回 `True` 只证明驱动看得见卡，不证明
     这个 wheel 编进了 sm_75 的 kernel。旧版会放行「看起来健康、一跑就炸」的状态。
  3. 新增 matplotlib 后端检查（WARN 级）——本机 WSLg 使后端为 `tkagg`，
     `plt.show()` 会弹窗**阻塞**而非报错，最容易被误认为「跑得慢」。
- **反向验证**：人为构造 no-cuda、kernel 缺失、backend FAIL 三种场景，确认都报
  `[fail]` **且退出码为 1**。校验器必须被证明会拒绝东西，否则「通过」没有意义。
- **复核环节抓到两个真 bug**（均已修）：`is_available()` 写在 `try` 外，驱动不匹配时
  会整脚本崩、后面所有检查都看不见；headless 探针的 FAIL 只 print 未计入失败清单，
  红字打出来而 `exit=0`。
- **脚手架 6 条已知缺陷**（P0×2 / P1×3 / P2×1）已登记进 `env_notes.md` 第 5 节。
  不影响 Day 2 开始，但会在对应那天变成理解障碍，尤其 P0 两条（`h5py` 持有规则自相
  矛盾、多模态轴序契约失效）分别在 Day 12 和 Day 20 兑现。
- **升级决定：不升**。torch 2.10.0 vs PyPI 2.14.1，落后 4 个 minor；理由与代价见
  `env_notes.md` 第 1 节。

**未完成（不阻塞 Day 2，但 Day 16 前必须解决）**：`data/raw/teleop_demo.hdf5` 仍是
Mock；`data/raw/test/` 残留 248 MB 待清理。

---

## 周测成绩

| 测试 | 日期 | 选择题 | 代码补全 | 实战 | 总分 | 是否 ≥60% |
|:---|:---|:---|:---|:---|:---|:---|
| 周测 1（Day 7） | — | /4 | /2 | /1 | — | ☐ |
| 周测 2（Day 14） | — | /4 | /2 | /1 | — | ☐ |
| 周测 3（Day 21） | — | /4 | /2 | /1 | — | ☐ |
| 周测 4（Day 28） | — | /4 | /2 | /1 | — | ☐ |

---

## 逐日记录

### 📌 W1（Day 1-7）｜地基 I：Tensor 与 autograd

- [x] **Day 1** 环境奠基与数据落地 ｜ 交付：`day_01_env_and_data.py`、`src/env_check.py` ｜ 耗时：0.5 h
- [ ] **Day 2** Tensor 三要素 dtype/shape/device ｜ 交付：`day_02_tensor_basics.py` ｜ 耗时：__
- [ ] **Day 3** 形状手术 view/reshape/permute ｜ 交付：`day_03_shape_ops.py` ｜ 耗时：__
- [ ] **Day 4** 索引、切片与布尔掩膜 ｜ 交付：`day_04_indexing.py` ｜ 耗时：__
- [ ] **Day 5** 广播与 in-place 陷阱 ｜ 交付：`day_05_broadcast.py` ｜ 耗时：__
- [ ] **Day 6** autograd 动态计算图 ｜ 交付：`day_06_autograd.py` ｜ 耗时：__
- [ ] **Day 7** 周测 1 + 复盘 ｜ 交付：`day_07_quiz_1.py` ｜ 耗时：__

**W1 复盘**

> Day 1 增量（Day 7 时再终稿）：

- 完成事项（Day 1）：`src/env_check.py` 通过（含 `torch built for CUDA 12.8` 实测校验）；
  `data/raw/teleop_demo.hdf5` 落地（6 demo / 5700 帧 / 116.1 MiB，
  路径含顶层 `data/` 组，逐字对齐 `dataset_spec.md` 第 3 节）；
  `docs/figs/day_01_mock_frame.png` 肉眼确认场景正确；目录骨架幂等校验；
  生成脚本**字节级确定性**已实测（两次 `--force` 的 md5 一致）
- 做错的练习及分析（记入 `error_log.md`）：Day 1 两条，均归类 **K**——
  ① 计划文档写 `pydata`、实测只有 `embodied_ai` 有 torch（文档 ≠ 实测）；
  ② Mock 漏掉顶层 `data/` 组，且校验器与生成器"一起错"而互相放行（自洽 ≠ 正确）。
  两条都由 code-reviewer 复核环节抓出，②已在提交前修掉并补了 9 个反例自测
- 新发现的薄弱点：
  1. **真实数据未落地**——`data/raw/teleop_demo.hdf5` 目前是 Mock。robomimic 原始 URL
     探测返回 404，且本机 curl 直连 HTTPS 报 TLS 校验失败（exit 60），需先排查证书/网络再下载。
     不影响 Day 2-14（Mock 结构等价），但 **Day 16 之前必须解决**
  2. **Mock 是 RGB 按构造写的**，`channel_order` 属性已显式标注。这是个便利也是一个陷阱：
     Day 13 在 Mock 上做的通道序实验**不能推断真实数据**，真判定仍留 Day 16
- 复述检验①（Tensor 与 NumPy 的关系）是否流畅：Day 7 执行
- 下一步计划（Day 2）：Tensor 三要素 dtype/shape/device（T01 T02 T03 B01），
  练习素材直接用今天落地的 `[N,84,84,3] uint8` 与 `[N,7] float64`

### 📌 W2（Day 8-14）｜地基 II：模块化、训练循环与数据接口

- [ ] **Day 8** nn.Module 与参数容器 ｜ 交付：`day_08_nn_module.py` ｜ 耗时：__
- [ ] **Day 9** 损失函数与优化器 ｜ 交付：`day_09_loss_optim.py` ｜ 耗时：__
- [ ] **Day 10** 完整训练循环 + 保存加载 ｜ 交付：`day_10_train_loop.py` ｜ 耗时：__
- [ ] **Day 11** Dataset / DataLoader 接口契约 ｜ 交付：`day_11_dataset_contract.py` ｜ 耗时：__
- [ ] **Day 12** 管道模块 A 起步：Dataset 骨架 ｜ 交付：`src/embodied_dataset.py` v0.1 ｜ 耗时：__
- [ ] **Day 13** 视觉张量缝合：通道序 / HWC→CHW ｜ 交付：`day_13_vision_bridge.py` ｜ 耗时：__
- [ ] **Day 14** 周测 2 + 复盘 ｜ 交付：`day_14_quiz_2.py` ｜ 耗时：__

**W2 复盘**
- 完成事项：
- 做错的练习及分析：
- 新发现的薄弱点：
- 复述检验②（Dataset 与 DataLoader 的分工）是否流畅：
- Day 13 通道序对比结论（尚未判定真实通道序，留到 Day 16）：
- 下一步计划：

### 📌 W3（Day 15-21）｜管道：真实 HDF5 与 I/O 破局

- [ ] **Day 15** HDF5 结构探针与懒加载 ｜ 交付：`day_15_hdf5_probe.py` ｜ 耗时：__
- [ ] **Day 16** 真实 hdf5 替换 Mock ｜ 交付：`src/embodied_dataset.py` v0.2 ｜ 耗时：__
- [ ] **Day 17** fork 死锁原理与复现 ｜ 交付：`day_17_fork_deadlock.py` ｜ 耗时：__
- [ ] **Day 18** worker_init_fn 修复 ｜ 交付：`src/embodied_dataset.py` v0.3 ｜ 耗时：__
- [ ] **Day 19** 内存占用实测与泄漏排查 ｜ 交付：`day_19_memory_profile.py` ｜ 耗时：__
- [ ] **Day 20** 多模态字典契约与自定义 collate_fn ｜ 交付：`src/embodied_dataset.py` v0.4 ｜ 耗时：__
- [ ] **Day 21** 周测 3 + 复盘 ｜ 交付：`day_21_quiz_3.py` ｜ 耗时：__

**W3 复盘**
- 完成事项：
- 做错的练习及分析：
- 新发现的薄弱点：
- 复述检验③（fork 死锁的生活类比）是否流畅：
- `agentview_image` 真实通道序判定结果（Day 16）：
- `num_workers=4` 内存峰值实测值：__ GB（目标 <2GB）
- 下一步计划：

### 📌 W4（Day 22-28）｜状态规范化、动作分块与压测

- [ ] **Day 22** 物理状态张量化 ｜ 交付：`day_22_state_tensor.py` ｜ 耗时：__
- [ ] **Day 23** Min-Max 归一化到 [-1,1] ｜ 交付：`src/normalize.py` + `action_stats.json` ｜ 耗时：__
- [ ] **Day 24** 动作分块滑窗 [K,7] ｜ 交付：`day_24_action_chunking.py` ｜ 耗时：__
- [ ] **Day 25** pin_memory 与 8GB 显存约束 ｜ 交付：`day_25_pin_memory.py` ｜ 耗时：__
- [ ] **Day 26** I/O 甜点扫描 ｜ 交付：`day_26_param_sweep.py` + `sweep.csv` ｜ 耗时：__
- [ ] **Day 27** FPS 基准与压测报告 ｜ 交付：`day_27_fps_bench.py` + `docs/BENCHMARK.md` ｜ 耗时：__
- [ ] **Day 28** 周测 4 + W4 复盘 ｜ 交付：`day_28_quiz_4.py` ｜ 耗时：__

**W4 复盘**
- 完成事项：
- 做错的练习及分析：
- 新发现的薄弱点：
- 复述检验④（瓶颈在哪、怎么证明）是否流畅：
- `robot0_joint_pos` 的 7 维语义判定结果（Day 22）：
- 最优 DataLoader 配置：batch_size=__ / num_workers=__ / prefetch_factor=__ / pin_memory=__
- 四周错题分布（C/A/E/K 各几次）：
- 下一步计划：

### 📌 冲刺（Day 29-30）｜端到端集成与验收展示

- [ ] **Day 29** 端到端集成 + 报告定稿 ｜ 交付：`src/pipeline_demo.py` + `docs/REPORT.md` ｜ 耗时：__
- [ ] **Day 30** 期末验收 + 项目展示 ｜ 交付：演示 + 结业复盘 ｜ 耗时：__

**冲刺复盘**
- 前面各天是否有欠账？在 Day 29 清偿了哪些：
- 端到端跑通是否一次成功？卡在哪：
- 展示时被问住的问题：
- 30 天能力增量总结：
- 下一步方向（Month 2）：

---

## 薄弱点汇总

> 每周末更新。正确率 <60% 的模块需安排补习日（规则 7）。
> 30 天制下，欠账优先用 **Day 29** 集中清偿，不顺延打乱结构。

| 概念编号 | 概念 | 出错次数 | 是否已补习 | 备注 |
|:---|:---|:---|:---|:---|
| — | — | — | — | — |

---

## 里程碑勾选

- [x] M1 环境就绪 + 数据落地（Day 1）
- [ ] M2 Tensor/autograd 打通（Day 7）
- [ ] M3 `dataset[0]` 返回三件套字典（Day 13）
- [ ] M4 `num_workers=4` 平稳 + 内存 <2GB（Day 19）
- [ ] M5 动作分块 `[K,7]` 输出（Day 24）
- [ ] M6 压测报告出炉（Day 27）
- [ ] M7 四周知识收口（Day 28）
- [ ] M8 项目验收展示（Day 30）
