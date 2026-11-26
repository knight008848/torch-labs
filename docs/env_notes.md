# 环境运维笔记

> **定位**：记录「换台机器或重装就会再踩一次」的事实。不记会话流水。
> 学习进度在 `progress.md`，错题在 `error_log.md`，两者与本文无关。
> **不是权威**：任何一条与 `python src/env_check.py` 的实测冲突，以实测为准（见 Day 1 错题）。

---

## 1. 环境拓扑

| 项 | 值 | 说明 |
|:---|:---|:---|
| 解释器 | `~/miniforge3/envs/embodied_ai/bin/python` | Python 3.10.19，压在 `requires-python` 下限上 |
| torch | 2.10.0+cu128 | `+cu128` 是编译时链接的 CUDA；`torch.version.cuda` 报 12.8 |
| GPU | RTX 2070 Max-Q，sm_75，8192 MiB | 空闲显存随桌面浮动，**不要拿历史数字当配额** |
| 磁盘 | `/mnt/d` 剩余约 570 GB | Day 16 的真实数据约 1.5 GB |

⚠️ **只认 `embodied_ai`**。`pydata`(3.11)、`base`(3.13)、`py314`(3.14) 都没装 torch，
激活错会直接 `ModuleNotFoundError`。这条在 Day 1 记过 K 类错题，见 `error_log.md`。

### 不升级 torch 的理由（2026-11-20 定）

PyPI 当时最新为 2.14.1，本机 2.10.0，落后 4 个 minor，但**决定不升**：

1. 到 Day 28 为止不需要任何新 API——`Dataset`/`DataLoader`/Tensor 基础操作自 1.x 稳定。
2. sm_75 是 2018 年的 Turing，新版 torch 在收缩老架构的预编译支持。升级要同时动
   torch / torchvision / torchaudio 三个硬绑定的包，错一个 minor 就废。
3. 收益 0，成本是 Day 25-27 的显存与 FPS 数据全部重做。

**唯一该升级的时机是换 50 系卡（sm_120）**，那时一次性重建环境，而不是原地升。
`pyproject.toml` 已给 torch 加上界 `<2.11`，并写明「先改上界数字、再跑 `env_check.py`」的两步流程。

---

## 2. WSLg 与「headless」

**本机其实有 GUI。** WSLg 在运行：`DISPLAY=:0`、`WAYLAND_DISPLAY=wayland-0`、tkinter 8.6 可用。

后果与两个库的差别：

| 库 | 失败方式 | 判定 |
|:---|:---|:---|
| `cv2`（headless wheel） | 真调用立刻抛 `The function is not implemented` | **好**，立刻发现 |
| `matplotlib`（选 `tkagg`） | `plt.show()` **弹窗阻塞**，等你关窗口 | **坏**，会被误认为「跑得慢」 |

所以 headless 是**纪律而非事实**：每个绘图脚本开头 `matplotlib.use("Agg")`，再 `savefig`。
`env_check.py` 会把非 `Agg` 后端报成 WARN（不判 FAIL：管道本身不受影响）。

`cv2` 的 `hasattr` 陷阱：headless 轮子**照样导出** `imshow`/`waitKey`/`createTrackbar`/
`setMouseCallback` 四个符号，`hasattr` 全返回 `True`。**符号在 ≠ 能用**，只能试调用才知道。

---

## 3. 网络与代理（会复发）

**症状**：

```
fatal: unable to access 'https://github.com/...': Failed to connect to github.com:443
       over proxy 127.0.0.1 after 2035 ms: Could not connect to server
```

**根因不在 GitHub，在本地两份代理配置打架**：

| 来源 | 值 | 状态 |
|:---|:---|:---|
| `git config http.proxy` | `http://127.0.0.1:7897`（Clash 默认端口） | **常不启动** |
| 环境变量 `https_proxy` | `http://127.0.0.1:11114`（举例） | **活着，但端口每次会话都可能变** |

git 优先读自己的 config，于是走了死的那个。**这是配置漂移，不是网络故障。**

⚠️ **不要硬编码代理端口**。环境变量的值是宿主每次注入的，同一天里 11640、11114 都出现过。
任何写死端口的命令都会在下次失效。**永远引用 `$https_proxy`。**

**可用的推送命令**：

```bash
GIT_TERMINAL_PROMPT=0 git -c http.proxy="$https_proxy" \
  -c https.proxy="$https_proxy" push origin main
```

- 走代理实测约 18 s；直连（`curl --noproxy '*'`）有时更快，有时超时（`exit 124`），不可依赖。
- 判断当前代理是否可用，先跑一句：
  `curl -s -o /dev/null -w "%{http_code}\n" -x "$https_proxy" https://github.com`，期望 `200`。
- 无 `credential.helper` 配置，凭据由 Git Credential Manager
  （`C:\Program Files\Git\mingw64\bin\`）存在 Windows 凭据管理器里。
- `GIT_TERMINAL_PROMPT=0` 是为了让认证失败时**立刻报错**而不是挂住等输入。
- 长期解法：`git config --global --unset http.proxy`（让它跟随环境变量），
  而不是把 7897 改成另一个会变的端口。

---

## 4. 验证习惯（改 `env_check.py` 必须做）

### 什么时候才该跑 `env_check.py`

**不是每天。** 它校验的是环境契约（CUDA major、kernel 可用性、依赖齐备），
这些条件不会因为过了一天就变。每天跑是仪式，不是检查。

| 该跑 | 不必跑 |
|:---|:---|
| `pip install` / `conda install` 之后 | 每天开工 |
| 驱动或 CUDA 更新之后 | 连续工作的中途 |
| 换机器 / 重新 clone / `pip install -e .` 之后 | 写普通脚本之前 |
| 中断很久后首次开工（如 68 天后） | |
| **出现解释不了的报错时**（排障第一步） | |
| Day 25-27 压测前（那几天显存是核心变量） | |

判断依据：它防的是**静默失败**（`is_available()` 为 `True` 而 kernel 跑不了），
不是「环境坏了」。环境真坏了，`import torch` 或脚本第一行就会当场报错，不需要专门的检查去发现。
Day 2 那种 CPU 上的 dtype 构造，碰不到 kernel 那条路径。

### 改完之后的双向验证

**校验器必须被证明会拒绝东西**，否则「通过」毫无意义——这是 Day 1 那条「自洽 ≠ 正确」错题的直接延伸。

改完 `env_check.py` 要做两步：

1. **正向**：`python src/env_check.py; echo exit=$?` → 期望 `exit=0`，新检查项为绿。
2. **反向**：人为构造失败（假 torch 对象、强制探针返回 FAIL），确认报告里出现 `[fail]`
   **且退出码为 1**。只看到红字、退出码仍 0，说明该探针的 FAIL 没进失败清单。

2026-11-20 的复核就是靠第 2 步抓到两个真 bug：`is_available()` 在 `try` 外会整脚本崩；
headless 探针的 FAIL 只 print 未计入 `failures`。**两个都在「正向」里全绿。**

---

## 5. 脚手架已知缺陷（4 条 P0，其中 2 条已修；P1 4 条，其中 2 条已修）

2026-11-19 静态分析定位；2026-11-25（Day 3）与 2026-11-26（Day 4）实跑时各补充 1 条 P0，
Day 4 另补 1 条 P1（均已修）。
P0 会直接给出**错误的契约或错误的验收标准**，须优先处理；P1/P2 不阻塞学习，
但会在对应那天变成理解障碍。

| 级别 | 位置 | 问题 |
|:---|:---|:---|
| **P0** | `AGENTS.md` 禁止事项 7 vs `src/embodied_dataset.py` v0.1 TODO | 前者禁止「在 `__init__` 实例化 `h5py.File`」，后者要求「`__init__` 中开→立即关」。技术正解是禁止**跨 fork 持有**句柄，不是禁止打开。需同步改 `knowledge_graph.md` 的 Day 21 复述题 |
| **P0** | `dataset_spec.md` / `00_LEARNING_PLAN.md` / `src` docstring / `day_20` | 多模态轴序无单一真源：spec 为 `[N,84,84,3]`，plan 为 `[B,3,84,84]`，另两处写 "or"。`torch.stack` 不看轴序，两种都能过，**契约实际失效** |
| **P0** | `day_03_shape_ops.py` 验收 3（**已修** 2026-11-25） | 原文断言 `permuted.view(3, -1)` 会抛 `RuntimeError`，**实测不抛**，返回 `(3, 7056)` / stride `(1, 3)`。`view` 只合并 stride 兼容的相邻维，**不要求严格连续**；真正会抛的是 `view(-1)` 全展平、`view(84*84, 3)`、`view(84, -1)` |
| **P0** | `day_04_indexing.py` 的 `main()`（**已修** 2026-11-26） | 传 `threshold=0.0`，而验收 2 与函数默认值都写 `> 0.5`。前者筛出 **447** 帧、后者 **187** 帧——**两条路都能跑通，而验收标准只认一条** |
| P1 | `day_04_indexing.py` 验收 3（**已修** 2026-11-26） | 称 `[7]` 会破坏 "downstream matrix multiplication"——**实测错**：`[7] @ [7,7]` 合法，返回 `(7,)`。`[7]` 真正会炸的是拼接：`torch.cat([v7, v17], dim=0)` 抛 `Tensors must have same number of dimensions: got 1 and 2` |
| P1 | `day_11_dataset_contract.py` | parity 断言在 `shuffle=True` 下语义不成立（batch 第 0 个不是 `dataset[0]`）；形状断言对任何样本都成立，是空洞检查。应改 `shuffle=False` 且比值 |
| P1 | `day_11` / `day_20` TODO | 直接给出答案（`{'dropped_total': 8}`、完整 `torch.stack` 写法），违反禁止事项 1「只给提示」 |
| P1 | 所有 `day_NN` 的 `main()` | 末尾硬编码 `>>> Day XX Acceptance: ALL PASS`；`day_15` 数据缺失时只 print 警告后 `return`（退出码 0）。后 27 天 assert 密度远低于 Day 1，「失败的运行必须响」名存实亡 |
| P2 | `docs/domain_map.md` | 缺 B 类编号表（B01-B05 只定义在 `knowledge_graph.md`）；M3 说 Day 13 交付 v0.1 但逐日表是 Day 12，M5 说 Day 24 交付 v0.5 但版本序列只到 v0.4 |

**新增那条 P0 的来历**（2026-11-25，Day 3 实跑时发现）：不是静态分析查出来的，是代码跑不通才定位的。
按 docstring 写的 `else: raise` 在运行时抢先触发——**断言拒绝了教材的假设**。随后实测确定边界：

| 调用（输入 stride `(1, 252, 3)`，非连续） | 结果 |
|:---|:---|
| `view(3, -1)` / `view(3, 84*84)` | **成功**，`(3, 7056)`，stride `(1, 3)` |
| `view(-1)` / `view(84*84, 3)` / `view(84, -1)` | `RuntimeError: view size is not compatible...` |

根因是 `view` 的规则被简化成「要求连续」传播了出去，而真实规则是「从最内维往外合并
stride 兼容的相邻维，再拿这个块结构比对目标形状」。`permuted` 的后两维满足
`252 == 3 × 84`，可以合并成 7056，而 C 轴 stride 1 独立在外——`view(3, -1)` 要的正是这个结构，
所以放行。**「连续」是「能否 view」的充分条件，不是必要条件。**

> 教训：**验收标准本身就是代码，没跑过就不算数**。这与 P1 里那条「后 27 天 assert 密度
> 远低于 Day 1」是同一个病的两种长相——一个是断言太少，一个是断言写错了但仍被当作标准。

**Day 4 补充的两条**（2026-11-26，同一类病的新长相）：

1. **同一个参数在两处写了两个值，两条路都能跑通。** `main()` 用 `0.0`、验收标准写 `0.5`，
   脚本不报错、输出还写着 `matched 447 / 900`，与验收要求的 187 差了一倍多。
   这类错误比「断言写错」更隐蔽：**代码没错，是「跑的是哪一条」没错**。
   处置：`main()` 对齐到 `0.5`，并在文档里写下两个数（447 / 187），下次一眼能看出跑的是哪条。
2. **「某操作会失败」这句话必须实测那个操作。** 原文说 `[7]` 破坏矩阵乘，实测
   `[7] @ [7,7] → (7,)`，是合法的；真正拒绝 1 维的是 `torch.cat`。
   推理出来的「会被拒绝」不可靠——**矩阵乘会自动补维，拼接不会**。

> 两天连起来看，§5 里 4 条 P0 有 2 条（`day_03` / `day_04`）**都是跑起来才发现的**，
> 静态分析一条也没查出来。其中两处措辞是「听起来合理但没人跑过」的类型——
> `view(3,-1)` 会报错、`[7]` 破坏矩阵乘。
> 所以这里立的规矩是：**凡文档里出现「会报错 / 会失败 / 返回视图」这类断言，
> 都要附一条能复现它的命令或实测值**；没有实测值的，标「未验证」，不当作标准。

---

## 6. 数据文件

`data/` 全部在 `.gitignore` 内（体积大、可重建、不入库）。**因此本文是它们唯一的出处记录**——
文件一旦丢失，除重新下载外没有恢复依据。2026-11-20 逐项实测。

### 6.1 `data/raw/teleop_demo.hdf5` —— Mock（116 MB）

由 Day 1 生成，`attrs` 自带来源标记，可自证身份：

```
synthetic = True
generator = experiments/day_01_env_and_data.py
seed      = 20260911
note      = stand-in for robomimic lift/ph image.hdf5
```

结构 100% 对齐 `dataset_spec.md`：`data/demo_N/obs/agentview_image` 为 `uint8 [N,84,84,3]`
（`chunks=(1,84,84,3)`），`robot0_joint_pos` 与 `actions` 为 `float64 [N,7]`，三键 N 全部对齐。

```
demo_0  900   demo_1 1000   demo_2 1100
demo_3 1200   demo_4  800   demo_5  700      total = 5700 帧
```

**颜色是刻意选的判据**：红块 `(220,40,40)` 占 2.0%，蓝底 `(30,87,200)` 占 1.5%。
通道序一反转，红蓝立刻互换，**肉眼即可判定**，比打印数字可靠——这是 Day 13 BGR/RGB 对照的判据来源。

**三条必须知道的局限**：

1. **六个 demo 的视觉内容几乎是同一段。** 方块都从 `(51.5, 7.5)` 匀速走到 `(51.5, 75.5)`，
   帧数不同而已；图像间最大差异 16/255，来自背景渐变噪声。**不是六段不同演示。**
2. **数值模态倒是独立的。** `actions` / `joint_pos` 各自随机生成，`demo_5` 不是 `demo_0` 的前缀切片。
3. **图像与动作无因果关系。** 方块按帧号自己横移，`actions` 是噪声。想验证「动作是否驱动画面」，
   Mock 给不出答案——Day 22-26 不需要这个，真需要时只能换真数据。

适用边界：Day 2-14 完全够用；Day 15 懒加载实验成立（`chunks` 已按单帧切好）；
**Day 16 必须换真数据**（缺 `mask/` 组与 `env_args` 属性）。

Day 4 用的是 `demo_0/obs/robot0_joint_pos`（**900×7**，文件内 `float64`，脚本转 `float32`）。上面三条局限对它
**都不成立**：`joint_pos` 是独立随机生成的数值模态，不受「六段演示画面几乎相同」影响，
也不需要「动作驱动画面」的因果。所以切片/掩膜的练习结论可以照用到真数据上——
变的是数值，不是索引语义。

### 6.2 `data/raw/test/` —— 真 robomimic Lift（3 个版本，247 MB）

三个文件是**同一份数据的三个格式版本**，`env_name: "Lift"`，10 个 demo，帧数分布一致：

```
[59, 58, 57, 55, 51, 58, 49, 49, 44, 51] = 531 帧
mask/train = demo_1,2,3,5,6,7,9     mask/valid = demo_0,4,8
```

| 文件 | `env_version` | obs 键 | depth | `total` 属性 | 实际帧数 | 大小 |
|:---|:---|:---|:---|:---|:---|:---|
| `test.hdf5` | **缺失**（旧格式） | 13 | ❌ | **9666** ← 假 | 531 | 44 MB |
| `test_v141.hdf5` | `1.4.1` | 15 | ✅ | 531 | 531 | 102 MB |
| `test_v15.hdf5` | `1.5.1` | 15 | ✅ | 531 | 531 | 102 MB |

**`v141` 与 `v15` 的数组内容完全一致**——抽查 `demo_0` / `demo_9` 的 `actions`、`agentview_image`、
`object` 六个组合，`max|diff| = 0`，形状也相同。**但两个文件不是字节级相同**（101.2 vs 101.6 MB）：
差异全部在 `env_args` 的版本标记里，数据本身是同一份。**约 102 MB 纯冗余。**

**`test.hdf5` 的 `total = 9666` 是错的**，实际只有 531 行 `actions`，差 18 倍（裁剪时留下的陈旧属性）。
这条比数据本身更值得记：**它是 Day 1「自洽 ≠ 正确」错题的活标本**——`attrs["total"]` 看起来权威得像
spec，实际与 `len(actions)` 对不上。`Dataset.__len__` 若照它写，会凭空多出 9135 个不存在的样本。
**写 `__len__` 一律数实际行数，不读元数据。**

**`test.hdf5` 与另两个的图像不同**：`actions`/`states`/`joint_pos` 全等，但 `agentview_image`
`max|diff| = 215`。同一轨迹、渲染结果不同——旧格式到 `1.4.1` 之间渲染管线变了（旧文件的
`env_version` 缺失，无法确定具体版本，所以这是推断而非结论）。不是数据损坏，
但**别拿它做像素级对照的基准**。

真实 Lift 的画面是**白色桌面 + 木色方块**（主色 `[252 251 247]`），与 Mock 的饱和蓝完全是两个世界。
这解释了 Day 13 那条边界：Mock 是纯色块、判据强；真图是渐变加光照、判据弱。两者**不能互相推断**
通道序，只能各自验证各自的转换逻辑。

Day 15 探针会看到 13-15 个 obs 键（`agentview_depth`、`robot0_eef_pos/quat/vel`、`gripper_qpos/qvel`、
`joint_pos_cos/sin` …），而 spec 只取用 3 个。**看到多余键不算文件损坏**，按 spec 验收即可。

**处置**：`test_v141.hdf5` 与 `test_v15.hdf5` 重复，可删其一省 102 MB。
`test.hdf5` 建议保留——它带着两个别处拿不到的对照价值：`total=9666` 这个活体反例，
以及「同轨迹不同渲染」的跨版本差异。删任何文件前先确认 Day 16 用的是哪一份。
