# 进度追踪

> **规则 4**：每次学习后必须更新本文件。
> 每天收工前勾选完成项、记录薄弱点、写下下一步。
> 计划版本：30 天制（Day 29-30 为冲刺日，Day 29 兼作缓冲）

---

## 当前状态

| 项 | 值 |
|:---|:---|
| 当前 Day | 4（2026-11-26） |
| 当前周 | W1 地基 I：Tensor 与 autograd |
| 当前里程碑 | M1 已达成，M2 待办（Day 7 周测） |
| 累计投入 | 3.5 h |
| 本周投入 | 3.5 h / 20 h |

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

**数据现状（不阻塞 Day 2，Day 16 前须换真数据）**：`data/raw/teleop_demo.hdf5` 是 Mock
（5700 帧、6 demo、结构 100% 对齐 spec）。`data/raw/test/` 的三个 247 MB 文件经实测**不是垃圾**，
是真 robomimic Lift 数据的三个格式版本——`v141` 与 `v15` 数组内容逐元素相等（可删其一省 102 MB），
`test.hdf5` 的 `total=9666` 与实际 531 帧不符（陈旧属性，留作反例）。出处、差异与处置
全部记入 `env_notes.md` 第 6 节，因为 `data/` 不入库，那份文档是唯一记录。

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
- [x] **Day 2** Tensor 三要素 dtype/shape/device ｜ 交付：`day_02_tensor_basics.py` ｜ 耗时：1 h
- [x] **Day 3** 形状手术 view/reshape/permute ｜ 交付：`day_03_shape_ops.py` ｜ 耗时：1 h
- [x] **Day 4** 索引、切片与布尔掩膜 ｜ 交付：`day_04_indexing.py` ｜ 耗时：≈1 h
- [ ] **Day 5** 广播与 in-place 陷阱 ｜ 交付：`day_05_broadcast.py` ｜ 耗时：__
- [ ] **Day 6** autograd 动态计算图 ｜ 交付：`day_06_autograd.py` ｜ 耗时：__
- [ ] **Day 7** 周测 1 + 复盘 ｜ 交付：`day_07_quiz_1.py` ｜ 耗时：__

**W1 复盘**

> Day 1-2 增量（Day 7 时再终稿）：

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

**Day 2 增量（2026-11-24）**

- 完成事项（Day 2）：`day_02_tensor_basics.py` 六个 Step 全通，`exit=0`，6.9 s。
  五 dtype 构造 / CPU↔CUDA 往返（`torch.equal` 校验值不变）/ uint8 与 float32 的字节比
  `1,354,752 : 5,419,008`（4.0x）/ `from_numpy` 零拷贝 + `clone` 独立性
  （双向改写验证，非只比 `data_ptr`）/ uint8 运算期回绕 `255+1=0`、`0-1=255` /
  从 Mock HDF5 取真切片 `[84,84,3] uint8` 与 `[7] float64`
- 做错的练习及分析（记入 `error_log.md`）：Day 2 三条——**A×2、K×1**
  ① 五个变量四个用浮点字面量，其中两个「碰巧对」（K：依赖默认 dtype）；
  ② `element_size()` 问 dtype 报 `AttributeError`，改对后又多乘一个 4，
  **而断言照样通过**（A：断言只校验比值，不校验量级）；
  ③ 零拷贝演示后先把值改回去再断言（A：函数产物就是改动后的状态）
- 新发现的薄弱点：
  1. **断言设计**——今天两次被「空洞断言」放过（比值对、量级错）。
     `assert` 只钉相对关系时，绝对量级无人把关。往后凡是能量出绝对值的，
     都要再配一条钉死量级的断言
  2. **K 类累计 3 次**，三次同属一个家族：「文档 / 默认值被当成了权威」。
     已按规则 7 排变体练习进 Day 3（先预测 `is_contiguous()` 与 `view` 是否报错，再执行）
  3. **`chunks` 差异未记录**：Mock 是 `chunks=(1,84,84,3)`，真 Lift 数据是 `chunks=None`
     （连续存储）。Day 15 的分块/懒加载实验只在 Mock 上成立，需提前知情
- 下一步计划（Day 3）：形状手术 `view`/`reshape`/`permute`（T04），
  素材继续用 `[N,84,84,3]`；顺带做 K 类变体练习

- 复述检验①（Tensor 与 NumPy 的关系）是否流畅：Day 7 执行

- 下一步计划（Day 2）：Tensor 三要素 dtype/shape/device（T01 T02 T03 B01），
  练习素材直接用今天落地的 `[N,84,84,3] uint8` 与 `[N,7] float64`

**Day 3 增量（2026-11-25）**

- 完成事项（Day 3）：`day_03_shape_ops.py` 四条验收全通，`exit=0`，**4.1 s**
  （其中约 3 s 是 `import torch` 的固定开销）。Step 1 permute HWC→CHW 得
  `(3, 84, 84)` / stride `(1, 252, 3)`；Step 2&3 真的撞上并捕获 `view(-1)` 的
  `RuntimeError`（消息以 `view size is not compatible` 开头），再用 `torch.equal`
  验证 `contiguous().view(-1)` 与 `reshape(-1)` 逐元素相同；Step 4 产出
  `docs/figs/day_03_axis_disaster.png`——左：红绿蓝三条**横**色带；右：84 条 RGB
  循环的**竖**细纹。两张图并置后差别一眼可辨。验收 3 的**两边**都执行了：
  `view(-1)` 抛错被捕获，`view(3, -1)` 通过并与 `reshape(3, -1)` 逐元素相同
  （stride `(1, 3)`）——此前只跑了抛错那一边
- 做错的练习及分析（记入 `error_log.md`）：Day 3 五条——**C×2、A×3**
  ① `t.stride()` 答成 shape 本身，`p.stride()` 答「不知道」（C：stride 停在名词层面）
  ② `b.T` 的 stride 答成 `(2, 1)`（C：通式「右侧 size 连乘」的前提未建立）
  ③ `b.shape()` 当方法调用（A：属性/方法形态不统一）
  ④ `e.message`（A：Python 2 遗留属性）
  ⑤ `except` 分支里手写死消息，日志说 A、代码做 B（A：证据不是从异常对象取的）
- 新发现的薄弱点：
  1. **stride 没有具象化**是全天的主线：stride 答错 → `view` 的判断只能靠猜。
     两错同源，不是两个独立问题。往后凡涉及形状，**先报 stride 再动手**
  2. **证据链**：两次「看起来对但没有证据」——手写日志、`e.message`。
     已立硬规矩：**异常处理块里的每句话来自 `str(e)`，断言里的每个结论来自数据**
  3. **教材本身有错**（见下）
- 教材与讲解的缺陷（均已修，照实记）：
  1. `day_03` 验收 3 声称 `view(3, -1)` 会抛 `RuntimeError`，**实测不抛**（返回
     `(3, 7056)`）。`view` 只合并 stride 兼容的相邻维，不要求严格连续。docstring 已改为
     `view(-1)`，并登记进 `env_notes.md` §5
  2. 讲解中「stride 最后一维恒为 1」漏了前提，被当场追问推翻；`error_log.md`
     对照表已补「连续 vs 可 view」一行
  3. 曾用 `b.T[2][1]` 作反查样本——末元素在 `(2,1)` 与 `(1,3)` 两种假设下都是
     offset 5，**零区分度**。已改用 `b.T[0][1]`
  4. **判错了一道题**：快测 Q2-4 答 `p.reshape(3, -1)` 返回视图，我判为错、记成 K 类，
     并写进 `error_log.md`。实测**答对了**——`reshape(3, -1)` 与 `p` 共享 `data_ptr`、
     `_base` 非空、双向改写生效，是视图；因为目标 shape 恰好能被现有 stride 凑出。
     会拷贝的是 `reshape(-1)`。根因是把「非连续 → 拷贝」当成无条件规则。已在
     `error_log.md` 立「判定更正」条目、统计回退（K 4→3），并同步本节与薄弱点表
- 外部资料摘录（《PyTorch Tensor 内存布局深度解析》，CSDN，2026-09-08）：
  可读范围仅到 §2.1 标题处（VIP 截断），后续 view/reshape 内部、`contiguous` 代价、
  `channels_last` 三节**未获取、未采信**。已读到并实测复核的三条：
  1. **Tensor = 一块内存 + 三个元字段**（`shape` / `stride` / `storage_offset`）。
     「元信息与数据分离，改标签比搬数据便宜」——比我们的放映厅类比更适合跨天复用，
     Day 13 通道序、Day 20 batch 拼接都是同一件事
  2. **`storage_offset` 是第三个元字段**（此前三天都没提）：实测 `x[1:]` 的 offset=4、
     `x[:, 1]` 的 offset=1，二者与 `x` 共享同一 storage；`x.t()` 的 offset 仍为 0 且
     `untyped_storage().data_ptr()` 相同 → 转置确实只换标签
  3. **偏移公式** `off = Σ(下标_k × stride_k) + storage_offset`：实测 `p[2][5][7]`
     = `2×1 + 5×252 + 7×3` = **1283** = `t.reshape(-1)[1283]`。以后验 stride 不必猜，
     算 offset 再取值即可；这正好补上今天用 `b.T[0][1]` 做反查的方法论
  另附两条量级事实（同样实测）：`permute`/`transpose` 是 **O(1)**（只动元信息），
  触发拷贝的 `contiguous`/`reshape` 是 O(n)；`p` 与 `t` 共用同一块 storage，
  `nbytes()` 恒为 `numel × element_size`（uint8 素材下 21,168 B = `21168 × 1`），
  **没有新开内存**
- 下一步计划（Day 4）：索引、切片与布尔掩膜（T05），素材改用 Mock 的 `[N, 7]` 动作序列。
  开场快测 Q2 正是 `actions[0:1, :]` 与 `actions[0, :]` 的维度差异——**先预测再执行**，
  验收断言要钉住 `ndim` 而不仅是元素值

**Day 4 增量（2026-11-26）**

- 完成事项（Day 4）：`day_04_indexing.py` 五条验收全通，`exit=0`。
  `main()` 自身 **0.02–0.03 s**；wall **2.5–13 s**（12 次连测多数落在 2.5–5.1 s，
  最慢一次 12.95 s），几乎全花在 `import torch` 上（单独测 2.4–7.8 s，随页面缓存波动）。
  所以文件头记的是「main / wall」两个数，而不是一个会漂的总时间——
  单记一个「3.1 s」在冷缓存下无法复现，那是**测法不可重复**，不是脚本变慢。
  `day_03`、`day_04` **两个文件**的 `ruff check` / `ruff format --check`
  全绿；全仓库仍有 344 处（162 `E501` + 122 `F821` 未写天的 TODO + 42 `F401` 等），
  不在本次范围。耗时约 1 h（未逐次计时，按 Day 2/3 同口径估）
  - Step 1 `traj[10:50:2]` → `(20, 7)`；`len(range(10,50,2)) = 20`，独立核对逐行相等
  - Step 2 掩膜 `joint[0] > 0.5` → **187 / 900** 帧（另用 numpy 独立计数核对同为 187）；
    源行号严格递增（`torch.all(src[1:] > src[:-1])` 为真），首命中在**第 15 行**而非第 0 行
    ——所以「筛出表头」这类错误会被数据本身证伪
  - Step 3 `[0:1, :]` → `(1, 7)` ndim 2；`[0, :]` → `(7,)` ndim 1
  - Step 4 全 False 掩膜 → `(0, 7)`、`numel=0`、dtype 仍 `float32`，且是**拷贝**
  - 全程用 `data/raw/teleop_demo.hdf5` 的 `data/demo_0/obs/robot0_joint_pos`（900×7）
  - 复核（reviewer）后补的三处：① 掩膜行数改用**纯 Python 独立重算**——原来的
    `expected_rows` 与 `mask` 是同一表达式，属同义反复，抓不到「掩膜建错了列/改错了算子」；
    新写法在故意用错列时给出 `64 ≠ 187` 并当场 `raise`（已双向验证）；
    ② `[7] @ [7,7]` 合法、`torch.cat` 拒绝这组陈述，从「docstring 里的说法」变成**真的跑一遍**
    （比对 `str(e)` 的关键词，不符就 `raise`）；③ loader 的 schema 检查由 `assert` 改 `raise`，
    因为 `python -O` 会把 `assert` 整条删掉，契约随之消失
  - 复核第二轮再补两处：④ `reduced @ torch.zeros(...)` 的 `zeros` 默认 float32，
    会让函数在 float64 输入上报 dtype 不匹配（改动前对 dtype 中立），已换
    `reduced.new_zeros(...)`；⑤ `except` 里的 `assert "<消息>" in str(e)` 在 `-O` 下同样被删，
    异常消息不符时会被静默放过——已一律改 `raise`，`day_03` 同一处也一并改了
- 一条贯穿今天的实测（承 Day 3 的 stride）：

  | 表达式 | storage 同 | offset | stride | 视图/拷贝 |
  |:---|:---|:---|:---|:---|
  | `traj[10:50:2]` | 是 | 70 | `(14, 1)` | **视图**，且**不连续** |
  | `traj[0]` / `traj[0:1,:]` / `traj[:,0]` | 是 | 0 | — | 视图 |
  | `traj[[0,5,9]]` / `traj[mask]` | 否 | 0 | — | 拷贝 |

  推论：`traj[10:50:2].view(-1)` **抛 RuntimeError**（`stride[0]=14 ≠ shape[1]=7`，
  切片留下了「隔行取」的空隙），`reshape(-1)` 成功但是**拷贝**。
  **切片是视图 ≠ 切片后仍连续**——两件事，别合并成一句。
- 做错的练习及分析（记入 `error_log.md`）：Day 4 三条——**C×3**（首次单日整体性缺口）
  ① `traj[10:50:2]` 判成**拷贝**（C：视图共享 storage 这一条没建立；判据用 `data_ptr`
  而非 storage，后者只能回答「起点同不同」）
  ② `[0:1, :]` 与 `[0, :]` 的 ndim 差异答「不会」（C：索引同时决定维度数）
  ③ `traj[mask]` 为何必然拷贝答「不会」（C：stride 只能表达等差，掩膜是任意子集）
- 新发现的薄弱点：
  1. **三错同源，都在「张量的元字段」上**：storage / offset / stride / ndim 四个词各自
     都没能当场说清。这不是写码能力问题——四个 TODO 全写对了——是**说不出操作在内存
     与维度上做了什么**。对策已排进 Day 5 变体练习（见 `error_log.md` C 类预警）
  2. **「通过」再次不等于「被验证」**：`>` 与 `>=` 在这份数据上**无法区分**
     （恰等于 0.5 的帧数为 0，两种写法都是 187）。代码语义是严格大于，但没有证据能证明
     它没有写成 `>=`——验收条件在这一个维度上是空的
  3. **把 Mock 当成了真数据**：我在开场称其为「真数据」，实际 `attrs["synthetic"]=True`、
     生成器是 `day_01`。素材本身没问题（结构对齐 spec），问题是我没看属性就下了断言
- 教材与讲解的缺陷（均已修，照实记）：
  1. `day_04` 的 `main()` 传 `threshold=0.0`，而验收 2 与函数默认值都写 `> 0.5`——
     前者筛出 447 帧、后者 187 帧，**两条路都能跑通而验收标准只认一条**。
     已把 `main()` 对齐到 0.5
  2. 验收 3 原文称 `[7]` 会破坏「downstream **matrix multiplication**」——**实测错**：
     `[7] @ [7,7]` 合法，返回 `(7,)`。真正会炸的是 `torch.cat` 混用 1 维与 2 维。
     docstring 已改正并登记进 `env_notes.md` §5
  3. 我给的判据「`data_ptr` 相同就是视图」漏了前提（offset≠0 时必然失效），
     被今天的实测撞出。判据已改为比 `untyped_storage()`，对照表与薄弱点表同步
  4. Step 3/4 的断言把关节数 `7` 写死（脚手架原有）。已改为从 `trajectory.shape[1]`
     推导；Step 4 的 `9999.0` 魔法数一并换成 `trajectory.max() + 1.0`。
     实测换 6 关节张量后不再误报，而旧写法会抛「形状不对」——把可移植性问题
     伪装成数据问题
- 下一步计划（Day 5）：**广播与 in-place 陷阱**（T06 / T07），`day_05_broadcast.py`。
  四个验收：`[N,7]` 用 `[7]` 的 mean/std 向量化归一化、故意触发并捕获 autograd 的
  in-place 报错、对比 `a = a + b` 与 `a += b` / `a.add_(b)`、以及 `[N,7]` 与 `[8]`
  不兼容时的报错边界。**开场先答 C 类变体三问**（`expand` vs `repeat`、
  `+=` vs `+`、广播出的行 stride 可能是 0），写完预测再跑

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
| T02 | dtype（默认推断 vs 显式声明） | 1（K） | 是 | 五个变量两个「碰巧对」；Day 3 变体练习已执行，未再犯 |
| T02 | `element_size` 的单位 | 1（A） | 否 | 多乘 4 且断言未拦住；配绝对量级断言 |
| B01 | `from_numpy` 零拷贝的验证顺序 | 1（A） | 否 | 断言前把值改回去；断言永远在最后 |
| T04 | **stride 是什么** | 2（C） | 是 | 答成 shape、答「不知道」；座位号类比重讲，Day 7 复验 |
| T04 | 通式的适用前提（只在连续时成立） | 1（C） | 是 | `b.T` 答成 `(2,1)`；两条规则已分开记 |
| T05 | **视图 vs 拷贝的判据** | 1（C） | 是 | 切片答成拷贝；判据须比 `untyped_storage()`，`data_ptr` 在 offset≠0 时必失效 |
| T05 | **整数索引削维 vs 切片保维** | 1（C） | 是 | `[0:1,:]` 与 `[0,:]` 答不出；顺带纠正「`[7]` 不能矩阵乘」这个错说法 |
| T05 | **stride 只能表达等差 → 掩膜必拷贝** | 1（C） | 是 | 答「不会」；统一判据：能 stride 表达→可是视图，否则必拷 |
| T04 | `reshape` 返回视图还是拷贝 | 0（判定更正） | 是 | 曾误判为错答，实测是视图；拷贝与否取决于目标 shape，用 **storage** 判 |
| T04 | 属性 vs 方法 | 1（A） | 是 | `b.shape()` 当方法调；判据表已给 |
| — | Python 3 异常 API | 1（A） | 是 | `e.message` 是 Py2 遗留；统一用 `str(e)` |
| T04 | **验证证据须取自对象** | 1（A） | 否 | 手写死消息冒充日志；硬规矩已立，待后续反复验证 |

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
