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

## 5. 脚手架已知缺陷（待修，非阻塞学习）

2026-11-19 静态分析定位。**均不影响 Day 2 开始**，但会在对应那天变成理解障碍。

| 级别 | 位置 | 问题 |
|:---|:---|:---|
| **P0** | `AGENTS.md` 禁止事项 7 vs `src/embodied_dataset.py` v0.1 TODO | 前者禁止「在 `__init__` 实例化 `h5py.File`」，后者要求「`__init__` 中开→立即关」。技术正解是禁止**跨 fork 持有**句柄，不是禁止打开。需同步改 `knowledge_graph.md` 的 Day 21 复述题 |
| **P0** | `dataset_spec.md` / `00_LEARNING_PLAN.md` / `src` docstring / `day_20` | 多模态轴序无单一真源：spec 为 `[N,84,84,3]`，plan 为 `[B,3,84,84]`，另两处写 "or"。`torch.stack` 不看轴序，两种都能过，**契约实际失效** |
| P1 | `day_11_dataset_contract.py` | parity 断言在 `shuffle=True` 下语义不成立（batch 第 0 个不是 `dataset[0]`）；形状断言对任何样本都成立，是空洞检查。应改 `shuffle=False` 且比值 |
| P1 | `day_11` / `day_20` TODO | 直接给出答案（`{'dropped_total': 8}`、完整 `torch.stack` 写法），违反禁止事项 1「只给提示」 |
| P1 | 所有 `day_NN` 的 `main()` | 末尾硬编码 `>>> Day XX Acceptance: ALL PASS`；`day_15` 数据缺失时只 print 警告后 `return`（退出码 0）。后 27 天 assert 密度远低于 Day 1，「失败的运行必须响」名存实亡 |
| P2 | `docs/domain_map.md` | 缺 B 类编号表（B01-B05 只定义在 `knowledge_graph.md`）；M3 说 Day 13 交付 v0.1 但逐日表是 Day 12，M5 说 Day 24 交付 v0.5 但版本序列只到 v0.4 |

**另**：`data/raw/test/` 残留三个 HDF5 共约 248 MB，无归属说明，可清理（不在版控内）。
