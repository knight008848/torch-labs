# 每日交付文件模板

> 复制以下内容到 `experiments/day_NN_主题.py` 头部。
> 规范来源：`AGENTS.md`（注释用英文、函数式风格、`pathlib`）。

## 模板

```python
"""
Day NN / 2026-MM-DD / <一行学习目标>
Runtime: <实际耗时，收工后回填>

Goal:
    <当天要达成的可验证目标，一句话>

Acceptance:
    <验收标准，逐条可勾选>

Related concepts:
    <domain_map.md 中的编号，如 T04 P02>

Hub position:
    <本日内容在 knowledge_graph.md 里的位置：引入/挂在哪个枢纽下>

Constraints:
    - Headless: no cv2.imshow / waitKey / createTrackbar / setMouseCallback
    - Use matplotlib savefig for any visualization
    - Max 3 new concepts
"""

from pathlib import Path

import numpy as np
import torch


# Project root resolved relative to this file, not the CWD
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
FIG_DIR = ROOT / "docs" / "figs"


def main() -> None:
    """Entry point. Keep each step a small pure function."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # 1) <step one>
    # 2) <step two>
    # 3) acceptance assertions at the end, so a failed run is loud
    raise NotImplementedError("fill in today's exercise")


if __name__ == "__main__":
    main()
```

## 约定

| 项 | 约定 |
|:---|:---|
| 文件命名 | `experiments/day_NN_主题.py`，NN 补零（`day_01`、`day_28`） |
| 注释语言 | 英文 |
| 风格 | 函数式优先，`main()` 只做编排，每步是纯函数 |
| 路径 | 一律 `pathlib.Path`，用 `Path(__file__)` 定位而非 CWD |
| 可视化 | matplotlib `savefig` 到 `docs/figs/day_NN_*.png` |
| 断言 | 验收标准写成 `assert`，失败的运行必须"响" |
| 收工 | 回填 `Runtime`，更新 `progress.md` |

## 边界情况示范（必写）

每个脚本至少覆盖一类：

- 索引越界 / 空数据
- 除零（如 `max == min` 的归一化）
- 形状不匹配（如 `[1,7]` vs `[7]`）
- 文件缺失或读取失败
- 数值类型溢出（`uint8` 相减回绕）

## 禁止

- ❌ `cv2.imshow` / `cv2.waitKey` / `cv2.createTrackbar` / `cv2.setMouseCallback`
- ❌ 一次引入超过 3 个新概念
- ❌ 省略 `try/except` 或边界分支
- ❌ 把答案直接写进注释
