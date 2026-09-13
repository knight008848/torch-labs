"""
Day 03 / 2026-09-13 / Shape operations: view, reshape, permute, and contiguity
Runtime: <fill in after completion>

Goal:
    Master tensor shape manipulation (view vs reshape vs permute), understand
    underlying memory stride layout and the contiguity constraint, and visualize
    the destructive effect of flattening/reshaping across wrong axes.

Acceptance:
    1. Transform synthetic [84, 84, 3] image to [3, 84, 84] (HWC -> CHW) using permute.
    2. Demonstrate that permute changes stride and leaves tensor non-contiguous (.is_contiguous() == False).
    3. Prove that .view() on a non-contiguous tensor raises RuntimeError, while .contiguous().view()
       or .reshape() succeeds.
    4. Save a comparison figure to docs/figs/day_03_axis_disaster.png contrasting correct permute
       vs disastrous reshape(3, 84, 84).

Related concepts:
    T04 (shape ops: view/reshape/permute/contiguous)

Hub position:
    Hub (1) Tensor. Shape operations are the spatial bridge between storage layouts (NumPy/OpenCV HWC)
    and PyTorch Conv2d expectations (NCHW).

Constraints:
    - Headless: no cv2.imshow / waitKey / createTrackbar / setMouseCallback
    - Save visualization to docs/figs/ using matplotlib savefig
    - Max 3 new concepts
    - Pure functions called from main()
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "teleop_demo.hdf5"
FIG_DIR = ROOT / "docs" / "figs"


def generate_synthetic_image(height: int = 84, width: int = 84) -> torch.Tensor:
    """Generate a simple synthetic RGB pattern [H, W, 3] uint8 with distinct color bands."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    # Red top third, green middle third, blue bottom third
    img[: height // 3, :, 0] = 255
    img[height // 3 : 2 * height // 3, :, 1] = 255
    img[2 * height // 3 :, :, 2] = 255
    return torch.from_numpy(img)


def permute_hwc_to_chw(img_hwc: torch.Tensor) -> torch.Tensor:
    """1. Permute image tensor from HWC to CHW.

    TODO(Student):
        Permute axes: dimension 0 (H), 1 (W), 2 (C) -> 2 (C), 0 (H), 1 (W).
        Hint: img_hwc.permute(2, 0, 1)
    """
    # --- [TODO: Student Implementation Start] ---
    raise NotImplementedError("Step 1: Permute tensor from [H, W, C] to [C, H, W]")
    # --- [TODO: Student Implementation End] ---

    assert img_chw.shape == (3, img_hwc.shape[0], img_hwc.shape[1]), f"Expected CHW, got {img_chw.shape}"
    return img_chw


def investigate_contiguity_and_view(img_hwc: torch.Tensor) -> None:
    """2 & 3. Prove contiguity changes and demonstrate view vs reshape.

    TODO(Student):
        1. Assert img_hwc.is_contiguous() is True.
        2. Create permuted = img_hwc.permute(2, 0, 1).
        3. Assert permuted.is_contiguous() is False.
        4. In a try/except RuntimeError block, attempt permuted.view(3, -1) and catch the error.
        5. Verify that permuted.contiguous().view(3, -1) and permuted.reshape(3, -1) succeed.
    """
    # --- [TODO: Student Implementation Start] ---
    raise NotImplementedError("Step 2 & 3: Investigate contiguity and catch view RuntimeError on non-contiguous tensor")
    # --- [TODO: Student Implementation End] ---


def save_axis_order_counterexample(img_hwc: torch.Tensor, save_path: Path) -> None:
    """4. Contrast correct permute vs catastrophic reshape, saving to docs/figs.

    TODO(Student):
        1. Create correct_chw using permute(2, 0, 1).
        2. Create disastrous_chw using img_hwc.reshape(3, 84, 84).
        3. Convert both back to HWC for matplotlib rendering:
           - correct_back: correct_chw.permute(1, 2, 0).numpy()
           - disastrous_back: disastrous_chw.permute(1, 2, 0).numpy()
        4. Plot side by side with plt.subplots(1, 2) and save to save_path.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # --- [TODO: Student Implementation Start] ---
    raise NotImplementedError("Step 4: Generate side-by-side plot of correct permute vs disastrous reshape")
    # --- [TODO: Student Implementation End] ---

    assert save_path.exists(), f"Figure file not found at {save_path}"


def main() -> None:
    """Entry point for Day 03."""
    print("=== Day 03: Shape Operations & Contiguity ===")

    # Setup
    img_hwc = generate_synthetic_image(84, 84)
    print(f"  [Setup] Created synthetic pattern: shape={img_hwc.shape}, contiguous={img_hwc.is_contiguous()}")

    # Step 1: Permute HWC -> CHW
    img_chw = permute_hwc_to_chw(img_hwc)
    print(f"  [Step 1] Permuted to CHW: shape={img_chw.shape}, stride={img_chw.stride()}")

    # Step 2 & 3: Contiguity and view vs reshape
    investigate_contiguity_and_view(img_hwc)
    print("  [Step 2 & 3] Contiguity breakdown and view RuntimeError boundary caught")

    # Step 4: Plot visual proof
    fig_dest = FIG_DIR / "day_03_axis_disaster.png"
    save_axis_order_counterexample(img_hwc, fig_dest)
    print(f"  [Step 4] Visual counterexample saved to {fig_dest}")

    print("\n>>> Day 03 Acceptance: ALL PASS")


if __name__ == "__main__":
    main()
