"""
Day 13 / 2026-09-23 / Vision Tensor Bridge: Channel Order, HWC->CHW, and Normalization
Runtime: <fill in after completion>

Goal:
    Establish image preprocessing pipeline converting raw visual observations into PyTorch
    model inputs: verify color channel order (BGR vs RGB), permute HWC to CHW layout,
    and normalize pixel dynamic range from uint8 [0, 255] to float32 [0.0, 1.0].

Acceptance:
    1. Construct a synthetic calibration image with known color blocks (e.g. pure red and pure blue).
    2. Compare BGR2RGB vs identity paths, rendering side-by-side comparison via matplotlib savefig
       to docs/figs/day_13_color_check.png (HEADLESS: strictly no cv2.imshow).
    3. Assert model-ready tensor invariants: shape == (3, 84, 84), dtype == torch.float32,
       0.0 <= min <= max <= 1.0.
    4. Implement parameterized to_model_input(img, assume_bgr: bool) without hardcoding assumptions
       prior to Day 16 empirical validation on real robot cameras.
    5. Handle boundary condition: gracefully detect and reject invalid image shapes (e.g. 2D grayscale
       or 4-channel RGBA) with informative ValueError.

Related concepts:
    P01 (BGR->RGB conversion), P02 (HWC->CHW permutation), P03 (uint8 -> float32 [0, 1] normalization),
    X01 (OpenCV headless color inspection)

Hub position:
    Hub (4) Dataset/DataLoader (Bridging raw vision byte streams to neural network inputs).

Constraints:
    - Headless: no cv2.imshow / waitKey / createTrackbar / setMouseCallback
    - Visual outputs saved to docs/figs/
    - Pure functions called from main()
    - Max 3 new concepts
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "docs" / "figs"


def generate_color_calibration_image(height: int = 84, width: int = 84) -> np.ndarray:
    """1. Generate synthetic calibration test image with known color quadrants.

    TODO(Student):
        1. Create image of shape [height, width, 3] with dtype np.uint8.
        2. Top half: Pure Red (RGB: [255, 0, 0]).
        3. Bottom half: Pure Blue (RGB: [0, 0, 255]).
        4. Return the uint8 NumPy array.
    """
    # --- [TODO: Student Implementation Start] ---
    raise NotImplementedError("Step 1: Construct calibration image with known red and blue regions")
    # --- [TODO: Student Implementation End] ---

    return img


def compare_channel_order_paths(
    img: np.ndarray,
    save_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """2. Compare cv2.cvtColor(BGR2RGB) vs identity, saving headless comparison.

    TODO(Student):
        1. path_bgr2rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).
        2. path_identity = img.copy().
        3. Create 1x2 subplot with matplotlib, displaying path_identity and path_bgr2rgb.
        4. Set titles: 'Raw as RGB (Identity)' vs 'Interpreted as BGR (Converted to RGB)'.
        5. Ensure save_path parent exists, call plt.savefig(save_path, bbox_inches='tight').
        6. plt.close().
        7. Return (path_identity, path_bgr2rgb).
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # --- [TODO: Student Implementation Start] ---
    raise NotImplementedError("Step 2: Compare color paths and save headless side-by-side plot")
    # --- [TODO: Student Implementation End] ---

    return path_identity, path_bgr2rgb


def to_model_input(img: np.ndarray, assume_bgr: bool = False) -> torch.Tensor:
    """3 & 4. Bridge raw vision array to normalized PyTorch tensor [3, H, W].

    Args:
        img: [H, W, 3] uint8 NumPy array.
        assume_bgr: If True, converts BGR to RGB via cv2.cvtColor. If False, keeps channels as-is.

    Returns:
        torch.Tensor of shape [3, H, W], dtype torch.float32, normalized to [0.0, 1.0].

    TODO(Student):
        1. Check image dimensions: must be 3D with 3 channels. If not, raise ValueError.
        2. If assume_bgr:
           converted = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
           converted = img
        3. Convert to torch tensor: t = torch.from_numpy(converted)
        4. Permute HWC -> CHW: t = t.permute(2, 0, 1).contiguous()
        5. Normalize to float32 [0, 1]: t = t.to(torch.float32) / 255.0
        6. Return t.
    """
    # --- [TODO: Student Implementation Start] ---
    raise NotImplementedError("Step 3 & 4: Implement to_model_input with permute and normalization")
    # --- [TODO: Student Implementation End] ---

    assert out_tensor.shape == (3, img.shape[0], img.shape[1])
    assert out_tensor.dtype == torch.float32
    assert 0.0 <= out_tensor.min().item() <= out_tensor.max().item() <= 1.0
    return out_tensor


def demonstrate_invalid_shape_boundary() -> bool:
    """5. Demonstrate boundary condition: invalid image dimensions raise ValueError."""
    # --- [TODO: Student Implementation Start] ---
    # Pass invalid shape (e.g. 2D grayscale [84, 84] or 4D array) to to_model_input.
    # Catch ValueError and return True if caught.
    raise NotImplementedError("Step 5: Catch ValueError on non-3-channel input")
    # --- [TODO: Student Implementation End] ---


def main() -> None:
    """Entry point for Day 13."""
    print("=== Day 13: Vision Tensor Bridge ===")

    # Step 1: Calibration image
    calib_img = generate_color_calibration_image(84, 84)
    print(f"  [Step 1] Created calibration image: shape={calib_img.shape}, dtype={calib_img.dtype}")

    # Step 2: Headless side-by-side comparison
    fig_path = FIG_DIR / "day_13_color_check.png"
    ident, converted = compare_channel_order_paths(calib_img, fig_path)
    print(f"  [Step 2] Color comparison plot saved headlessly to {fig_path}")

    # Step 3 & 4: Tensor bridge transformation
    tensor_rgb = to_model_input(calib_img, assume_bgr=False)
    tensor_bgr = to_model_input(calib_img, assume_bgr=True)
    print(f"  [Step 3 & 4] Tensor output verified: shape={tensor_rgb.shape}, dtype={tensor_rgb.dtype}, range=[{tensor_rgb.min():.2f}, {tensor_rgb.max():.2f}]")

    # Step 5: Boundary demonstration
    invalid_caught = demonstrate_invalid_shape_boundary()
    print(f"  [Step 5] Invalid shape boundary caught: {invalid_caught}")

    print("\n>>> Day 13 Acceptance: ALL PASS")


if __name__ == "__main__":
    main()
