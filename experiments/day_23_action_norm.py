"""
Day 23 / 2026-10-03 / Action Space Min-Max Normalization to [-1.0, 1.0]
Runtime: <fill in after completion>

Goal:
    Compute full-dataset action extreme statistics across all robot trajectories, serialize
    parameters to data/processed/action_stats.json, verify linear mapping to [-1.0, 1.0],
    validate round-trip unnormalization parity, and handle zero-variance boundary dimensions.

Acceptance:
    1. Scan all demonstration episodes in data/raw/teleop_demo.hdf5 to compute global 7-dim
       action min and max bounds, persisting stats to data/processed/action_stats.json.
    2. Normalize action batches using MinMaxNormalizer: assert min >= -1.0 and max <= 1.0.
    3. Verify unnormalization round-trip: assert torch.allclose(normalizer.unnormalize(norm_a), a, atol=1e-5).
    4. Handle boundary condition: gracefully handle constant dimensions (max == min) without
       triggering ZeroDivisionError or generating NaNs.

Related concepts:
    P09 (Min-Max normalization, statistical serialization, boundary safeguards)

Hub position:
    Hub (4) Dataset/DataLoader (Continuous control output standardization for policy models).

Constraints:
    - Headless: no cv2.imshow / waitKey / createTrackbar / setMouseCallback
    - Stats saved to data/processed/action_stats.json
    - Pure functions called from main()
    - Max 3 new concepts
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.normalize import MinMaxNormalizer

HDF5_PATH = ROOT / "data" / "raw" / "teleop_demo.hdf5"
STATS_PATH = ROOT / "data" / "processed" / "action_stats.json"


def compute_global_action_stats(hdf5_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """1. Traverse all demonstration groups and compute global action min and max vectors.

    TODO(Student):
        1. Open HDF5 file in read mode.
        2. Iterate over all demo keys (e.g. data/demo_*):
           Read 'actions' dataset.
        3. Compute overall min_vals across all frames: global_min = min(mins).
        4. Compute overall max_vals across all frames: global_max = max(maxs).
        5. Return (global_min, global_max) as NumPy arrays of shape (7,).
    """
    # --- [TODO: Student Implementation Start] ---
    raise NotImplementedError("Step 1: Compute dataset-wide global action min and max")
    # --- [TODO: Student Implementation End] ---

    assert global_min.shape == (7,) and global_max.shape == (7,)
    return global_min, global_max


def verify_normalization_and_roundtrip(
    normalizer: MinMaxNormalizer,
    sample_actions: torch.Tensor,
) -> tuple[float, float]:
    """2 & 3. Normalize actions, verify [-1, 1] range, and verify unnormalization parity.

    TODO(Student):
        1. norm_actions = normalizer.normalize(sample_actions).
        2. Assert norm_actions.min().item() >= -1.0 - 1e-6.
        3. Assert norm_actions.max().item() <= 1.0 + 1e-6.
        4. recovered = normalizer.unnormalize(norm_actions).
        5. Assert torch.allclose(recovered, sample_actions, atol=1e-5).
        6. Return (norm_actions.min().item(), norm_actions.max().item()).
    """
    # --- [TODO: Student Implementation Start] ---
    raise NotImplementedError("Step 2 & 3: Assert normalized bounds and round-trip parity")
    # --- [TODO: Student Implementation End] ---

    return min_val, max_val


def demonstrate_zero_variance_boundary() -> bool:
    """4. Demonstrate boundary condition: constant dimensions (max == min) produce no NaNs."""
    # --- [TODO: Student Implementation Start] ---
    # Create constant actions: const_a = torch.tensor([[5.0, 5.0, 5.0]])
    # min_vals = max_vals = torch.tensor([5.0, 5.0, 5.0])
    # normalizer = MinMaxNormalizer(min_vals, max_vals)
    # norm = normalizer.normalize(const_a)
    # Assert not torch.isnan(norm).any()
    # Return True
    raise NotImplementedError("Step 4: Verify division by zero protection on constant dimension")
    # --- [TODO: Student Implementation End] ---


def main() -> None:
    """Entry point for Day 23."""
    print("=== Day 23: Action Space Min-Max Normalization ===")

    if not HDF5_PATH.exists():
        print(f"  [Warning] {HDF5_PATH} does not exist. Run mock generator first.")
        return

    # Step 1: Compute and save stats
    min_bounds, max_bounds = compute_global_action_stats(HDF5_PATH)
    normalizer = MinMaxNormalizer(min_bounds, max_bounds)
    normalizer.save_stats(STATS_PATH)
    print(f"  [Step 1] Computed global action stats. Saved to {STATS_PATH}")
    print(f"    Min bounds: {min_bounds.round(3).tolist()}")
    print(f"    Max bounds: {max_bounds.round(3).tolist()}")

    # Step 2 & 3: Normalization and parity test
    with h5py.File(HDF5_PATH, "r") as h5:
        demo_actions = torch.from_numpy(h5["data/demo_0/actions"][:]).float()

    min_norm, max_norm = verify_normalization_and_roundtrip(normalizer, demo_actions)
    print(f"  [Step 2 & 3] Normalization range verified: [{min_norm:.3f}, {max_norm:.3f}] with roundtrip parity")

    # Step 4: Zero-variance boundary test
    zero_var_ok = demonstrate_zero_variance_boundary()
    print(f"  [Step 4] Zero-variance division protection verified: {zero_var_ok}")

    print("\n>>> Day 23 Acceptance: ALL PASS")


if __name__ == "__main__":
    main()
