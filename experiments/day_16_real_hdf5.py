"""
Day 16 / 2026-09-26 / Replace Mock with Real HDF5 and Determine Camera Channel Order
Runtime: <fill in after completion>

Goal:
    Transition EmbodiedDataset from synthetic mock to real teleoperation HDF5 dataset (v0.2):
    verify dataset[0] index pass, ensure continuous frame indexing across demo_0 without off-by-one
    errors, empirically determine true channel order (RGB vs BGR), and document conclusions.

Acceptance:
    1. Verify dataset[0] retrieves multi-modal dictionary under actual HDF5 archive without error.
    2. Sequentially index all N frames of demonstration 0, verifying monotonicity and bounds integrity.
    3. Empirically determine the true camera channel order of obs/agentview_image based on Day 15 visual
       evidence, documenting the decision in src/embodied_dataset.py comments.
    4. Handle boundary condition: verify that indexing beyond the total dataset length raises IndexError.

Related concepts:
    P04 (HDF5 continuous indexing), P01 (Camera channel order determination)

Hub position:
    Hub (4) Dataset/DataLoader (Bridging raw robotic telemetry to PyTorch runtime).

Constraints:
    - Headless: no cv2.imshow / waitKey / createTrackbar / setMouseCallback
    - Pure functions called from main()
    - Max 3 new concepts
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import h5py
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.embodied_dataset import EmbodiedDataset

HDF5_PATH = ROOT / "data" / "raw" / "teleop_demo.hdf5"


def verify_real_dataset_sample_zero(dataset: EmbodiedDataset) -> dict[str, Any]:
    """1. Index dataset[0] and verify multi-modal contract on real HDF5 archive.

    TODO(Student):
        1. Fetch sample = dataset[0].
        2. Assert isinstance(sample['image'], torch.Tensor).
        3. Assert isinstance(sample['state'], torch.Tensor) and sample['state'].shape == (7,).
        4. Assert isinstance(sample['action'], torch.Tensor) and sample['action'].shape == (7,).
        5. Return dictionary of sample metadata.
    """
    # --- [TODO: Student Implementation Start] ---
    raise NotImplementedError("Step 1: Index dataset[0] on real HDF5 archive")
    # --- [TODO: Student Implementation End] ---

    return info


def verify_demo_sequential_indexing(dataset: EmbodiedDataset, demo_length: int) -> bool:
    """2. Sequentially index through all N frames of demo_0 without boundary errors.

    TODO(Student):
        1. For frame_idx in range(demo_length):
           sample = dataset[frame_idx]
           assert sample is not None
        2. Return True if all frames indexed successfully.
    """
    # --- [TODO: Student Implementation Start] ---
    raise NotImplementedError("Step 2: Traverse all N frames of demo_0 to verify index stability")
    # --- [TODO: Student Implementation End] ---

    return True


def determine_camera_channel_order(sample_img: torch.Tensor) -> str:
    """3. Determine true camera channel order based on observation ground truth.

    TODO(Student):
        Review Day 15 exported frame docs/figs/day_15_raw_frame.png:
        - If simulated agent camera stores raw buffer in RGB format, channel order is 'RGB'.
        - If OpenCV BGR order was recorded, channel order is 'BGR'.
        - Document the empirical rationale and return 'RGB' or 'BGR'.
    """
    # --- [TODO: Student Implementation Start] ---
    raise NotImplementedError("Step 3: Document camera channel order determination ('RGB' or 'BGR')")
    # --- [TODO: Student Implementation End] ---

    assert channel_order in ("RGB", "BGR")
    return channel_order


def demonstrate_out_of_range_index_boundary(dataset: EmbodiedDataset) -> bool:
    """4. Demonstrate boundary condition: indexing index == len(dataset) raises IndexError."""
    # --- [TODO: Student Implementation Start] ---
    # Attempt dataset[len(dataset)], catch IndexError, return True if caught.
    raise NotImplementedError("Step 4: Catch IndexError on out-of-range index")
    # --- [TODO: Student Implementation End] ---


def main() -> None:
    """Entry point for Day 16."""
    print("=== Day 16: Real HDF5 Integration & Channel Order Determination ===")

    if not HDF5_PATH.exists():
        print(f"  [Warning] {HDF5_PATH} does not exist yet. Please create mock or real data first.")
        return

    dataset = EmbodiedDataset(HDF5_PATH)
    print(f"  [Setup] Initialized EmbodiedDataset v0.2 with {len(dataset)} total frames")

    # Step 1: Sample 0 contract
    meta = verify_real_dataset_sample_zero(dataset)
    print(f"  [Step 1] dataset[0] passed: img_shape={meta['img_shape']}, dtype={meta['img_dtype']}")

    # Step 2: Sequential demo indexing
    # Get frame count of demo_0 directly via h5py inspection for verification
    with h5py.File(HDF5_PATH, "r") as h5:
        demo_0_len = h5["data/demo_0/actions"].shape[0]

    seq_ok = verify_demo_sequential_indexing(dataset, demo_0_len)
    print(f"  [Step 2] Sequentially indexed all {demo_0_len} frames in demo_0: {seq_ok}")

    # Step 3: Channel order determination
    channel_order = determine_camera_channel_order(dataset[0]["image"])
    print(f"  [Step 3] Camera channel order determined: {channel_order}")

    # Step 4: Boundary check
    boundary_ok = demonstrate_out_of_range_index_boundary(dataset)
    print(f"  [Step 4] Out-of-bounds IndexError caught: {boundary_ok}")

    print("\n>>> Day 16 Acceptance: ALL PASS")


if __name__ == "__main__":
    main()
