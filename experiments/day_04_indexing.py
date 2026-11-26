"""
Day 04 / 2026-11-26 / Tensor indexing, slicing, and boolean masking
Runtime: 3.1 s

Goal:
    Master standard slicing, advanced indexing, and boolean masking on trajectory
    tensors, and strictly distinguish dimension-preserving slicing [1, D] from
    rank-reducing indexing [D].

Acceptance:
    1. Slice a trajectory segment from joint position tensor [N, 7] using step intervals.
    2. Apply boolean masking to filter all frames where joint index 0 exceeds a threshold
       (> 0.5 rad).
    3. Contrast dimension-preserving slice [0:1, :] with rank-reducing index [0, :], asserting
       shapes [1, 7] vs [7]. Note: [7] @ [7, 7] is legal (yields [7]); the rank-1 form breaks
       torch.cat instead, which rejects mixing 1-D and 2-D tensors.
    4. Handle boundary condition: empty boolean mask returning an empty tensor of shape [0, 7].
    5. Exercise real trajectory data from data/raw/teleop_demo.hdf5 (demo_0/obs/robot0_joint_pos).

Related concepts:
    T05 (indexing, slicing, boolean masks)

Hub position:
    Hub (1) Tensor. Indexing and masking are the primary mechanisms for window slicing,
    episode boundary splitting, and condition-based frame filtering in the data pipeline.

Constraints:
    - Headless: no cv2.imshow / waitKey / createTrackbar / setMouseCallback
    - Max 3 new concepts
    - Pure functions called from main()
"""

from __future__ import annotations

from pathlib import Path

import h5py
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "teleop_demo.hdf5"


def load_joint_trajectory() -> torch.Tensor:
    """Load robot joint position trajectory from teleop_demo.hdf5."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Run Day 01 first.")

    with h5py.File(DATA_PATH, "r") as f:
        joint_pos_np = f["data/demo_0/obs/robot0_joint_pos"][:]

    trajectory = torch.from_numpy(joint_pos_np).float()
    assert trajectory.ndim == 2 and trajectory.shape[1] == 7
    return trajectory


def slice_trajectory_subsegment(
    trajectory: torch.Tensor, start_idx: int = 10, end_idx: int = 50, step: int = 2
) -> torch.Tensor:
    """1. Slice trajectory segment from start_idx to end_idx with step."""
    segment = trajectory[start_idx:end_idx:step, :]

    expected_len = len(range(start_idx, end_idx, step))
    expected_shape = (expected_len, trajectory.shape[1])
    if segment.shape != expected_shape:
        raise ValueError(f"Expected shape {expected_shape}, got {segment.shape}")
    return segment


def boolean_mask_filtering(
    trajectory: torch.Tensor, joint_idx: int = 0, threshold: float = 0.5
) -> torch.Tensor:
    """2. Filter frames where trajectory[:, joint_idx] > threshold."""
    mask = trajectory[:, joint_idx] > threshold
    filtered = trajectory[mask]

    expected_rows = int((trajectory[:, joint_idx] > threshold).sum())
    assert filtered.shape == (expected_rows, trajectory.shape[1]), (
        f"Expected ({expected_rows}, {trajectory.shape[1]}), got {filtered.shape}"
    )
    if filtered.numel() > 0:
        assert (filtered[:, joint_idx] > threshold).all()
    return filtered


def contrast_rank_reduction_vs_preservation(
    trajectory: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """3. Contrast rank-reduction [D] with dimension preservation [1, D]."""
    reduced = trajectory[0, :]
    preserved = trajectory[0:1, :]

    n_joints = trajectory.shape[1]
    assert reduced.shape == (n_joints,) and reduced.ndim == 1, (
        f"Expected [{n_joints}], got {reduced.shape}"
    )
    assert preserved.shape == (1, n_joints) and preserved.ndim == 2, (
        f"Expected [1, {n_joints}], got {preserved.shape}"
    )
    return reduced, preserved


def demonstrate_empty_mask_boundary(trajectory: torch.Tensor) -> torch.Tensor:
    """4. Demonstrate boundary condition: all-False mask produces [0, D] tensor."""
    # Derive an impossible threshold instead of a magic number, so the mask is
    # all-False for any trajectory rather than only for this one.
    mask = trajectory[:, 0] > trajectory.max() + 1.0
    empty_tensor = trajectory[mask]

    n_joints = trajectory.shape[1]
    assert empty_tensor.shape == (0, n_joints), (
        f"Expected (0, {n_joints}), got {empty_tensor.shape}"
    )
    return empty_tensor


def main() -> None:
    """Entry point for Day 04."""
    print("=== Day 04: Tensor Indexing, Slicing & Boolean Masking ===")

    # Setup: Load real data
    traj = load_joint_trajectory()
    print(f"  [Setup] Loaded robot0_joint_pos: shape={traj.shape}, dtype={traj.dtype}")

    # Step 1: Subsegment slicing
    sub = slice_trajectory_subsegment(traj, start_idx=10, end_idx=50, step=2)
    print(f"  [Step 1] Sliced subsegment [10:50:2]: shape={sub.shape}")

    # Step 2: Boolean mask
    threshold = 0.5
    filtered = boolean_mask_filtering(traj, joint_idx=0, threshold=threshold)
    print(
        f"  [Step 2] Filtered frames where joint[0] > {threshold}: "
        f"matched {filtered.shape[0]} / {traj.shape[0]} frames"
    )

    # Step 3: Rank reduction vs dimension preservation
    reduced, preserved = contrast_rank_reduction_vs_preservation(traj)
    print(
        f"  [Step 3] Preserved vs Reduced: [0:1, :] -> {preserved.shape}, [0, :] -> {reduced.shape}"
    )

    # Step 4: Empty mask boundary condition
    empty = demonstrate_empty_mask_boundary(traj)
    print(f"  [Step 4] Empty mask boundary verified: shape={empty.shape}, numel={empty.numel()}")

    print("\n>>> Day 04 Acceptance: ALL PASS")


if __name__ == "__main__":
    main()
