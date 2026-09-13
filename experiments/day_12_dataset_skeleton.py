"""
Day 12 / 2026-09-22 / EmbodiedDataset Pipeline Module A Skeleton
Runtime: <fill in after completion>

Goal:
    Validate EmbodiedDataset v0.1 interface contract: verify that indexing dataset[0]
    yields a multi-modal dictionary containing 'image', 'state', and 'action' tensors
    with expected shapes and dtypes on synthetic/mock HDF5 trajectories.

Acceptance:
    1. Initialize EmbodiedDataset pointing to data/raw/teleop_demo.hdf5 without opening
       persistent file handles in __init__.
    2. Extract sample 0: verify returned dict contains keys 'image', 'state', 'action'.
    3. Assert tensor shapes and types:
       - image: uint8 tensor of shape [84, 84, 3] (or [3, 84, 84])
       - state: float32 tensor of shape [7]
       - action: float32 tensor of shape [7]
    4. Handle boundary conditions: non-existent file raises FileNotFoundError, out-of-bounds
       index raises IndexError.

Related concepts:
    D04 (num_workers consideration), P04 (HDF5 lazy loading)

Hub position:
    Hub (4) Dataset/DataLoader. First embodied multi-modal dataset pipeline module.

Constraints:
    - Headless: no cv2.imshow / waitKey / createTrackbar / setMouseCallback
    - Pure functions called from main()
    - Max 3 new concepts
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.embodied_dataset import EmbodiedDataset


def verify_dataset_keys_and_shapes(dataset: EmbodiedDataset) -> dict[str, Any]:
    """1, 2 & 3. Extract sample 0 and assert key presence and tensor specifications.

    TODO(Student):
        1. Fetch sample = dataset[0].
        2. Assert isinstance(sample, dict).
        3. Assert all(k in sample for k in ('image', 'state', 'action')).
        4. Assert isinstance(sample['image'], torch.Tensor).
        5. Assert isinstance(sample['state'], torch.Tensor) and sample['state'].shape == (7,).
        6. Assert isinstance(sample['action'], torch.Tensor) and sample['action'].shape == (7,).
        7. Return dictionary of extracted metadata: {'keys': list(sample.keys()), 'img_shape': tuple(sample['image'].shape)}.
    """
    # --- [TODO: Student Implementation Start] ---
    raise NotImplementedError("Step 1-3: Index dataset[0] and assert multi-modal tensor contract")
    # --- [TODO: Student Implementation End] ---

    return metadata


def demonstrate_missing_file_boundary(bogus_path: Path) -> bool:
    """4. Demonstrate boundary condition: instantiating with non-existent path raises FileNotFoundError."""
    # --- [TODO: Student Implementation Start] ---
    # Attempt EmbodiedDataset(bogus_path), catch FileNotFoundError, return True if caught.
    raise NotImplementedError("Step 4: Catch FileNotFoundError on non-existent HDF5 path")
    # --- [TODO: Student Implementation End] ---


def demonstrate_out_of_bounds_boundary(dataset: EmbodiedDataset) -> bool:
    """4. Demonstrate boundary condition: accessing index >= len(dataset) raises IndexError."""
    # --- [TODO: Student Implementation Start] ---
    # Attempt dataset[len(dataset)], catch IndexError, return True if caught.
    raise NotImplementedError("Step 4: Catch IndexError on out-of-bounds frame index")
    # --- [TODO: Student Implementation End] ---


def main() -> None:
    """Entry point for Day 12."""
    print("=== Day 12: EmbodiedDataset v0.1 Skeleton ===")

    hdf5_file = ROOT / "data" / "raw" / "teleop_demo.hdf5"
    print(f"  [Setup] Target HDF5 path: {hdf5_file} (exists: {hdf5_file.exists()})")

    # Step 1-3: Interface contract verification
    dataset = EmbodiedDataset(hdf5_file)
    meta = verify_dataset_keys_and_shapes(dataset)
    print(f"  [Step 1-3] Successfully indexed dataset[0]: keys={meta['keys']}, img_shape={meta['img_shape']}")

    # Step 4: Boundary demonstrations
    missing_ok = demonstrate_missing_file_boundary(ROOT / "data" / "raw" / "non_existent.hdf5")
    print(f"  [Step 4] FileNotFoundError boundary caught: {missing_ok}")

    oob_ok = demonstrate_out_of_bounds_boundary(dataset)
    print(f"  [Step 4] IndexError boundary caught: {oob_ok}")

    print("\n>>> Day 12 Acceptance: ALL PASS")


if __name__ == "__main__":
    main()
