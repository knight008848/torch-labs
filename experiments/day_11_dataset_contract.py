"""
Day 11 / 2026-09-21 / Dataset and DataLoader Interface Contract
Runtime: <fill in after completion>

Goal:
    Master PyTorch Dataset and DataLoader contract: implement __len__ and __getitem__,
    configure DataLoader batching, and verify sample-to-batch shape/dtype parity.

Acceptance:
    1. Implement a custom Dataset subclass satisfying __len__ and __getitem__ contracts.
    2. Instantiate DataLoader(batch_size=4, shuffle=True) and extract an orchestrated batch.
    3. Verify shape and dtype parity: dataset[0] structure matches sample 0 in next(iter(loader)).
    4. Handle boundary conditions: raise IndexError on out-of-bounds indices, and demonstrate
       why shuffle=False and drop_last=False are mandatory for reproducible throughput benchmarks.

Related concepts:
    D01 (Dataset contract), D02 (DataLoader), D03 (collate_fn intro), D06 (drop_last / shuffle)

Hub position:
    Hub (4) Dataset/DataLoader. Establishes the foundational interface contract connecting
    storage layers to batch computation.

Constraints:
    - Headless: no cv2.imshow / waitKey / createTrackbar / setMouseCallback
    - Pure functions called from main()
    - Max 3 new concepts
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]


class SyntheticTrajectoryDataset(Dataset[dict[str, torch.Tensor]]):
    """Synthetic dataset simulating multi-modal embodied robot observation frames."""

    def __init__(self, num_samples: int = 64) -> None:
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")
        self.num_samples = num_samples

    def __len__(self) -> int:
        """Return total number of samples in the dataset.

        TODO(Student):
            Return the total sample count (self.num_samples).
        """
        # --- [TODO: Student Implementation Start] ---
        raise NotImplementedError("Step 1: Implement __len__")
        # --- [TODO: Student Implementation End] ---

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Fetch a single sample dictionary by index.

        Args:
            index: Integer index within [0, len(self) - 1].

        Returns:
            Dictionary containing:
                - 'image': [84, 84, 3] uint8 tensor
                - 'state': [7] float32 tensor
                - 'action': [7] float32 tensor

        TODO(Student):
            1. Validate index: if index < 0 or index >= self.num_samples: raise IndexError.
            2. Generate deterministic or synthetic data for the given index:
               - image: torch.full((84, 84, 3), fill_value=(index % 256), dtype=torch.uint8)
               - state: torch.ones(7, dtype=torch.float32) * float(index)
               - action: torch.zeros(7, dtype=torch.float32) + float(index) * 0.1
            3. Return dictionary with keys 'image', 'state', 'action'.
        """
        # --- [TODO: Student Implementation Start] ---
        raise NotImplementedError("Step 1: Implement __getitem__ with boundary check and sample dict")
        # --- [TODO: Student Implementation End] ---


def inspect_dataloader_batch(
    dataset: Dataset[dict[str, torch.Tensor]],
    batch_size: int = 4,
    shuffle: bool = True,
) -> dict[str, torch.Tensor]:
    """2 & 3. Create DataLoader and extract a batch, verifying shape and type contracts.

    TODO(Student):
        1. Instantiate loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle).
        2. Fetch first batch = next(iter(loader)).
        3. Assert batch['image'].shape == (batch_size, 84, 84, 3).
        4. Assert batch['state'].shape == (batch_size, 7).
        5. Assert batch['action'].shape == (batch_size, 7).
        6. Verify parity with dataset[0]:
           - Assert batch['state'][0].shape == dataset[0]['state'].shape
           - Assert batch['image'][0].dtype == dataset[0]['image'].dtype
        7. Return the extracted batch.
    """
    # --- [TODO: Student Implementation Start] ---
    raise NotImplementedError("Step 2 & 3: Create DataLoader, fetch batch, and assert shape parity")
    # --- [TODO: Student Implementation End] ---

    return first_batch


def demonstrate_benchmark_flag_invariants(num_samples: int = 10, batch_size: int = 4) -> dict[str, int]:
    """4. Demonstrate why shuffle=False and drop_last=False are vital for benchmarks.

    TODO(Student):
        1. Test drop_last=True vs drop_last=False:
           - Calculate total samples processed when drop_last=True vs drop_last=False.
           - For num_samples=10, batch_size=4:
             drop_last=True yields 2 batches = 8 samples (2 dropped).
             drop_last=False yields 3 batches = 10 samples (all retained).
        2. Test shuffle=False determinism:
           - Sequential index access allows cache-friendly sequential reading and bit-exact reproducibility.
        3. Return dict summarizing sample counts: {'dropped_total': 8, 'retained_total': 10}.
    """
    # --- [TODO: Student Implementation Start] ---
    raise NotImplementedError("Step 4: Demonstrate sample count differences with drop_last=True/False")
    # --- [TODO: Student Implementation End] ---

    return results


def demonstrate_index_error_boundary(dataset: Dataset[Any]) -> bool:
    """Demonstrate boundary condition: out-of-bounds indexing raises IndexError."""
    # --- [TODO: Student Implementation Start] ---
    # Attempt dataset[len(dataset)] and dataset[-100] (or out of bounds)
    # Catch IndexError and return True if caught.
    raise NotImplementedError("Step 4: Catch IndexError on out-of-bounds index")
    # --- [TODO: Student Implementation End] ---


def main() -> None:
    """Entry point for Day 11."""
    print("=== Day 11: Dataset and DataLoader Interface Contract ===")

    # Step 1: Instantiate synthetic dataset
    dataset = SyntheticTrajectoryDataset(num_samples=32)
    print(f"  [Step 1] Initialized SyntheticTrajectoryDataset with {len(dataset)} samples")
    sample_0 = dataset[0]
    print(f"    dataset[0] keys: {list(sample_0.keys())}, image shape: {sample_0['image'].shape}")

    # Step 2 & 3: DataLoader batch parity
    batch = inspect_dataloader_batch(dataset, batch_size=4, shuffle=True)
    print(f"  [Step 2 & 3] Fetched batch. Image batch shape: {batch['image'].shape}")

    # Step 4: Boundary & Benchmark invariants
    bench_info = demonstrate_benchmark_flag_invariants(num_samples=10, batch_size=4)
    print(f"  [Step 4] Benchmark flag invariants: {bench_info}")

    boundary_passed = demonstrate_index_error_boundary(dataset)
    print(f"  [Step 4] IndexError boundary caught successfully: {boundary_passed}")

    print("\n>>> Day 11 Acceptance: ALL PASS")


if __name__ == "__main__":
    main()
