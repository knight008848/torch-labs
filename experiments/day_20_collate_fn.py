"""
Day 20 / 2026-09-30 / Multi-Modal Dictionary Contract and Custom collate_fn
Runtime: <fill in after completion>

Goal:
    Implement and rigorously test a custom collate_fn function that transforms a list of
    individual sample dictionaries into an orchestrated multi-modal batch dictionary
    with strict shape invariants, without relying on PyTorch's default collate heuristic.

Acceptance:
    1. Implement custom_multimodal_collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]
       assembling:
       - 'image': torch.Tensor [B, 3, 84, 84] (or [B, 84, 84, 3])
       - 'state': torch.Tensor [B, 7]
       - 'action': torch.Tensor [B, 7]
    2. Pass custom collate function to DataLoader and verify batch shape and dtype invariants.
    3. Cover three distinct boundary error conditions:
       - Empty batch list: raise ValueError
       - Mismatched shape across samples: raise ValueError / RuntimeError
       - Missing expected multi-modal key: raise KeyError with clear message.

Related concepts:
    T08 (torch.stack across batch dimension), D03 (Custom collate_fn implementation)

Hub position:
    Hub (4) Dataset/DataLoader (Multi-modal batch assembly bridging sample items to network inputs).

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


def custom_multimodal_collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """1. Explicit multi-modal batch collator without relying on default_collate.

    Args:
        batch: List of sample dictionaries, each with 'image', 'state', 'action'.

    Returns:
        Batched dictionary containing stacked torch.Tensors along dimension 0.

    TODO(Student):
        1. Boundary check: if not batch: raise ValueError("Cannot collate an empty batch").
        2. Verify all samples contain required keys ('image', 'state', 'action'). If any missing, raise KeyError.
        3. Verify shapes are consistent across all items in batch.
        4. Stack tensors along dim=0:
           images = torch.stack([item['image'] for item in batch], dim=0)
           states = torch.stack([item['state'] for item in batch], dim=0)
           actions = torch.stack([item['action'] for item in batch], dim=0)
        5. Return {'image': images, 'state': states, 'action': actions}.
    """
    # --- [TODO: Student Implementation Start] ---
    raise NotImplementedError("Step 1: Implement custom_multimodal_collate with validation and torch.stack")
    # --- [TODO: Student Implementation End] ---

    return batched


class SyntheticDictDataset(Dataset[dict[str, torch.Tensor]]):
    """Synthetic dataset for testing collation invariants."""

    def __init__(self, size: int = 16) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "image": torch.zeros((3, 84, 84), dtype=torch.float32),
            "state": torch.ones((7,), dtype=torch.float32) * float(idx),
            "action": torch.zeros((7,), dtype=torch.float32) + float(idx),
        }


def verify_dataloader_collate_integration() -> dict[str, tuple[int, ...]]:
    """2. Integrate custom collate_fn with DataLoader and assert batch shapes."""
    dataset = SyntheticDictDataset(size=16)

    # --- [TODO: Student Implementation Start] ---
    # loader = DataLoader(dataset, batch_size=4, collate_fn=custom_multimodal_collate)
    # batch = next(iter(loader))
    # Assert batch['image'].shape == (4, 3, 84, 84)
    # Assert batch['state'].shape == (4, 7)
    # Assert batch['action'].shape == (4, 7)
    raise NotImplementedError("Step 2: Connect custom collator to DataLoader and verify batch invariants")
    # --- [TODO: Student Implementation End] ---

    return shapes


def demonstrate_collate_boundary_cases() -> dict[str, bool]:
    """3. Demonstrate boundary conditions: empty batch, shape mismatch, and missing keys."""
    results = {}

    # Boundary 1: Empty batch
    # --- [TODO: Student Implementation Start] ---
    # Call custom_multimodal_collate([]), catch ValueError, record results['empty_batch'] = True
    raise NotImplementedError("Step 3: Catch ValueError on empty batch")
    # --- [TODO: Student Implementation End] ---

    # Boundary 2: Shape mismatch across samples
    # --- [TODO: Student Implementation Start] ---
    # batch_mismatched = [
    #     {'image': torch.zeros(3, 84, 84), 'state': torch.zeros(7), 'action': torch.zeros(7)},
    #     {'image': torch.zeros(3, 84, 84), 'state': torch.zeros(10), 'action': torch.zeros(7)},
    # ]
    # Attempt collate, catch ValueError or RuntimeError, record results['shape_mismatch'] = True
    raise NotImplementedError("Step 3: Catch exception on shape mismatch across samples")
    # --- [TODO: Student Implementation End] ---

    # Boundary 3: Missing expected key
    # --- [TODO: Student Implementation Start] ---
    # batch_missing_key = [
    #     {'image': torch.zeros(3, 84, 84), 'state': torch.zeros(7)},
    # ]
    # Attempt collate, catch KeyError, record results['missing_key'] = True
    raise NotImplementedError("Step 3: Catch KeyError on missing multimodal dictionary key")
    # --- [TODO: Student Implementation End] ---

    return results


def main() -> None:
    """Entry point for Day 20."""
    print("=== Day 20: Multi-Modal Dictionary Contract & Custom collate_fn ===")

    # Step 1 & 2: Collate integration with DataLoader
    shapes = verify_dataloader_collate_integration()
    print(f"  [Step 1 & 2] DataLoader custom collation verified. Batch shapes: {shapes}")

    # Step 3: Boundary error cases
    boundary_results = demonstrate_collate_boundary_cases()
    print(f"  [Step 3] Collate boundary tests passed: {boundary_results}")

    print("\n>>> Day 20 Acceptance: ALL PASS")


if __name__ == "__main__":
    main()
