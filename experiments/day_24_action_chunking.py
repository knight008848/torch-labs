"""
Day 24 / 2026-10-04 / Temporal Action Chunking with Sliding Window and Boundary Padding
Runtime: <fill in after completion>

Goal:
    Implement temporal action chunking (sliding window extracting future K action steps [K, 7]),
    handle trailing boundary frames where remaining trajectory length < K using an explicit
    padding/repeat strategy, and validate tensor outputs across horizons K=8 and K=16.

Acceptance:
    1. Implement slice_action_chunk(actions: torch.Tensor, current_idx: int, chunk_size: int) -> torch.Tensor
       returning future action matrix [chunk_size, 7].
    2. Handle end-of-episode boundary condition where current_idx + chunk_size > episode_length
       by either repeating the final action or zero-padding, with strategy documented in comments.
    3. Run and verify chunking invariants across both horizon sizes K=8 and K=16.
    4. Handle boundary condition: verify correct behavior when episode length is shorter than
       chunk_size (e.g. episode_length = 3, chunk_size = 8).

Related concepts:
    P08 (Action chunking temporal sliding window, trailing boundary strategies)

Hub position:
    Hub (4) Dataset/DataLoader (Temporal sequence dimensioning for imitation learning policies).

Constraints:
    - Headless: no cv2.imshow / waitKey / createTrackbar / setMouseCallback
    - Pure functions called from main()
    - Max 3 new concepts
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]


def slice_action_chunk(
    actions: torch.Tensor,
    current_idx: int,
    chunk_size: int = 8,
    pad_strategy: str = "repeat_last",
) -> torch.Tensor:
    """1 & 2. Slice future K-step actions starting from current_idx with boundary padding.

    Args:
        actions: Full episode action sequence of shape [N, 7].
        current_idx: Current temporal frame index in [0, N - 1].
        chunk_size: Future prediction horizon K.
        pad_strategy: 'repeat_last' (repeat final action) or 'zero' (zero padding).

    Returns:
        torch.Tensor of shape [chunk_size, 7].

    TODO(Student):
        1. Calculate available frames: remaining = actions.shape[0] - current_idx.
        2. If remaining >= chunk_size:
           return actions[current_idx : current_idx + chunk_size].clone()
        3. If remaining < chunk_size:
           slice available: available_chunk = actions[current_idx:]
           pad_count = chunk_size - remaining
           if pad_strategy == 'repeat_last':
               pad_tensor = available_chunk[-1:].repeat(pad_count, 1)
           elif pad_strategy == 'zero':
               pad_tensor = torch.zeros((pad_count, actions.shape[1]), dtype=actions.dtype)
           return torch.cat([available_chunk, pad_tensor], dim=0)
    """
    # --- [TODO: Student Implementation Start] ---
    raise NotImplementedError("Step 1 & 2: Implement slice_action_chunk with boundary padding")
    # --- [TODO: Student Implementation End] ---

    assert chunk.shape == (chunk_size, actions.shape[1]), f"Expected shape ({chunk_size}, 7), got {chunk.shape}"
    return chunk


class ActionChunkingDataset(Dataset[dict[str, torch.Tensor]]):
    """Dataset wrapper outputting action chunks for policy training."""

    def __init__(self, actions: torch.Tensor, chunk_size: int = 8) -> None:
        self.actions = actions
        self.chunk_size = chunk_size

    def __len__(self) -> int:
        return self.actions.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        chunk = slice_action_chunk(self.actions, idx, chunk_size=self.chunk_size)
        return {"action_chunk": chunk}


def verify_chunk_horizons_k8_and_k16(actions: torch.Tensor) -> dict[int, tuple[int, ...]]:
    """3. Verify action chunk shapes across horizons K=8 and K=16."""
    results = {}
    for k in (8, 16):
        # --- [TODO: Student Implementation Start] ---
        # Test middle frame and final boundary frame for chunk size k
        # ds = ActionChunkingDataset(actions, chunk_size=k)
        # mid_chunk = ds[len(ds) // 2]['action_chunk']
        # end_chunk = ds[len(ds) - 1]['action_chunk']
        # Assert both have shape (k, 7)
        raise NotImplementedError("Step 3: Verify K=8 and K=16 chunking across middle and end frames")
        # --- [TODO: Student Implementation End] ---
        results[k] = (k, actions.shape[1])
    return results


def demonstrate_short_episode_boundary(chunk_size: int = 8) -> bool:
    """4. Demonstrate boundary condition: episode length < chunk_size still returns [K, 7]."""
    # --- [TODO: Student Implementation Start] ---
    # Create ultra-short episode: short_actions = torch.randn(3, 7)
    # chunk = slice_action_chunk(short_actions, current_idx=0, chunk_size=chunk_size)
    # Assert chunk.shape == (chunk_size, 7)
    # Return True
    raise NotImplementedError("Step 4: Verify short episode (< K) correctly padded to [K, 7]")
    # --- [TODO: Student Implementation End] ---


def main() -> None:
    """Entry point for Day 24."""
    print("=== Day 24: Temporal Action Chunking & Boundary Padding ===")

    # Synthetic trajectory of 50 steps
    trajectory_actions = torch.cumsum(torch.randn(50, 7) * 0.1, dim=0)
    print(f"  [Setup] Created synthetic action trajectory of {len(trajectory_actions)} steps")

    # Step 1-3: Verify K=8 and K=16
    horizons = verify_chunk_horizons_k8_and_k16(trajectory_actions)
    print(f"  [Step 1-3] Action chunking verified for horizons: {horizons}")

    # Step 4: Boundary test with short episode
    short_ep_ok = demonstrate_short_episode_boundary(chunk_size=8)
    print(f"  [Step 4] Short episode boundary padding verified: {short_ep_ok}")

    print("\n>>> Day 24 Acceptance: ALL PASS")


if __name__ == "__main__":
    main()
