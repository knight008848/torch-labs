"""
EmbodiedDataset: Multi-modal embodied robot demonstration dataset.

Progression across learning sprint:
    - v0.1 (Day 12): Initial skeleton reading mock HDF5 returning {"image", "state", "action"}.
    - v0.2 (Day 16): Real HDF5 file integration and channel order determination.
    - v0.3 (Day 18): IPC-safe worker_init_fn per-worker handle opening.
    - v0.4 (Day 20): Custom collate_fn and batch multimodal packaging.

Constraints & Iron Rules:
    - NEVER instantiate h5py.File inside __init__ (avoids fork deadlock in multi-worker DataLoader).
    - File handles must be opened lazily in __getitem__ or inside worker_init_fn.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class EmbodiedDataset(Dataset[dict[str, torch.Tensor]]):
    """Multi-modal dataset indexing demonstration trajectories from HDF5 archives."""

    def __init__(
        self,
        hdf5_path: str | Path,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        """Initialize dataset index from HDF5 file without holding an open handle.

        Args:
            hdf5_path: Path to the HDF5 archive (mock or real teleop demo).
            transform: Optional vision transformation applied to image tensors.

        TODO(Student - v0.1):
            1. Validate that hdf5_path exists on disk; raise FileNotFoundError if not.
            2. Store self.hdf5_path as Path.
            3. Open HDF5 temporarily to build index mapping (demo keys and cumulative frame counts),
               then IMMEDIATELY close it.
            4. self._file = None (DO NOT keep persistent open file here!).
        """
        self.hdf5_path = Path(hdf5_path)
        self.transform = transform
        self._file: h5py.File | None = None

        # --- [TODO: Student Implementation Start] ---
        raise NotImplementedError("v0.1: Build frame index metadata and ensure file is closed")
        # --- [TODO: Student Implementation End] ---

    def __len__(self) -> int:
        """Return total number of frames across all indexed trajectories."""
        # --- [TODO: Student Implementation Start] ---
        raise NotImplementedError("v0.1: Return total frame count")
        # --- [TODO: Student Implementation End] ---

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Fetch frame at index as multi-modal dictionary.

        Returns:
            dict containing:
                - 'image': torch.Tensor [C, H, W] or [H, W, C]
                - 'state': torch.Tensor [7]
                - 'action': torch.Tensor [7]

        TODO(Student - v0.1):
            1. Check boundary: if index < 0 or index >= len(self), raise IndexError.
            2. Locate corresponding demo and intra-demo frame offset.
            3. Lazily open h5py.File if self._file is None.
            4. Read raw observation, state, and action slice for this single frame.
            5. Convert NumPy arrays to PyTorch tensors.
            6. Return dictionary {'image': image_tensor, 'state': state_tensor, 'action': action_tensor}.
        """
        # --- [TODO: Student Implementation Start] ---
        raise NotImplementedError("v0.1: Lazily read frame slice and return multimodal dict")
        # --- [TODO: Student Implementation End] ---

    def close(self) -> None:
        """Close underlying HDF5 file handle if open."""
        if self._file is not None:
            self._file.close()
            self._file = None
