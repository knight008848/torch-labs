"""
Min-Max Normalization Module for Robot Actions and States.

Supports mapping continuous vectors to/from [-1.0, 1.0] with zero-variance protection.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


class MinMaxNormalizer:
    """Normalizes tensors/arrays linearly to [-1.0, 1.0] using precomputed statistics."""

    def __init__(
        self,
        min_vals: torch.Tensor | np.ndarray,
        max_vals: torch.Tensor | np.ndarray,
        eps: float = 1e-8,
    ) -> None:
        """Initialize with minimum and maximum reference vectors.

        Args:
            min_vals: Minimum vector across dimensions.
            max_vals: Maximum vector across dimensions.
            eps: Epsilon to prevent division by zero when max == min.
        """
        if isinstance(min_vals, np.ndarray):
            min_vals = torch.from_numpy(min_vals).float()
        if isinstance(max_vals, np.ndarray):
            max_vals = torch.from_numpy(max_vals).float()

        self.min_vals = min_vals.float()
        self.max_vals = max_vals.float()
        self.eps = eps

        # Detect degenerate zero-range dimensions
        self.range_vals = self.max_vals - self.min_vals
        self.range_vals = torch.where(self.range_vals < self.eps, torch.tensor(1.0), self.range_vals)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize x from [min_vals, max_vals] to [-1.0, 1.0].

        Formula: 2 * (x - min) / (max - min) - 1.0

        TODO(Student):
            1. Ensure device parity between x and self.min_vals / self.range_vals.
            2. Compute normalized = 2.0 * (x - min_vals) / range_vals - 1.0.
            3. Clamp output to [-1.0, 1.0] to safeguard against out-of-distribution extremes.
            4. Return normalized tensor.
        """
        # --- [TODO: Student Implementation Start] ---
        raise NotImplementedError("Implement Min-Max normalization to [-1.0, 1.0]")
        # --- [TODO: Student Implementation End] ---

    def unnormalize(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Denormalize x_norm from [-1.0, 1.0] back to original scale.

        Formula: (x_norm + 1.0) / 2.0 * (max - min) + min

        TODO(Student):
            1. Compute x = (x_norm + 1.0) * 0.5 * range_vals + min_vals.
            2. Return denormalized tensor.
        """
        # --- [TODO: Student Implementation Start] ---
        raise NotImplementedError("Implement unnormalize back to original domain")
        # --- [TODO: Student Implementation End] ---

    def save_stats(self, save_path: Path) -> None:
        """Serialize normalization parameters to JSON."""
        save_path.parent.mkdir(parents=True, exist_ok=True)
        stats = {
            "min": self.min_vals.tolist(),
            "max": self.max_vals.tolist(),
            "eps": self.eps,
        }
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

    @classmethod
    def load_stats(cls, stats_path: Path) -> MinMaxNormalizer:
        """Instantiate normalizer from serialized JSON stats."""
        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
        return cls(
            min_vals=torch.tensor(stats["min"]),
            max_vals=torch.tensor(stats["max"]),
            eps=stats.get("eps", 1e-8),
        )
