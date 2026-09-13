"""
Day 10 / 2026-09-20 / Complete training loop, train/eval modes, and checkpoint persistence
Runtime: <fill in after completion>

Goal:
    Master full PyTorch training loop orchestration (two-tier epoch/batch iterations,
    model.train() vs model.eval() toggling, loss monitoring) and checkpoint persistence
    (torch.save / load_state_dict verification) with state parity assertion.

Acceptance:
    1. Execute a complete 5-epoch training loop on synthetic batch data with loss decrease logging.
    2. Correctly toggle model.train() during optimization and model.eval() with torch.no_grad()
       during evaluation.
    3. Save model checkpoint to models/day_10_mlp.pt and verify that a fresh model instance loaded
       with load_state_dict produces bit-exact predictions (torch.allclose).
    4. Handle boundary condition: demonstrate strict=True vs strict=False behavior when loading
       a state_dict with missing/unexpected keys.

Related concepts:
    M06 (train/eval loop architecture), M07 (torch.save / load_state_dict persistence)

Hub position:
    Hub (3) nn.Module. Completes the pipeline consumer lifecycle: Data -> Tensors -> Forward -> Backward -> Weights on disk.

Constraints:
    - Headless: no cv2.imshow / waitKey / createTrackbar / setMouseCallback
    - Save checkpoints to models/
    - Max 3 new concepts
    - Pure functions called from main()
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"


class SimpleRegressionModel(nn.Module):
    """Simple 2-layer MLP with BatchNorm to clearly differentiate train() vs eval()."""

    def __init__(self, in_features: int = 7, out_features: int = 7) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, 16)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.bn1(self.fc1(x))))


def run_training_loop(
    model: nn.Module,
    epochs: int = 5,
    batch_size: int = 16,
    batches_per_epoch: int = 4,
) -> list[float]:
    """1 & 2. Run a 5-epoch training loop and log epoch loss.

    TODO(Student):
        1. Set optimizer = optim.Adam(model.parameters(), lr=0.01).
        2. Set criterion = nn.MSELoss().
        3. For epoch in range(epochs):
           - model.train()
           - For each batch:
               x = torch.randn(batch_size, 7)
               y = x * 1.5 + 0.5  # target mapping
               optimizer.zero_grad()
               loss = criterion(model(x), y)
               loss.backward()
               optimizer.step()
           - Evaluation step:
               model.eval()
               with torch.no_grad():
                   val_x = torch.randn(32, 7)
                   val_y = val_x * 1.5 + 0.5
                   val_loss = criterion(model(val_x), val_y).item()
               record val_loss into history list.
        4. Return val_loss history across epochs.
    """
    # --- [TODO: Student Implementation Start] ---
    raise NotImplementedError("Step 1 & 2: Implement 5-epoch training loop with train()/eval() switching")
    # --- [TODO: Student Implementation End] ---

    assert len(loss_history) == epochs
    assert loss_history[-1] < loss_history[0], f"Expected loss decrease from {loss_history[0]} to {loss_history[-1]}"
    return loss_history


def save_and_verify_checkpoint(model: nn.Module, save_path: Path) -> None:
    """3. Save checkpoint via torch.save and load into a new model instance.

    TODO(Student):
        1. Ensure save_path parent directory exists.
        2. torch.save(model.state_dict(), save_path).
        3. Instantiate new_model = SimpleRegressionModel().
        4. new_model.load_state_dict(torch.load(save_path, weights_only=True)).
        5. Set both models to eval().
        6. Pass identical test_x = torch.randn(8, 7) through both models.
        7. Assert torch.allclose(model(test_x), new_model(test_x), atol=1e-6).
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # --- [TODO: Student Implementation Start] ---
    raise NotImplementedError("Step 3: Save state_dict and assert exact prediction equality after reload")
    # --- [TODO: Student Implementation End] ---

    assert save_path.exists(), f"Checkpoint not found at {save_path}"


def demonstrate_mismatched_checkpoint_boundary(model: nn.Module) -> str:
    """4. Demonstrate boundary condition: strict=True vs strict=False with key mismatch.

    TODO(Student):
        1. Obtain a copy of model.state_dict().
        2. Delete one key (e.g. 'fc2.bias') or add a bogus key ('extra_layer.weight').
        3. Attempt model.load_state_dict(corrupted_dict, strict=True) in try/except.
        4. Catch the RuntimeError, verify 'Missing key(s)' or 'Unexpected key(s)' in str(e).
        5. Verify model.load_state_dict(corrupted_dict, strict=False) runs without error.
        6. Return caught error string.
    """
    # --- [TODO: Student Implementation Start] ---
    raise NotImplementedError("Step 4: Catch RuntimeError on strict=True mismatched state_dict")
    # --- [TODO: Student Implementation End] ---

    assert "missing" in caught_err.lower() or "unexpected" in caught_err.lower()
    return caught_err


def main() -> None:
    """Entry point for Day 10."""
    print("=== Day 10: Training Loop & Checkpoint Persistence ===")

    # Initialize model
    model = SimpleRegressionModel(in_features=7, out_features=7)
    print(f"  [Setup] Initialized SimpleRegressionModel: {sum(p.numel() for p in model.parameters())} params")

    # Step 1 & 2: Complete training loop
    val_losses = run_training_loop(model, epochs=5)
    print(f"  [Step 1 & 2] 5 epochs completed. Loss trajectory: {[round(l, 4) for l in val_losses]}")

    # Step 3: Checkpoint save & load parity
    ckpt_file = MODELS_DIR / "day_10_mlp.pt"
    save_and_verify_checkpoint(model, ckpt_file)
    print(f"  [Step 3] Checkpoint saved and verified with bit-exact parity at {ckpt_file}")

    # Step 4: Key mismatch boundary condition
    boundary_msg = demonstrate_mismatched_checkpoint_boundary(model)
    print(f"  [Step 4] Mismatched state_dict boundary caught:\n    '{boundary_msg[:70]}...'")

    print("\n>>> Day 10 Acceptance: ALL PASS")


if __name__ == "__main__":
    main()
