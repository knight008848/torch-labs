"""
Day 09 / 2026-09-19 / Loss functions, optimizers, and the four-step optimization contract
Runtime: <fill in after completion>

Goal:
    Master PyTorch loss criteria (nn.MSELoss) and optimizers (SGD vs Adam), contrast
    their convergence dynamics on linear regression (y = 2x + 1) with headless visualization,
    and memorize the four-step optimization contract (zero_grad -> forward -> backward -> step).

Acceptance:
    1. Fit a linear model (nn.Linear(1, 1)) to y = 2x + 1 using nn.MSELoss and assert weight ~ 2.0, bias ~ 1.0.
    2. Track convergence curves of SGD vs Adam, saving comparison plot to docs/figs/day_09_sgd_vs_adam.png.
    3. Strictly enforce the four-step order: zero_grad -> forward -> backward -> step.
    4. Handle boundary condition: demonstrate loss divergence (exploding to NaN/Inf) with extreme learning rate (e.g. lr=100).

Related concepts:
    M04 (loss criteria), M05 (optimizers & four-step loop), X03 (matplotlib savefig headless)

Hub position:
    Hub (3) nn.Module & Optimization. Pairs the loss objective with parameter updates to close
    the learning loop driven by data.

Constraints:
    - Headless: no cv2.imshow / waitKey / createTrackbar / setMouseCallback
    - Save figure to docs/figs/ using matplotlib savefig
    - Max 3 new concepts
    - Pure functions called from main()
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "docs" / "figs"


def generate_linear_data(n_samples: int = 200, noise_std: float = 0.05) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate 1D regression points: y = 2x + 1 + N(0, noise_std)."""
    torch.manual_seed(20260919)
    x = torch.linspace(-2.0, 2.0, n_samples).unsqueeze(1)  # [N, 1]
    noise = torch.randn_like(x) * noise_std
    y = 2.0 * x + 1.0 + noise
    return x, y


def train_single_run(
    optimizer_name: str,
    x: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 100,
    lr: float = 0.05,
) -> tuple[list[float], tuple[float, float]]:
    """1 & 3. Train a linear model using the 4-step loop and return loss history and (weight, bias).

    TODO(Student):
        1. Initialize model = nn.Linear(1, 1).
        2. Set criterion = nn.MSELoss().
        3. Set optimizer (SGD or Adam with lr=lr).
        4. For each epoch:
           - Step 1: optimizer.zero_grad()
           - Step 2: pred = model(x)
           - Step 3: loss = criterion(pred, y)
           - Step 4: loss.backward()
           - Step 5: optimizer.step()
           - Record loss.item() into loss_history list.
        5. Extract learned weight and bias scalars.
    """
    # --- [TODO: Student Implementation Start] ---
    raise NotImplementedError("Step 1 & 3: Implement 4-step training loop with MSELoss and optimizer")
    # --- [TODO: Student Implementation End] ---

    assert len(loss_history) == epochs
    return loss_history, (w, b)


def save_convergence_comparison(
    sgd_losses: list[float], adam_losses: list[float], save_path: Path
) -> None:
    """2. Plot convergence curve of SGD vs Adam and save via matplotlib savefig."""
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(sgd_losses, label="SGD (lr=0.05)", color="tab:blue", lw=2)
    plt.plot(adam_losses, label="Adam (lr=0.05)", color="tab:orange", lw=2)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Day 09: SGD vs Adam Convergence on y = 2x + 1")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()

    assert save_path.exists(), f"Failed to save convergence plot to {save_path}"


def demonstrate_divergence_boundary() -> float:
    """4. Demonstrate boundary condition: exploding loss with excessive learning rate.

    TODO(Student):
        Run a 5-step training loop with SGD and lr=100.0.
        Verify that final loss either exceeds 1e6 or becomes NaN / Inf.
        Return the final loss value.
    """
    x, y = generate_linear_data(n_samples=50)

    # --- [TODO: Student Implementation Start] ---
    raise NotImplementedError("Step 4: Demonstrate loss explosion with extreme lr=100.0")
    # --- [TODO: Student Implementation End] ---

    assert np.isnan(exploded_loss) or np.isinf(exploded_loss) or exploded_loss > 1e6
    return exploded_loss


def main() -> None:
    """Entry point for Day 09."""
    print("=== Day 09: Loss Functions, Optimizers & Convergence ===")

    x, y = generate_linear_data(200)
    print(f"  [Setup] Generated synthetic dataset: x={x.shape}, y={y.shape}")

    # Step 1: SGD Training
    sgd_losses, (w_sgd, b_sgd) = train_single_run("sgd", x, y, epochs=100, lr=0.05)
    print(f"  [Step 1] SGD result: w={w_sgd:.4f} (target 2.0), b={b_sgd:.4f} (target 1.0), final loss={sgd_losses[-1]:.6f}")

    # Step 2: Adam Training
    adam_losses, (w_adam, b_adam) = train_single_run("adam", x, y, epochs=100, lr=0.05)
    print(f"  [Step 2] Adam result: w={w_adam:.4f} (target 2.0), b={b_adam:.4f} (target 1.0), final loss={adam_losses[-1]:.6f}")

    # Save visualization
    fig_path = FIG_DIR / "day_09_sgd_vs_adam.png"
    save_convergence_comparison(sgd_losses, adam_losses, fig_path)
    print(f"  [Step 2] Convergence comparison plot saved to {fig_path}")

    # Step 4: Divergence boundary
    div_loss = demonstrate_divergence_boundary()
    print(f"  [Step 4] Divergence boundary verified: exploded loss = {div_loss}")

    print("\n>>> Day 09 Acceptance: ALL PASS")


if __name__ == "__main__":
    main()
