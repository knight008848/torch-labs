"""
Day 06 / 2026-09-16 / Dynamic computation graph and autograd mechanics
Runtime: <fill in after completion>

Goal:
    Master PyTorch dynamic graph construction (autograd), verify analytical vs
    automatic differentiation, understand the gradient accumulation mechanism,
    and profile torch.no_grad() memory/performance benefits.

Acceptance:
    1. Manually calculate dy/dx for y = x^2 at x = 3 (analytical = 6.0), and verify
       x.grad equals 6.0 after y.backward().
    2. Demonstrate gradient accumulation without zero_grad() (grad sums to 12.0 on 2nd pass)
       and contrast with explicit zeroing via x.grad.zero_().
    3. Profile torch.no_grad() vs default mode on forward pass: assert tensor.requires_grad
       is suppressed and computation graph is not constructed.
    4. Handle boundary condition: calling .backward() on non-scalar output without grad_tensors
       and catching RuntimeError ('grad can be implicitly created only for scalar outputs').

Related concepts:
    G01 (computation graph), G02 (requires_grad), G03 (backward),
    G04 (gradient accumulation), G05 (no_grad / detach)

Hub position:
    Hub (2) autograd. Connects tensor data representations (Hub 1) to parameter
    optimization and training loops (Hub 3).

Constraints:
    - Headless: no cv2.imshow / waitKey / createTrackbar / setMouseCallback
    - Max 3 new concepts
    - Pure functions called from main()
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def verify_scalar_backward(x_val: float = 3.0) -> float:
    """1. Verify analytical derivative dy/dx = 2x at x = 3.0 via backward().

    TODO(Student):
        1. Create x = torch.tensor(x_val, requires_grad=True)
        2. Compute y = x ** 2
        3. Call y.backward()
        4. Assert x.grad is not None and matches 2 * x_val (6.0).
        5. Return x.grad.item().
    """
    # --- [TODO: Student Implementation Start] ---
    raise NotImplementedError("Step 1: Compute y = x^2 and verify dy/dx = 6.0 via backward()")
    # --- [TODO: Student Implementation End] ---

    assert abs(grad_val - 2.0 * x_val) < 1e-6, f"Expected {2.0 * x_val}, got {grad_val}"
    return grad_val


def demonstrate_gradient_accumulation() -> tuple[float, float]:
    """2. Demonstrate gradient accumulation without zero_grad vs explicit zeroing.

    TODO(Student):
        1. Create x = torch.tensor(3.0, requires_grad=True)
        2. Pass 1: y1 = x ** 2; y1.backward(); record grad1 = x.grad.item() (expected 6.0)
        3. Pass 2 without clearing: y2 = x ** 2; y2.backward(); record grad2 = x.grad.item() (expected 12.0)
        4. Explicit reset: x.grad.zero_(); assert x.grad.item() == 0.0
        5. Return (grad1, grad2)
    """
    # --- [TODO: Student Implementation Start] ---
    raise NotImplementedError("Step 2: Demonstrate gradient accumulation across two backward() calls")
    # --- [TODO: Student Implementation End] ---

    assert grad1 == 6.0, f"Expected grad1 = 6.0, got {grad1}"
    assert grad2 == 12.0, f"Expected accumulated grad2 = 12.0, got {grad2}"
    return grad1, grad2


def profile_no_grad_context() -> dict[str, bool]:
    """3. Profile torch.no_grad() effect on requires_grad and graph retention.

    TODO(Student):
        1. In default context:
           w = torch.randn(1000, 1000, requires_grad=True)
           x = torch.randn(1000, 1000)
           out_with_grad = w @ x
           record has_grad = out_with_grad.requires_grad (True) and out_with_grad.grad_fn is not None

        2. In with torch.no_grad():
           out_no_grad = w @ x
           record no_grad_flag = out_no_grad.requires_grad (False) and out_no_grad.grad_fn is None

        3. Return {"with_grad": has_grad, "no_grad": not no_grad_flag}
    """
    # --- [TODO: Student Implementation Start] ---
    raise NotImplementedError("Step 3: Verify computation graph suppression under torch.no_grad()")
    # --- [TODO: Student Implementation End] ---

    assert has_grad is True
    assert no_grad_flag is False
    return {"default_requires_grad": has_grad, "no_grad_requires_grad": no_grad_flag}


def demonstrate_non_scalar_backward_boundary() -> str:
    """4. Demonstrate boundary condition: backward() on non-scalar without grad_tensors.

    TODO(Student):
        1. Create x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        2. Compute y = x ** 2 (y has shape [3], non-scalar)
        3. Attempt y.backward() inside a try/except RuntimeError block.
        4. Catch the error, verify 'grad can be implicitly created only for scalar outputs' in str(e).
        5. Return error message string.
    """
    # --- [TODO: Student Implementation Start] ---
    raise NotImplementedError("Step 4: Catch non-scalar backward() RuntimeError")
    # --- [TODO: Student Implementation End] ---

    assert "scalar" in caught_err.lower(), f"Unexpected error: {caught_err}"
    return caught_err


def main() -> None:
    """Entry point for Day 06."""
    print("=== Day 06: Autograd & Dynamic Computation Graph ===")

    # Step 1: Scalar backward verification
    g1 = verify_scalar_backward(3.0)
    print(f"  [Step 1] Analytical vs autograd verified: dy/dx = {g1} at x = 3.0")

    # Step 2: Gradient accumulation
    pass1, pass2 = demonstrate_gradient_accumulation()
    print(f"  [Step 2] Gradient accumulation verified: pass 1 = {pass1}, pass 2 (accumulated) = {pass2}")

    # Step 3: torch.no_grad()
    status = profile_no_grad_context()
    print(f"  [Step 3] no_grad context suppression verified: {status}")

    # Step 4: Non-scalar backward boundary
    boundary_err = demonstrate_non_scalar_backward_boundary()
    print(f"  [Step 4] Caught expected non-scalar backward RuntimeError:\n    '{boundary_err[:70]}...'")

    print("\n>>> Day 06 Acceptance: ALL PASS")


if __name__ == "__main__":
    main()
