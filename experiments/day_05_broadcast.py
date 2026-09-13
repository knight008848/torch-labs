"""
Day 05 / 2026-09-15 / Broadcasting semantics and in-place mutation pitfalls
Runtime: <fill in after completion>

Goal:
    Master PyTorch broadcasting rules for vectorized tensor normalization across
    joint/action dimensions without Python loops, and deeply understand why in-place
    operations break autograd backward graph tracking.

Acceptance:
    1. Vectorized normalization of [N, 7] tensor using broadcasted mean [7] and std [7] (no loops).
    2. Intentionally trigger and catch the classic autograd in-place error:
       'RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation'.
    3. Contrast out-of-place (a = a + b) vs in-place (a += b, a.add_(b)) in autograd contexts.
    4. Demonstrate broadcasting boundary condition: incompatible dimensions (e.g. [N, 7] vs [8])
       failing with descriptive RuntimeError.

Related concepts:
    T06 (broadcasting rules), T07 (in-place mutation traps)

Hub position:
    Hub (1) Tensor. Broadcasting is the foundation of batch-wise preprocessing and feature scaling,
    while understanding in-place traps prevents subtle, silent corruptions in autograd computation.

Constraints:
    - Headless: no cv2.imshow / waitKey / createTrackbar / setMouseCallback
    - Max 3 new concepts
    - Pure functions called from main()
"""

from __future__ import annotations

from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def broadcast_normalize(tensor_n7: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """1. Vectorized feature normalization using broadcasting.

    TODO(Student):
        1. Compute mean across dimension 0: mean = tensor_n7.mean(dim=0)  # shape [7]
        2. Compute std across dimension 0: std = tensor_n7.std(dim=0)    # shape [7]
        3. Add epsilon (1e-6) to std to prevent division by zero.
        4. Broadcast: normalized = (tensor_n7 - mean) / (std + 1e-6)
        5. Return (normalized, mean, std). Strictly NO Python for-loops!
    """
    assert tensor_n7.ndim == 2 and tensor_n7.shape[1] == 7

    # --- [TODO: Student Implementation Start] ---
    raise NotImplementedError("Step 1: Broadcast normalization of [N, 7] with [7] statistics without loops")
    # --- [TODO: Student Implementation End] ---

    assert normalized.shape == tensor_n7.shape
    # Normalized tensor should have near-zero mean and unit std
    assert torch.allclose(normalized.mean(dim=0), torch.zeros(7), atol=1e-5)
    assert torch.allclose(normalized.std(dim=0), torch.ones(7), atol=1e-3)

    return normalized, mean, std


def trigger_autograd_inplace_error() -> str:
    """2. Intentionally trigger and catch the in-place autograd RuntimeError.

    TODO(Student):
        1. Create x = torch.tensor([2.0, 3.0], requires_grad=True)
        2. Create y = x ** 2
        3. Perform an in-place mutation on x or a tensor needed for gradient:
           e.g., x.add_(1.0) or x[0] += 1.0
        4. Attempt y.sum().backward() inside a try/except RuntimeError block.
        5. Catch the RuntimeError, verify 'modified by an inplace operation' is in str(e),
           and return str(e).
    """
    # --- [TODO: Student Implementation Start] ---
    raise NotImplementedError("Step 2: Trigger autograd in-place modification RuntimeError")
    # --- [TODO: Student Implementation End] ---

    assert "inplace operation" in caught_error.lower(), f"Unexpected error message: {caught_error}"
    return caught_error


def contrast_out_of_place_vs_inplace() -> None:
    """3. Contrast out-of-place vs in-place semantics with autograd.

    TODO(Student):
        1. Out-of-place branch:
           a = torch.tensor([5.0], requires_grad=True)
           b = a + 2.0  # new tensor allocated, graph intact
           c = b ** 2
           c.backward()
           assert a.grad is not None and a.grad.item() == 14.0  # 2 * (5 + 2)

        2. In-place branch (leaf variable mutation error):
           a_leaf = torch.tensor([5.0], requires_grad=True)
           In try/except, call a_leaf.add_(2.0), catching the leaf modification error:
           'a leaf Variable that requires grad is being used in an in-place operation'
    """
    # --- [TODO: Student Implementation Start] ---
    raise NotImplementedError("Step 3: Contrast out-of-place vs in-place autograd behavior")
    # --- [TODO: Student Implementation End] ---


def demonstrate_incompatible_broadcast_boundary() -> str:
    """4. Demonstrate boundary condition: broadcasting failure with incompatible shapes.

    TODO(Student):
        Attempt to add a tensor of shape [10, 7] to a tensor of shape [8].
        In a try/except RuntimeError block, catch the dimension mismatch error.
        Return the error message.
    """
    # --- [TODO: Student Implementation Start] ---
    raise NotImplementedError("Step 4: Catch incompatible broadcast RuntimeError")
    # --- [TODO: Student Implementation End] ---

    assert "match" in error_msg.lower() or "shape" in error_msg.lower()
    return error_msg


def main() -> None:
    """Entry point for Day 05."""
    print("=== Day 05: Broadcasting & In-Place Pitfalls ===")

    # Step 1: Broadcasting normalization
    raw_data = torch.randn(100, 7) * 5.0 + 10.0
    norm, mu, sigma = broadcast_normalize(raw_data)
    print(f"  [Step 1] Broadcast normalization verified on shape {norm.shape}")

    # Step 2: Intentional in-place failure
    err_msg = trigger_autograd_inplace_error()
    print(f"  [Step 2] Caught expected autograd in-place RuntimeError:\n    '{err_msg[:80]}...'")

    # Step 3: Out-of-place vs in-place
    contrast_out_of_place_vs_inplace()
    print("  [Step 3] Out-of-place grad computation vs leaf in-place guard validated")

    # Step 4: Incompatible broadcast boundary
    incompat_err = demonstrate_incompatible_broadcast_boundary()
    print(f"  [Step 4] Incompatible broadcast boundary caught: '{incompat_err[:70]}...'")

    print("\n>>> Day 05 Acceptance: ALL PASS")


if __name__ == "__main__":
    main()
