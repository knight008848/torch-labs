"""
Day 08 / 2026-09-18 / nn.Module architecture, parameter containers, and persistent buffers
Runtime: <fill in after completion>

Goal:
    Master Hub (3) nn.Module foundation: build a minimal MLP mapping robot joint
    states to actions, inspect state_dict parameter names and shapes, and strictly
    distinguish learnable parameters from non-learnable persistent buffers.

Acceptance:
    1. Define SimpleRobotMLP(nn.Module) taking [B, 7] joint states and outputting [B, 7] actions.
    2. Inspect and assert all keys and shapes in state_dict() (linear1.weight, linear1.bias, etc.).
    3. Distinguish parameters() vs buffers(): register a persistent buffer (running_mean) via
       register_buffer(), proving it is tracked in state_dict() but excluded from model.parameters().
    4. Handle boundary condition: demonstrate that omitting super().__init__() raises AttributeError
       when assigning submodules.

Related concepts:
    M01 (nn.Module & forward), M02 (parameters vs buffers), M03 (state_dict serialization)

Hub position:
    Hub (3) nn.Module. Serves as the consumer contract for the data pipeline: batches produced
    by DataLoader (Hub 4) feed directly into forward().

Constraints:
    - Headless: no cv2.imshow / waitKey / createTrackbar / setMouseCallback
    - Max 3 new concepts
    - Pure functions called from main()
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]


class SimpleRobotMLP(nn.Module):
    """Minimal MLP for joint state -> action prediction with a persistent buffer."""

    def __init__(self, in_features: int = 7, hidden_dim: int = 32, out_features: int = 7) -> None:
        """TODO(Student):
            1. Call super().__init__().
            2. Define self.linear1 = nn.Linear(in_features, hidden_dim).
            3. Define self.relu = nn.ReLU().
            4. Define self.linear2 = nn.Linear(hidden_dim, out_features).
            5. Register persistent buffer: self.register_buffer("running_mean", torch.zeros(in_features)).
        """
        # --- [TODO: Student Implementation Start] ---
        super().__init__()
        raise NotImplementedError("Step 1: Implement SimpleRobotMLP.__init__")
        # --- [TODO: Student Implementation End] ---

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """TODO(Student):
            Pass x through linear1 -> relu -> linear2 and return output [B, out_features].
        """
        # --- [TODO: Student Implementation Start] ---
        raise NotImplementedError("Step 1: Implement SimpleRobotMLP.forward")
        # --- [TODO: Student Implementation End] ---


def inspect_model_state_dict(model: nn.Module) -> dict[str, tuple[int, ...]]:
    """2. Inspect state_dict() keys and shapes.

    TODO(Student):
        Extract a dictionary mapping each parameter/buffer name to its shape tuple.
        E.g. {"linear1.weight": (32, 7), ...}
    """
    # --- [TODO: Student Implementation Start] ---
    raise NotImplementedError("Step 2: Extract key -> shape mapping from state_dict()")
    # --- [TODO: Student Implementation End] ---

    assert "linear1.weight" in shapes
    assert "linear2.weight" in shapes
    assert "running_mean" in shapes
    return shapes


def verify_parameters_vs_buffers(model: nn.Module) -> tuple[int, int]:
    """3. Strictly verify parameters() vs buffers().

    TODO(Student):
        1. Collect parameter names: param_names = [name for name, _ in model.named_parameters()]
        2. Collect buffer names: buffer_names = [name for name, _ in model.named_buffers()]
        3. Assert 'running_mean' is in buffer_names but NOT in param_names.
        4. Return (len(param_names), len(buffer_names)).
    """
    # --- [TODO: Student Implementation Start] ---
    raise NotImplementedError("Step 3: Separate and verify named_parameters vs named_buffers")
    # --- [TODO: Student Implementation End] ---

    assert "running_mean" in buffer_names
    assert "running_mean" not in param_names
    assert n_params >= 4  # linear1.weight, linear1.bias, linear2.weight, linear2.bias
    assert n_buffers >= 1  # running_mean

    return n_params, n_buffers


def demonstrate_missing_super_init_boundary() -> str:
    """4. Demonstrate boundary condition: forgetting super().__init__().

    TODO(Student):
        Define a faulty class inheriting from nn.Module that forgets super().__init__()
        and tries to assign self.layer = nn.Linear(7, 7).
        Catch the AttributeError ('cannot assign module before Module.__init__() call')
        and return the error string.
    """
    # --- [TODO: Student Implementation Start] ---
    raise NotImplementedError("Step 4: Demonstrate AttributeError when super().__init__() is omitted")
    # --- [TODO: Student Implementation End] ---

    assert "cannot assign module" in error_msg.lower() or "init" in error_msg.lower()
    return error_msg


def main() -> None:
    """Entry point for Day 08."""
    print("=== Day 08: nn.Module, Parameter Containers & Buffers ===")

    # Step 1: Model instantiation and forward
    model = SimpleRobotMLP(in_features=7, hidden_dim=32, out_features=7)
    dummy_input = torch.randn(4, 7)
    out = model(dummy_input)
    print(f"  [Step 1] Model forward pass verified: input={dummy_input.shape} -> output={out.shape}")

    # Step 2: Inspect state_dict
    shapes = inspect_model_state_dict(model)
    print(f"  [Step 2] state_dict entries ({len(shapes)}): {shapes}")

    # Step 3: Parameters vs buffers
    n_p, n_b = verify_parameters_vs_buffers(model)
    print(f"  [Step 3] Parameter/Buffer distinction: {n_p} learnable params, {n_b} persistent buffers")

    # Step 4: Missing super().__init__() boundary
    err_str = demonstrate_missing_super_init_boundary()
    print(f"  [Step 4] Caught missing super().__init__() AttributeError:\n    '{err_str[:70]}...'")

    print("\n>>> Day 08 Acceptance: ALL PASS")


if __name__ == "__main__":
    main()
