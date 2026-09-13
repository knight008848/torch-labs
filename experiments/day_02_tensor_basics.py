"""
Day 02 / 2026-09-12 / Tensor trinity: dtype, shape, and device
Runtime: <fill in after completion>

Goal:
    Master the three fundamental pillars of PyTorch Tensors (dtype, shape, device)
    and understand zero-copy memory sharing with NumPy arrays.

Acceptance:
    1. Create tensors of 5 distinct dtypes (float32, float64, int32, int64, uint8)
       and verify their .dtype and .element_size().
    2. Perform CPU -> CUDA -> CPU roundtrip and assert exact value preservation.
    3. Explain and calculate why images use uint8 (storage/bandwidth) while models use float32
       (continuous gradient dynamic range for backprop).
    4. Demonstrate torch.from_numpy zero-copy memory sharing: mutating NumPy array
       reflects in Tensor, and mutating Tensor reflects in NumPy array, contrasted with clone().
    5. Demonstrate boundary condition: uint8 overflow and underflow wrap-around.

Related concepts:
    T01 (dtype), T02 (shape), T03 (device), B01 (NumPy bridge / memory sharing)

Hub position:
    Hub (1) Tensor foundation. Everything downstream (autograd, nn.Module, Dataset)
    operates on Tensors defined by these three pillars.

Constraints:
    - Headless: no cv2.imshow / waitKey / createTrackbar / setMouseCallback
    - Max 3 new concepts
    - Pure functions called from main()
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "teleop_demo.hdf5"


def run_daily_quick_quiz() -> dict[str, bool]:
    """Daily 3-minute conceptual quick check (1 Review + 1 Trap/Prediction).

    Q1 (Review Day 1): In our environment setup, why does `import torch` succeed under
    the WSL `embodied_ai` conda environment but fail under `pydata` or `base`?
    - Option A: `pydata` environment has no PyTorch installed; torch with CUDA 12.x is exclusively in `embodied_ai`.
    - Option B: `pydata` has torch installed, but it only runs in GUI mode.

    Q2 (Day 2 Trap): When converting a NumPy array to a PyTorch tensor via `torch.from_numpy(arr)`:
    - Option A: A deep copy is created; modifying arr leaves the tensor unchanged.
    - Option B: Memory is zero-copy shared on CPU; mutating arr in-place mutates the tensor.

    TODO(Student):
        Assign user_answers['Q1'] and user_answers['Q2'] with 'A' or 'B'.
    """
    user_answers = {
        "Q1": "",  # Fill with "A" or "B"
        "Q2": "",  # Fill with "A" or "B"
    }

    # --- [TODO: Quick Quiz Start] ---
    raise NotImplementedError("Step 0: Answer the 2 quick check questions")
    # --- [TODO: Quick Quiz End] ---

    correct_answers = {"Q1": "A", "Q2": "B"}
    results = {k: user_answers[k].strip().upper() == correct_answers[k] for k in correct_answers}
    assert all(results.values()), f"Quick quiz check failed: {results}"
    return results


def create_five_dtypes() -> dict[str, torch.Tensor]:
    """1. Create tensors with 5 distinct dtypes.

    TODO(Student):
        Construct tensors with torch.float32, torch.float64, torch.int32,
        torch.int64, and torch.uint8.
        Hint: Use torch.tensor([...], dtype=torch.<type>)
    """
    # --- [TODO: Student Implementation Start] ---
    raise NotImplementedError("Step 1: Construct tensors for float32, float64, int32, int64, uint8")
    # --- [TODO: Student Implementation End] ---

    # Acceptance verification
    assert t_f32.dtype == torch.float32 and t_f32.element_size() == 4
    assert t_f64.dtype == torch.float64 and t_f64.element_size() == 8
    assert t_i32.dtype == torch.int32 and t_i32.element_size() == 4
    assert t_i64.dtype == torch.int64 and t_i64.element_size() == 8
    assert t_u8.dtype == torch.uint8 and t_u8.element_size() == 1

    return {
        "float32": t_f32,
        "float64": t_f64,
        "int32": t_i32,
        "int64": t_i64,
        "uint8": t_u8,
    }


def cpu_cuda_cpu_roundtrip(tensor_cpu: torch.Tensor) -> torch.Tensor:
    """2. Perform CPU -> CUDA -> CPU roundtrip and assert exact equality.

    TODO(Student):
        1. Check torch.cuda.is_available() (raise RuntimeError if unavailable).
        2. Move tensor_cpu to cuda:0.
        3. Move it back to cpu.
        Hint: tensor.to(torch.device("cuda:0")) or tensor.cuda(), then tensor.to("cpu")
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required per AGENTS.md hardware specification")

    # --- [TODO: Student Implementation Start] ---
    raise NotImplementedError("Step 2: Implement CPU -> CUDA:0 -> CPU roundtrip")
    # --- [TODO: Student Implementation End] ---

    assert tensor_gpu.device.type == "cuda"
    assert tensor_back_cpu.device.type == "cpu"
    assert torch.equal(tensor_cpu, tensor_back_cpu), "Values diverged during device transfer!"

    return tensor_back_cpu


def calculate_image_memory_footprint(
    batch_size: int = 64, channels: int = 3, height: int = 84, width: int = 84
) -> tuple[int, int]:
    """3. Calculate memory size (in bytes) of raw uint8 images vs model float32 inputs.

    TODO(Student):
        Calculate total bytes for [batch_size, channels, height, width] in uint8 (1 byte/element)
        and float32 (4 bytes/element).
        Return (uint8_bytes, float32_bytes).
    """
    num_elements = batch_size * channels * height * width

    # --- [TODO: Student Implementation Start] ---
    raise NotImplementedError("Step 3: Calculate uint8 and float32 byte counts")
    # --- [TODO: Student Implementation End] ---

    assert float32_bytes == uint8_bytes * 4, "Float32 must consume exactly 4x the memory of uint8"
    return uint8_bytes, float32_bytes


def demonstrate_numpy_memory_sharing() -> None:
    """4. Demonstrate zero-copy memory sharing with torch.from_numpy vs clone.

    TODO(Student):
        1. Create a NumPy array np_arr: np.array([10.0, 20.0, 30.0], dtype=np.float32).
        2. Create shared_tensor using torch.from_numpy(np_arr).
        3. Create cloned_tensor using shared_tensor.clone().
        4. Mutate np_arr[0] = 999.0, assert shared_tensor[0] reflects it.
        5. Mutate shared_tensor[1] = 888.0, assert np_arr[1] reflects it.
        6. Verify cloned_tensor remains unchanged.
    """
    np_arr = np.array([10.0, 20.0, 30.0], dtype=np.float32)

    # --- [TODO: Student Implementation Start] ---
    raise NotImplementedError("Step 4: Demonstrate zero-copy memory sharing and clone independence")
    # --- [TODO: Student Implementation End] ---

    assert np_arr.ctypes.data == shared_tensor.data_ptr(), "Pointers must match for zero-copy!"
    assert shared_tensor.data_ptr() != cloned_tensor.data_ptr(), "Clone must allocate independent memory!"
    assert shared_tensor[0].item() == 999.0, "Tensor should reflect NumPy mutation!"
    assert cloned_tensor[0].item() == 10.0, "Cloned tensor should remain unchanged!"
    assert np_arr[1] == 888.0, "NumPy array should reflect in-place Tensor mutation!"


def demonstrate_boundary_uint8_wrap() -> tuple[int, int]:
    """5. Demonstrate uint8 overflow and underflow wrap-around behavior.

    TODO(Student):
        In uint8, verify that 255 + 1 wraps to 0, and 0 - 1 wraps to 255.
        Hint: torch.tensor([255], dtype=torch.uint8) + torch.tensor([1], dtype=torch.uint8)
        Return (overflow_val, underflow_val).
    """
    # --- [TODO: Student Implementation Start] ---
    raise NotImplementedError("Step 5: Demonstrate uint8 overflow (255+1) and underflow (0-1)")
    # --- [TODO: Student Implementation End] ---

    assert overflow_val == 0, f"Expected 255 + 1 = 0 in uint8, got {overflow_val}"
    assert underflow_val == 255, f"Expected 0 - 1 = 255 in uint8, got {underflow_val}"

    return overflow_val, underflow_val


def load_real_hdf5_slice() -> tuple[torch.Tensor, torch.Tensor]:
    """Extra: Load a real frame and action from teleop_demo.hdf5 into PyTorch tensors.

    TODO(Student):
        Open DATA_PATH using h5py, read 'data/demo_0/obs/agentview_image'[0] and
        'data/demo_0/actions'[0], and wrap them as PyTorch tensors using torch.from_numpy.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing {DATA_PATH}. Run experiments/day_01_env_and_data.py first.")

    # --- [TODO: Student Implementation Start] ---
    raise NotImplementedError("Step 6: Read first frame image and action from teleop_demo.hdf5")
    # --- [TODO: Student Implementation End] ---

    assert img_tensor.dtype == torch.uint8 and img_tensor.shape == (84, 84, 3)
    assert action_tensor.dtype == torch.float64 and action_tensor.shape == (7,)

    return img_tensor, action_tensor


def main() -> None:
    """Entry point for Day 2."""
    print("=== Day 02: Tensor Trinity & NumPy Memory Bridge ===")

    # Step 0: Daily Quick Quiz (3-minute check)
    quiz_results = run_daily_quick_quiz()
    print(f"  [Step 0] Daily quick quiz passed: {quiz_results}")

    # Step 1: Five dtypes
    dtypes = create_five_dtypes()
    print(f"  [Step 1] Created dtypes: {list(dtypes.keys())}")

    # Step 2: Device roundtrip
    cpu_sample = torch.randn(64, 7, dtype=torch.float32)
    cpu_cuda_cpu_roundtrip(cpu_sample)
    print("  [Step 2] CPU -> CUDA:0 -> CPU roundtrip validated")

    # Step 3: uint8 vs float32
    u8_b, f32_b = calculate_image_memory_footprint()
    print(f"  [Step 3] Memory comparison: uint8={u8_b}B, float32={f32_b}B (ratio={f32_b/u8_b:.1f}x)")

    # Step 4: Memory sharing
    demonstrate_numpy_memory_sharing()
    print("  [Step 4] torch.from_numpy zero-copy pointer and bidirectional mutation verified")

    # Step 5: Boundary conditions
    ov, un = demonstrate_boundary_uint8_wrap()
    print(f"  [Step 5] uint8 wrap-around verified: 255+1={ov}, 0-1={un}")

    # Step 6: Real HDF5 integration
    img, act = load_real_hdf5_slice()
    print(f"  [Step 6] Loaded HDF5 frame: img={img.shape} {img.dtype}, act={act.shape} {act.dtype}")

    print("\n>>> Day 02 Acceptance: ALL PASS")


if __name__ == "__main__":
    main()
