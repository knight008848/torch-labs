"""
Day 15 / 2026-09-25 / HDF5 Structure Probe and Lazy Slicing Mechanics
Runtime: <fill in after completion>

Goal:
    Deeply inspect robomimic demonstration HDF5 hierarchy, cross-check dataset schemas
    against dataset_spec.md, verify single-frame lazy slicing without memory blowup,
    and headlessly save an observation frame as PNG to establish visual ground truth.

Acceptance:
    1. Recursively traverse HDF5 hierarchy using h5py visititems, printing group and dataset paths.
    2. Audit keys against dataset_spec.md schema: verify presence, shapes, and dtypes of:
       - obs/agentview_image [N, 84, 84, 3] uint8
       - obs/robot0_joint_pos [N, 7] float64 (or float32)
       - actions [N, 7] float64 (or float32)
    3. Prove lazy slicing mechanics: slice single frame obs/agentview_image[0] and demonstrate
       that RSS memory does not increase by the size of the full dataset.
    4. Export a raw observation frame as docs/figs/day_15_raw_frame.png via matplotlib savefig
       (HEADLESS: strictly no cv2.imshow) as the primary color-order empirical evidence.
    5. Handle boundary condition: trap KeyError when probing non-existent dataset paths.

Related concepts:
    P04 (HDF5 structure traversal, lazy slicing, metadata inspection)

Hub position:
    Hub (4) Dataset/DataLoader (Storage layout foundation for high-throughput robot learning).

Constraints:
    - Headless: no cv2.imshow / waitKey / createTrackbar / setMouseCallback
    - Figures saved to docs/figs/
    - Pure functions called from main()
    - Max 3 new concepts
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HDF5_PATH = ROOT / "data" / "raw" / "teleop_demo.hdf5"
FIG_DIR = ROOT / "docs" / "figs"


def probe_hdf5_hierarchy(h5_file: h5py.File) -> list[str]:
    """1. Traverse full HDF5 group/dataset hierarchy with visititems.

    TODO(Student):
        1. Collect all paths into a list of strings.
        2. Print or record dataset shapes and dtypes:
           def visitor(name: str, obj: Any) -> None:
               ...
           h5_file.visititems(visitor)
        3. Return the collected list of paths.
    """
    # --- [TODO: Student Implementation Start] ---
    raise NotImplementedError("Step 1: Traverse HDF5 hierarchy using visititems")
    # --- [TODO: Student Implementation End] ---

    assert len(all_paths) > 0, "No datasets discovered in HDF5 archive"
    return all_paths


def audit_dataset_spec_schema(h5_file: h5py.File, demo_key: str = "data/demo_0") -> dict[str, Any]:
    """2. Cross-check demo_0 datasets against dataset_spec.md specification.

    TODO(Student):
        1. Access demo group: grp = h5_file[demo_key].
        2. Extract dataset descriptors:
           - img_ds = grp['obs/agentview_image']
           - state_ds = grp['obs/robot0_joint_pos']
           - action_ds = grp['actions']
        3. Check shapes and dtypes:
           assert img_ds.ndim == 4 and img_ds.shape[1:] == (84, 84, 3)
           assert state_ds.ndim == 2 and state_ds.shape[1] == 7
           assert action_ds.ndim == 2 and action_ds.shape[1] == 7
        4. Return dictionary summarizing schema metadata.
    """
    # --- [TODO: Student Implementation Start] ---
    raise NotImplementedError("Step 2: Audit demo_0 datasets against dataset_spec.md schema")
    # --- [TODO: Student Implementation End] ---

    return schema_summary


def verify_lazy_slicing_memory(h5_file: h5py.File, demo_key: str = "data/demo_0") -> int:
    """3. Verify single-frame slicing reads only requested frame without full load.

    TODO(Student):
        1. Locate img_ds = h5_file[f"{demo_key}/obs/agentview_image"].
        2. Calculate full dataset byte size: full_bytes = np.prod(img_ds.shape) * img_ds.dtype.itemsize.
        3. Read only single frame: frame = img_ds[0].
        4. Calculate frame byte size: frame_bytes = frame.nbytes.
        5. Assert frame_bytes < full_bytes, proving slice indexing avoids loading the entire array.
        6. Return frame_bytes.
    """
    # --- [TODO: Student Implementation Start] ---
    raise NotImplementedError("Step 3: Prove lazy slice memory footprint vs full dataset size")
    # --- [TODO: Student Implementation End] ---

    return frame_bytes


def export_sample_frame_png(h5_file: h5py.File, save_path: Path, demo_key: str = "data/demo_0") -> None:
    """4. Export frame 0 to PNG for color channel visual verification.

    TODO(Student):
        1. Slice frame_0 = h5_file[f"{demo_key}/obs/agentview_image"][0].
        2. Create matplotlib figure and render image with plt.imshow(frame_0).
        3. Save to save_path via plt.savefig(save_path, bbox_inches='tight').
        4. plt.close().
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # --- [TODO: Student Implementation Start] ---
    raise NotImplementedError("Step 4: Save raw frame to PNG headlessly for visual channel inspection")
    # --- [TODO: Student Implementation End] ---

    assert save_path.exists(), f"Failed to save frame PNG to {save_path}"


def demonstrate_missing_key_boundary(h5_file: h5py.File) -> bool:
    """5. Demonstrate boundary condition: querying invalid HDF5 dataset path raises KeyError."""
    # --- [TODO: Student Implementation Start] ---
    # Attempt h5_file["data/non_existent_group/bad_key"]
    # Catch KeyError and return True if caught.
    raise NotImplementedError("Step 5: Catch KeyError on invalid HDF5 path")
    # --- [TODO: Student Implementation End] ---


def main() -> None:
    """Entry point for Day 15."""
    print("=== Day 15: HDF5 Structure Probe & Lazy Slicing ===")

    if not HDF5_PATH.exists():
        print(f"  [Warning] {HDF5_PATH} does not exist yet. Please run Day 1 mock generator or download real data.")
        return

    with h5py.File(HDF5_PATH, "r") as h5:
        # Step 1: Traverse hierarchy
        paths = probe_hdf5_hierarchy(h5)
        print(f"  [Step 1] Traversed HDF5 archive: discovered {len(paths)} entries")

        # Step 2: Audit against dataset_spec.md
        schema = audit_dataset_spec_schema(h5, demo_key="data/demo_0")
        print(f"  [Step 2] Audited schema: {schema}")

        # Step 3: Lazy slice memory verification
        sliced_bytes = verify_lazy_slicing_memory(h5, demo_key="data/demo_0")
        print(f"  [Step 3] Single frame slice size: {sliced_bytes} bytes (lazy read verified)")

        # Step 4: Export frame PNG
        png_path = FIG_DIR / "day_15_raw_frame.png"
        export_sample_frame_png(h5, png_path, demo_key="data/demo_0")
        print(f"  [Step 4] Exported visual probe frame to {png_path}")

        # Step 5: Boundary demonstration
        key_err_ok = demonstrate_missing_key_boundary(h5)
        print(f"  [Step 5] KeyError boundary caught: {key_err_ok}")

    print("\n>>> Day 15 Acceptance: ALL PASS")


if __name__ == "__main__":
    main()
