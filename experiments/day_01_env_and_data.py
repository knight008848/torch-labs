"""
Day 01 / 2026-09-11 / Build the project skeleton and land the dataset
Runtime: 0.5 h

Goal:
    Produce data/raw/teleop_demo.hdf5 -- a synthetic stand-in whose keys, dtypes
    and shapes match docs/dataset_spec.md section 3 exactly -- so that every
    Tensor exercise from Day 2 onward runs against real HDF5 data instead of
    ad-hoc arrays. Swapping in the real robomimic file on Day 16 must not force
    a rewrite of any slicing logic.

Acceptance:
    1. src/env_check.py exits 0: required packages present AND CUDA usable
    2. Project skeleton exists (data/raw, data/processed, experiments, ...)
    3. data/raw/teleop_demo.hdf5 exists and h5py can open it
    4. File is >= 100 MiB and every demo satisfies the section-3 spec table,
       including the top-level `data/` group real robomimic files use
    5. A sample frame is saved to docs/figs/ for eyeball verification
    6. The verifier itself is proven to reject malformed files (see selftest)

Related concepts:
    X04 (pathlib) -- today's only new number.
    P04 (HDF5 lazy loading) previewed but NOT introduced; lands Day 15.
    T02 (dtype) previewed but NOT introduced; lands Day 2.

Hub position:
    Day 1 introduces no hub. It lays the ground both hub (1) Tensor (Day 2-5)
    and hub (4) Dataset/DataLoader (Day 11+) will stand on.

Constraints:
    - Headless: no cv2.imshow / waitKey / createTrackbar / setMouseCallback
    - Use matplotlib savefig for any visualization
    - Max 3 new concepts (today: 1 -- pathlib)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

import h5py
import matplotlib
import numpy as np

# Must precede pyplot: this WSL env has no display, so force the file backend.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# experiments/ is not a package; make the sibling src/ importable for reuse.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from env_check import ensure_dirs  # noqa: E402

DATA_RAW = ROOT / "data" / "raw"
DATASET_PATH = DATA_RAW / "teleop_demo.hdf5"
FIG_PATH = ROOT / "docs" / "figs" / "day_01_mock_frame.png"

# Episode lengths, in frames. Deliberately unequal: the real dataset has demos
# of differing length, and a "by demo group, then concatenate" traversal only
# gets exercised if the lengths actually differ.
DEMO_LENGTHS = (900, 1000, 1100, 1200, 800, 700)
TOTAL_FRAMES = sum(DEMO_LENGTHS)

IMAGE_SHAPE = (84, 84, 3)  # HWC, per dataset_spec.md section 3
N_JOINTS = 7
SEED = 20260911

# Target from 00_LEARNING_PLAN.md Day 1 acceptance (4). Asserted in MiB, which
# is the stricter reading of "100MB".
MIN_BYTES = 100 * 1024 * 1024

# Real robomimic files nest every episode under a top-level `data` group, and
# dataset_spec.md section 3 spells the paths out as `data/demo_0/obs/...`.
# Getting this wrong is invisible on the mock and fatal on Day 16, so the
# generator and the verifier both build paths from this constant.
DATASET_ROOT = "data"
DEMO_PREFIX = "demo_"

# Spec table: path below a demo group -> (dtype, per-frame shape). Single source
# of truth for both the writer and the verifier, so they cannot drift apart.
SPEC: dict[str, tuple[type, tuple[int, ...]]] = {
    "obs/agentview_image": (np.uint8, IMAGE_SHAPE),
    "obs/robot0_joint_pos": (np.float64, (N_JOINTS,)),
    "actions": (np.float64, (N_JOINTS,)),
}


# --------------------------------------------------------------------------
# Synthesis
# --------------------------------------------------------------------------


def synth_frame(rng: np.random.Generator, step: int, n_frames: int) -> np.ndarray:
    """Render one synthetic RGB frame as uint8 [H, W, 3].

    The scene is a blue backdrop with a red cube sliding left to right. The
    red/blue split is on purpose: a BGR<->RGB mix-up turns this image orange on
    a blue backdrop, which is unmistakable in the Day 13 side-by-side figure.

    A fine dither is added to the background so the stored bytes are not one
    huge run of identical values. Flat data would compress and cache
    unrealistically well and would flatter the Day 26-27 throughput numbers.
    """
    height, width, _ = IMAGE_SHAPE
    frame = np.zeros(IMAGE_SHAPE, dtype=np.uint8)

    # Background: channel 0 = red (weak), channel 1 = green (vertical ramp),
    # channel 2 = blue (strong).
    frame[..., 0] = 30
    frame[..., 1] = np.linspace(60, 140, height, dtype=np.uint8)[:, None]
    frame[..., 2] = 200

    dither = rng.integers(-8, 9, size=(height, width, 1), dtype=np.int16)
    frame[..., 1] = np.clip(frame[..., 1].astype(np.int16) + dither[..., 0], 0, 255).astype(np.uint8)

    # Cube: solid red, travelling horizontally over the episode.
    progress = step / max(n_frames - 1, 1)
    cube_half = 6
    centre_x = int(round(cube_half + 2 + progress * (width - 2 * cube_half - 4)))
    centre_y = int(height * 0.62)
    frame[
        centre_y - cube_half : centre_y + cube_half,
        centre_x - cube_half : centre_x + cube_half,
    ] = (220, 40, 40)

    # Gripper: a light bar riding above the cube, tracking it horizontally.
    frame[4:10, centre_x - 2 : centre_x + 2] = (235, 235, 235)

    return frame


def synth_state(rng: np.random.Generator, n_frames: int) -> np.ndarray:
    """Build a smooth float64 [N, 7] joint trajectory.

    Each joint is a sinusoid with its own amplitude, frequency and phase, plus
    a whisper of noise. Smooth and per-joint distinct, so a slicing or
    broadcasting bug is visible as a curve that jumps or flattens.
    """
    steps = np.arange(n_frames, dtype=np.float64)
    amplitude = rng.uniform(0.3, 0.8, size=N_JOINTS)
    frequency = rng.uniform(0.010, 0.040, size=N_JOINTS)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=N_JOINTS)

    trajectory = amplitude * np.sin(2.0 * np.pi * frequency * steps[:, None] + phase)
    trajectory += rng.normal(0.0, 0.002, size=trajectory.shape)
    return trajectory.astype(np.float64)


def synth_actions(rng: np.random.Generator, state: np.ndarray) -> np.ndarray:
    """Derive float64 [N, 7] actions from the state trajectory.

    Real robomimic actions are per-step deltas of the controller target, so the
    mock mirrors that: action[t] ~= state[t+1] - state[t]. Keeping the two
    physically consistent means a misalignment between state and action shows up
    as a failed assert on Day 22, rather than as a silently wrong training pair.

    The final step has no successor, so its delta is defined as zero.
    """
    deltas = np.diff(state, axis=0, append=state[-1:])
    return (deltas + rng.normal(0.0, 0.0005, size=deltas.shape)).astype(np.float64)


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------


def write_demo(
    handle: h5py.File,
    demo_name: str,
    n_frames: int,
    rng: np.random.Generator,
    block: int = 200,
) -> None:
    """Write one demo group, generating frames in fixed-size blocks.

    Frames are synthesised *inside* the block loop rather than passed in whole,
    so peak RAM stays at `block` frames (~4 MiB) instead of a full episode. The
    rng is consumed in a fixed order (state, then actions, then frames in
    sequence), so output is byte-identical for a given seed.

    Chunks are one frame deep so that fetching a single frame touches exactly
    one chunk -- the property Day 15 measures as lazy loading.
    """
    group = handle.create_group(f"{DATASET_ROOT}/{demo_name}")

    state = synth_state(rng, n_frames)
    actions = synth_actions(rng, state)
    group.create_dataset("obs/robot0_joint_pos", data=state, dtype=np.float64)
    group.create_dataset("actions", data=actions, dtype=np.float64)

    image_ds = group.create_dataset(
        "obs/agentview_image",
        shape=(n_frames, *IMAGE_SHAPE),
        dtype=np.uint8,
        chunks=(1, *IMAGE_SHAPE),
    )
    for start in range(0, n_frames, block):
        stop = min(start + block, n_frames)
        image_ds[start:stop] = np.stack(
            [synth_frame(rng, step, n_frames) for step in range(start, stop)]
        )

    group.attrs["num_samples"] = n_frames
    # Cheap self-documentation: nobody mistakes the stand-in for the real thing,
    # and load_or_create reads this flag back before overwriting anything.
    group.attrs["source"] = "synthetic"


def create_dataset(path: Path, lengths: tuple[int, ...], seed: int) -> None:
    """Generate the full synthetic HDF5 file at `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    # Write to a temporary name first: an interrupted run then leaves a `.part`
    # file rather than a half-written hdf5 that later days would happily open.
    tmp_path = path.with_name(path.name + ".part")

    with h5py.File(tmp_path, "w") as handle:
        # File-level attrs record provenance. They are metadata, not data, so
        # they do not affect the section-3 structure table.
        handle.attrs["synthetic"] = True
        handle.attrs["generator"] = "experiments/day_01_env_and_data.py"
        handle.attrs["seed"] = seed
        # The mock is RGB by construction. This is NOT evidence about the real
        # robomimic file -- that channel order stays undecided until Day 16.
        handle.attrs["channel_order"] = "RGB"
        handle.attrs["note"] = "stand-in for robomimic lift/ph image.hdf5"

        for index, n_frames in enumerate(lengths):
            demo_name = f"{DEMO_PREFIX}{index}"
            print(f"  {demo_name}: {n_frames} frames", flush=True)
            write_demo(handle, demo_name, n_frames, rng)

    # Atomic on the same filesystem, so a reader never sees a partial file.
    tmp_path.replace(path)


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------


def demo_index(name: str) -> int:
    """Numeric index of a `demo_N` group.

    Plain `sorted()` is lexicographic, which orders demo_10 before demo_2. The
    mock only has 6 demos so it would not bite today, but the real file has
    hundreds -- and Day 12 groups by demo before concatenating.
    """
    if not name.startswith(DEMO_PREFIX):
        raise ValueError(f"unexpected group {name!r}, expected {DEMO_PREFIX}<int>")
    suffix = name[len(DEMO_PREFIX) :]
    if not suffix.isdigit():
        raise ValueError(f"unexpected group {name!r}: {suffix!r} is not an integer")
    return int(suffix)


def list_demos(handle: h5py.File) -> list[str]:
    """Demo group names under the dataset root, in natural numeric order."""
    if DATASET_ROOT not in handle:
        raise KeyError(f"top-level group {DATASET_ROOT!r} is missing")
    names = list(handle[DATASET_ROOT].keys())
    if not names:
        raise ValueError(f"{DATASET_ROOT!r} contains no demo groups")
    ordered = sorted(names, key=demo_index)
    indices = [demo_index(name) for name in ordered]
    if len(set(indices)) != len(indices):
        raise ValueError(f"duplicate demo indices under {DATASET_ROOT!r}")
    return ordered


def verify_handle(
    handle: h5py.File, expected: dict[str, int] | None = None
) -> dict[str, object]:
    """Check an open HDF5 handle against the spec table. Raises on mismatch.

    Split out from verify_dataset so the selftest can feed it malformed
    in-memory files without touching the disk.
    """
    stats: dict[str, object] = {"demos": 0, "frames": 0, "image_bytes": 0}
    demo_names = list_demos(handle)

    for demo_name in demo_names:
        group = handle[f"{DATASET_ROOT}/{demo_name}"]
        lengths: set[int] = set()

        for key, (dtype, trailing_shape) in SPEC.items():
            if key not in group:
                raise KeyError(f"{demo_name}/{key} is missing")
            dataset = group[key]

            if dataset.dtype != dtype:
                raise TypeError(f"{demo_name}/{key}: dtype {dataset.dtype}, expected {dtype}")
            # First axis is N and varies per demo; the rest must match exactly.
            if tuple(dataset.shape[1:]) != trailing_shape:
                raise ValueError(
                    f"{demo_name}/{key}: trailing shape {dataset.shape[1:]}, "
                    f"expected {trailing_shape}"
                )
            if dataset.shape[0] == 0:
                raise ValueError(f"{demo_name}/{key} is empty")
            lengths.add(dataset.shape[0])

        # The most invisible bug in this whole pipeline: a frame count that
        # disagrees between image, state and action.
        if len(lengths) != 1:
            raise ValueError(f"{demo_name}: N disagrees across keys -> {sorted(lengths)}")

        n_frames = lengths.pop()
        if expected is not None and demo_name in expected and expected[demo_name] != n_frames:
            raise ValueError(
                f"{demo_name}: N={n_frames}, expected {expected[demo_name]}"
            )

        stats["frames"] = int(stats["frames"]) + n_frames
        stats["image_bytes"] = int(stats["image_bytes"]) + int(
            group["obs/agentview_image"].nbytes
        )

    if expected is not None:
        missing = set(expected) - set(demo_names)
        if missing:
            raise ValueError(f"expected demos absent from file: {sorted(missing)}")

    stats["demos"] = len(demo_names)
    stats["demo_names"] = demo_names
    stats["channel_order"] = str(handle.attrs.get("channel_order", "unknown"))
    stats["synthetic"] = bool(handle.attrs.get("synthetic", False))
    return stats


def verify_dataset(path: Path, expected: dict[str, int] | None = None) -> dict[str, object]:
    """Open `path`, verify it against the spec table, and report file stats."""
    with h5py.File(path, "r") as handle:
        stats = verify_handle(handle, expected)

    size = path.stat().st_size
    stats["size_bytes"] = size
    stats["size_mib"] = size / (1024 * 1024)

    if size < MIN_BYTES:
        raise ValueError(
            f"{path} is {size / (1024 * 1024):.1f} MiB, "
            f"below the {MIN_BYTES / (1024 * 1024):.0f} MiB floor"
        )
    return stats


def is_synthetic(path: Path) -> bool:
    """Read the provenance flag. Raises OSError if the file cannot be opened."""
    with h5py.File(path, "r") as handle:
        return bool(handle.attrs.get("synthetic", False))


# --------------------------------------------------------------------------
# Self-test: prove the verifier actually rejects things
# --------------------------------------------------------------------------


def _core_file(build: Callable[[h5py.File], None]) -> h5py.File:
    """Build a small in-memory HDF5 file (no disk I/O) for negative testing."""
    handle = h5py.File("selftest.h5", "w", driver="core", backing_store=False)
    build(handle)
    return handle


def _good_demo(handle: h5py.File, name: str, n: int = 4) -> h5py.Group:
    """Write a schema-correct demo, so each negative case can break exactly one thing."""
    group = handle.create_group(f"{DATASET_ROOT}/{name}")
    group.create_dataset("obs/agentview_image", data=np.zeros((n, *IMAGE_SHAPE), np.uint8))
    group.create_dataset("obs/robot0_joint_pos", data=np.zeros((n, N_JOINTS), np.float64))
    group.create_dataset("actions", data=np.zeros((n, N_JOINTS), np.float64))
    return group


def selftest() -> list[str]:
    """Confirm verify_handle rejects every malformed layout. Returns caught cases.

    A verifier that has only ever been shown good input proves nothing. Each
    case below breaks exactly one rule and must raise; if one slips through,
    the assert fires loud.
    """
    caught: list[str] = []

    def expect_reject(name: str, build: Callable[[h5py.File], None], expected=None) -> None:
        with _core_file(build) as handle:
            try:
                verify_handle(handle, expected)
            except (KeyError, TypeError, ValueError):
                caught.append(name)
            else:
                raise AssertionError(f"verifier accepted a malformed file: {name}")

    def missing_key(handle):
        group = _good_demo(handle, "demo_0")
        del group["actions"]

    def wrong_dtype(handle):
        group = _good_demo(handle, "demo_0")
        del group["actions"]
        group.create_dataset("actions", data=np.zeros((4, N_JOINTS), np.float32))

    def wrong_trailing_shape(handle):
        group = _good_demo(handle, "demo_0")
        del group["obs/agentview_image"]
        group.create_dataset("obs/agentview_image", data=np.zeros((4, 84, 84), np.uint8))

    def n_disagrees(handle):
        group = _good_demo(handle, "demo_0")
        del group["actions"]
        group.create_dataset("actions", data=np.zeros((3, N_JOINTS), np.float64))

    def bad_demo_name(handle):
        _good_demo(handle, "episode_0")

    def empty_demo(handle):
        _good_demo(handle, "demo_0", n=0)

    def no_data_group(handle):
        group = handle.create_group("demo_0")
        group.create_dataset("actions", data=np.zeros((4, N_JOINTS), np.float64))

    expect_reject("missing key", missing_key)
    expect_reject("wrong dtype", wrong_dtype)
    expect_reject("wrong trailing shape", wrong_trailing_shape)
    expect_reject("N disagrees across keys", n_disagrees)
    expect_reject("bad demo name", bad_demo_name)
    expect_reject("empty demo", empty_demo)
    expect_reject("no data group", no_data_group)
    expect_reject("per-demo length mismatch", lambda h: _good_demo(h, "demo_0"), {"demo_0": 99})
    expect_reject("expected demo absent", lambda h: _good_demo(h, "demo_1"), {"demo_0": 4})

    return caught


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def save_sample_frame(path: Path, figure_path: Path, demo: str = "demo_0", step: int = 0) -> None:
    """Write one frame to PNG so the scene can be checked by eye later.

    Day 15 repeats this against the real file; that comparison is the first
    real evidence about channel order.
    """
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "r") as handle:
        frame = handle[f"{DATASET_ROOT}/{demo}"]["obs/agentview_image"][step]

    figure, axes = plt.subplots(1, 2, figsize=(7, 3.6))
    axes[0].imshow(frame)
    axes[0].set_title(f"{demo} frame {step}\nas stored")
    axes[1].imshow(frame[..., ::-1])
    axes[1].set_title("channel-reversed\n(BGR view)")
    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle("Day 01 synthetic frame -- RGB by construction")
    figure.tight_layout()
    figure.savefig(figure_path, dpi=120)
    plt.close(figure)


def load_or_create(path: Path, force: bool, expected: dict[str, int]) -> bool:
    """Ensure `path` holds a valid synthetic dataset. True if it was generated.

    Refuses to touch a file that does not carry the `synthetic` flag: on Day 16
    the real 1.5GB robomimic file lands at this exact path, and a regenerate
    triggered by a schema mismatch would otherwise destroy it silently.
    """
    if not path.exists():
        print(f"{path} not found; generating synthetic stand-in")
        print(f"generating {sum(expected.values())} frames across {len(expected)} demos...")
        create_dataset(path, tuple(expected.values()), SEED)
        return True

    try:
        synthetic = is_synthetic(path)
    except OSError as exc:
        # Unreadable (truncated, partial copy). Preserve it under a new name
        # rather than deleting: it may be someone's only copy of real data.
        corrupt_path = path.with_name(path.name + ".corrupt")
        print(f"existing file is unreadable ({exc});")
        print(f"  moving it to {corrupt_path.name} and regenerating")
        path.replace(corrupt_path)
        create_dataset(path, tuple(expected.values()), SEED)
        return True

    if not synthetic:
        raise SystemExit(
            f"\nREFUSING to overwrite {path}.\n"
            f"  It does not carry the `synthetic` flag, so it is probably the real\n"
            f"  robomimic dataset. Delete it yourself if you really mean to:\n"
            f"    rm {path}\n"
        )

    if force:
        print("--force: regenerating synthetic stand-in")
        create_dataset(path, tuple(expected.values()), SEED)
        return True

    try:
        verify_dataset(path, expected)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        print(f"existing synthetic file failed verification ({exc}); regenerating")
        create_dataset(path, tuple(expected.values()), SEED)
        return True

    print(f"reusing verified {path} (pass --force to regenerate)")
    return False


def main() -> None:
    """Entry point. Keeps each step a small pure function."""
    parser = argparse.ArgumentParser(
        description="Generate the Day 1 synthetic stand-in for the robomimic dataset."
    )
    parser.add_argument(
        "--force", action="store_true", help="regenerate a synthetic dataset in place"
    )
    args = parser.parse_args()

    expected = {f"{DEMO_PREFIX}{i}": n for i, n in enumerate(DEMO_LENGTHS)}

    # 1) Directory skeleton (acceptance 2). Idempotent, safe to call every day.
    created = ensure_dirs(ROOT)
    print(f"skeleton: {len(created)} path(s) created")

    # 2) Land the dataset (acceptance 3, 4). The `data/` nesting is checked by
    #    verify_dataset, which raises typed exceptions -- unlike assert, those
    #    survive a `python -O` run.
    generated = load_or_create(DATASET_PATH, args.force, expected)
    stats = verify_dataset(DATASET_PATH, expected)

    # 3) Prove the verifier rejects malformed input (acceptance 6).
    caught = selftest()
    print(f"selftest: verifier rejected {len(caught)} malformed fixtures")

    # 4) Visual sanity check (acceptance 5).
    save_sample_frame(DATASET_PATH, FIG_PATH)

    # 5) Acceptance summary. The hard checks already ran above; these are the
    #    readable restatement. Run with -O and only these disappear.
    assert DATASET_PATH.is_file(), "dataset missing after generation"
    assert len(caught) == 9, f"selftest covered {len(caught)}/9 cases"
    assert stats["demos"] == len(DEMO_LENGTHS), "demo count mismatch"
    assert stats["frames"] == TOTAL_FRAMES, "frame count mismatch"
    assert stats["size_bytes"] >= MIN_BYTES, "file below the 100 MiB floor"
    assert FIG_PATH.is_file(), "sample figure was not written"

    print()
    print(f"dataset : {DATASET_PATH.relative_to(ROOT)}")
    print(f"demos   : {stats['demos']}  frames: {stats['frames']}")
    print(f"size    : {stats['size_mib']:.1f} MiB ({stats['size_bytes']} bytes)")
    print(f"channel : {stats['channel_order']} (synthetic -- NOT evidence for Day 16)")
    print(f"figure  : {FIG_PATH.relative_to(ROOT)}")
    print(f"generated: {generated}")
    print()
    print("Day 01 acceptance: PASS")


if __name__ == "__main__":
    main()
