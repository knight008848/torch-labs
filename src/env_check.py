"""
Environment probe for the torch-labs data-pipeline track.

Day 01 / 2026-09-11 / Confirm the `embodied_ai` conda env can carry the pipeline.

Run:
    source ~/miniforge3/bin/activate embodied_ai
    python src/env_check.py

Exit status is 1 when any check fails -- a missing required package, a torch
build that is not CUDA 12.x, or an unusable GPU. Day 1's acceptance criterion
is "get torch.cuda.is_available() passing first", so an ungraded CUDA section
would make the criterion unfalsifiable. Failing loudly here beats silently
degrading on Day 15.

Related concepts: X04 (pathlib), T03 (device -- preview only)
Hub position: Day 1 introduces no hub. It only clears the ground that hub (1)
    Tensor and hub (4) Dataset/DataLoader will be built on.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# The pipeline cannot load a single frame without these. cv2 is listed here
# rather than as optional because Day 1 acceptance (1) installs it and the plan
# treats the environment as incomplete until it is present. It is not *used*
# until Day 13 (BGR/RGB comparison), but a missing cv2 on Day 1 is a real
# deviation from the day's acceptance criteria, so it should fail loudly.
#
# torchvision is here for a subtler reason. Its metadata pins torch to an exact
# version (`torch (==2.10.0)`), so the pair cannot drift on its own -- but a
# half-finished upgrade leaves torch new and torchvision old, and that surfaces
# as "operator torchvision::nms does not exist" at import time. A Day-13
# symptom for a Day-1 mistake, which is exactly what this file exists to
# prevent.
REQUIRED_PACKAGES = (
    "torch",
    "torchvision",
    "numpy",
    "h5py",
    "pandas",
    "matplotlib",
    "cv2",
)

# Packages that may legitimately be absent without blocking the pipeline.
OPTIONAL_PACKAGES: tuple[str, ...] = ()

# torch must be a CUDA build for the 8GB-VRAM work in days 25-27, and the plan
# pins CUDA 12.x. A CPU-only wheel would otherwise pass every check silently.
REQUIRED_CUDA_MAJOR = "12"

# Project skeleton. Empty directories do not survive a `git clone`, so the ones
# matched by .gitignore carry a .gitkeep to keep them alive.
REQUIRED_DIRS = (
    "data/raw",
    "data/processed",
    "experiments",
    "notebooks",
    "src",
    "models",
    "docs/figs",
)
GITKEEP_DIRS = ("data/raw", "models")

OK = "ok"
WARN = "warn"
FAIL = "fail"


@dataclass(frozen=True)
class Probe:
    """One line of the environment report."""

    label: str
    status: str
    detail: str


def probe_package(name: str, required: bool = True) -> Probe:
    """Import one package and report its version. Never raises.

    A missing optional package is a WARN, not a FAIL: it must not turn the
    process exit code red, because the pipeline is still fully runnable.
    """
    missing_status = FAIL if required else WARN
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        # Not just ImportError: a package can be installed yet still fail on
        # import -- missing shared library (OSError), broken wheel (SystemError),
        # version conflict (RuntimeError). Report the real reason rather than
        # guessing, and never let the probe itself take the process down.
        return Probe(name, missing_status, f"import failed: {exc!r}")
    version = getattr(module, "__version__", "unknown")
    return Probe(name, OK, version)


def probe_cuda(torch_module) -> list[Probe]:
    """Report CUDA availability, GPU name and total VRAM in MiB."""
    try:
        available = torch_module.cuda.is_available()
    except Exception as exc:  # driver mismatch, missing libcuda, ...
        return [
            Probe("cuda.is_available()", FAIL, f"raised {exc!r}"),
            Probe("GPU name", FAIL, "unknown"),
            Probe("total VRAM", FAIL, "unknown"),
        ]

    if not available:
        # A hard failure: the plan's acceptance criterion is "get
        # torch.cuda.is_available() passing first", and days 25-27 measure an
        # 8GB VRAM budget that cannot exist without a GPU.
        return [
            Probe("cuda.is_available()", FAIL, "False - no usable GPU"),
            Probe("GPU name", FAIL, "n/a"),
            Probe("total VRAM", FAIL, "n/a"),
        ]

    # A CPU-only wheel still reports is_available() == False, so reaching here
    # means a CUDA build -- but possibly the wrong major version. Day 1 asks for
    # CUDA 12.x, so check it rather than trusting the plan document.
    cuda_build = getattr(torch_module.version, "cuda", None)
    if not cuda_build:
        build_probe = Probe("torch built for CUDA", FAIL, "CPU-only wheel")
    elif not cuda_build.startswith(f"{REQUIRED_CUDA_MAJOR}."):
        build_probe = Probe(
            "torch built for CUDA", FAIL, f"{cuda_build}, want {REQUIRED_CUDA_MAJOR}.x"
        )
    else:
        build_probe = Probe("torch built for CUDA", OK, cuda_build)

    try:
        props = torch_module.cuda.get_device_properties(0)
        free_bytes, total_bytes = torch_module.cuda.mem_get_info()
    except Exception as exc:
        return [
            Probe("cuda.is_available()", OK, "True"),
            build_probe,
            Probe("GPU name", WARN, f"query raised {exc!r}"),
            Probe("total VRAM", WARN, "unknown"),
        ]

    mib = 1024 * 1024
    return [
        Probe("cuda.is_available()", OK, "True"),
        build_probe,
        Probe("GPU name", OK, props.name),
        Probe("total VRAM", OK, f"{total_bytes / mib:.0f} MiB"),
        Probe("free VRAM", OK, f"{free_bytes / mib:.0f} MiB"),
        Probe("compute capability", OK, f"sm_{props.major}{props.minor}"),
    ]


def probe_gpu_compute(torch_module) -> Probe:
    """Launch one real matmul on the GPU and check the result.

    cuda.is_available() only proves the driver can see a device. It says
    nothing about whether *this wheel* carries kernels for the GPU's compute
    capability: a build that dropped sm_75 still answers True, and only fails
    once a kernel is actually launched. `pyproject.toml` warns that newer torch
    builds may drop sm_75; this probe is the gate that enforces that warning,
    because a comment cannot fail a build.

    All-ones input keeps the expected value exact in float32 -- every element
    of ones @ ones is the inner dimension, an integer well inside the 24-bit
    mantissa -- so the comparison needs no tolerance. The two payload matrices
    are 512 KiB, but the first matmul leaves cuBLAS workspace resident:
    memory_allocated() settles at ~8.1 MiB and stays flat across repeated calls
    (measured, not a leak). Negligible against the 8 GiB budget, and it goes
    through cuBLAS rather than a scalar fallback, which is what makes this a
    real test.
    """
    size = 256
    expected = float(size)
    try:
        # is_available() itself raises when the driver is mismatched or libcuda
        # is missing -- probe_cuda() guards the same call for the same reason.
        # Leaving it outside the try would turn the most likely rebuild failure
        # into a traceback that hides every later section of the report.
        if not torch_module.cuda.is_available():
            return Probe("gpu compute (matmul)", FAIL, "not run - no usable GPU")

        ones = torch_module.ones(
            (size, size), device="cuda", dtype=torch_module.float32
        )
        product = ones @ ones
        mismatched = int((product != expected).sum().item())
        total = float(product.sum().item())
    except Exception as exc:
        # "no kernel image is available for execution on the device" arrives
        # here as RuntimeError; driver and cuBLAS faults arrive as OSError or
        # RuntimeError too. All of them mean the same thing: this wheel cannot
        # run on this card.
        return Probe("gpu compute (matmul)", FAIL, f"raised {exc!r}")

    if mismatched:
        # Report the count, not just the sum: a wrong result can still sum to
        # the expected value, so "wrong result, sum=16777216" would read as a
        # contradiction.
        return Probe(
            "gpu compute (matmul)",
            FAIL,
            f"{mismatched}/{size * size} elements != {expected:.0f}, sum={total:.0f}",
        )
    return Probe("gpu compute (matmul)", OK, f"{size}x{size} fp32, sum={total:.0f}")


def probe_matplotlib_backend() -> Probe:
    """Report the matplotlib backend, flagging GUI ones.

    This box runs WSLg, so DISPLAY is set and matplotlib picks tkagg instead of
    Agg. savefig still works -- it never needs a display -- but plt.show()
    opens a window and blocks until that window is closed, which reads as a
    hang rather than as an error. cv2's headless wheel fails loudly on the same
    kind of mistake; matplotlib's GUI fallback fails silently, and silent is
    the worse of the two.

    WARN, not FAIL: the pipeline itself is unaffected, and failing the whole
    check over a warning that one line in each plotting script prevents would
    block a day for no reason. main() still collects a FAIL raised here, so a
    future failure branch cannot silently pass the gate.
    """
    try:
        import matplotlib

        # str() covers matplotlib versions that return a BackendBase object
        # instead of a name. get_backend() resolves the backend and may import
        # pyplot on the way (3.10 auto-selects), so a broken install fails here
        # rather than at import time -- hence the try covers both calls.
        backend = str(matplotlib.get_backend())
    except Exception as exc:
        return Probe("matplotlib backend", FAIL, f"resolution failed: {exc!r}")

    if backend.lower() == "agg":
        return Probe("matplotlib backend", OK, backend)
    return Probe(
        "matplotlib backend",
        WARN,
        f"{backend} (GUI; call matplotlib.use('Agg') at the top of every plotting script)",
    )


def ensure_dirs(root: Path) -> list[Path]:
    """Create the project skeleton if absent. Idempotent; returns what it made."""
    created: list[Path] = []
    for relative in REQUIRED_DIRS:
        directory = root / relative
        if not directory.is_dir():
            directory.mkdir(parents=True, exist_ok=True)
            created.append(directory)
    for relative in GITKEEP_DIRS:
        marker = root / relative / ".gitkeep"
        if not marker.exists():
            marker.touch()
            created.append(marker)
    return created


def probe_dirs(root: Path) -> list[Probe]:
    """Report which skeleton directories exist under `root`."""
    probes = []
    for relative in REQUIRED_DIRS:
        directory = root / relative
        status = OK if directory.is_dir() else FAIL
        probes.append(Probe(f"dir {relative}", status, "present" if status == OK else "missing"))
    return probes


def format_report(title: str, probes: list[Probe]) -> str:
    """Render probes as a fixed-width block; status column stays aligned."""
    width = max((len(p.label) for p in probes), default=0)
    lines = [title, "-" * (width + 30)]
    for probe in probes:
        lines.append(f"  [{probe.status:>4}] {probe.label:<{width}}  {probe.detail}")
    return "\n".join(lines)


def main() -> int:
    """Entry point. Returns a process exit code rather than calling sys.exit."""
    print("torch-labs environment check")
    print(f"project root : {ROOT}")
    print(f"interpreter  : {sys.executable}")
    print(f"python       : {sys.version.split()[0]}")

    package_probes = [probe_package(name) for name in REQUIRED_PACKAGES]
    print()
    print(format_report("required packages", package_probes))

    optional_probes = [probe_package(name, required=False) for name in OPTIONAL_PACKAGES]
    print()
    print(format_report("optional packages", optional_probes))

    # torch is the one package we cannot fake our way around: skip the CUDA
    # section entirely if it is absent, instead of crashing on an AttributeError.
    torch_probe = next(p for p in package_probes if p.label == "torch")
    if torch_probe.status == OK:
        import torch

        cuda_probes = probe_cuda(torch)
        # Appended after the device report so a reader sees "which GPU" before
        # "can it compute". The probe degrades to FAIL with its own reason when
        # cuda is unusable, so it is safe to call unconditionally here.
        cuda_probes.append(probe_gpu_compute(torch))
    else:
        cuda_probes = [Probe("cuda", FAIL, "skipped - torch itself is unavailable")]
    print()
    print(format_report("cuda", cuda_probes))

    # Kept in a named variable, not printed inline: a FAIL raised here has to
    # reach the failure list below, or the report would say "all required checks
    # passed" while printing a red line above it.
    backend_probe = probe_matplotlib_backend()
    print()
    print(format_report("headless contract", [backend_probe]))

    created = ensure_dirs(ROOT)
    dir_probes = probe_dirs(ROOT)
    print()
    print(format_report("project skeleton", dir_probes))
    if created:
        print(f"  created {len(created)} path(s):")
        for path in created:
            print(f"    + {path.relative_to(ROOT)}")

    # CUDA probes only exist when torch imported, so collect them separately
    # rather than assuming the list is populated.
    failures = [p for p in package_probes + optional_probes if p.status == FAIL]
    failures += [p for p in cuda_probes if p.status == FAIL]
    failures += [p for p in dir_probes if p.status == FAIL]
    # WARN stays advisory; a FAIL from the headless probe has to count, which is
    # the whole point of collecting it into a variable above.
    if backend_probe.status == FAIL:
        failures.append(backend_probe)
    if failures:
        print()
        print(f"FAILED: {len(failures)} check(s) did not pass:")
        for probe in failures:
            print(f"  - {probe.label}: {probe.detail}")
        return 1

    print()
    print("all required checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
