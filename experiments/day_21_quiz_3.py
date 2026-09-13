"""
Day 21 / 2026-10-01 / Week 3 Quiz & Feynman Review: HDF5 Pipeline and Multiprocessing Concurrency
Runtime: <fill in after completion>

Goal:
    Consolidate Week 3 engineering topics (HDF5 structure traversal, lazy slicing, fork deadlock
    root cause, per-process handle isolation via worker_init_fn, RSS memory boundaries < 2GB,
    and custom multi-modal batch collation) through standard assessment with >= 60% score threshold.

Assessment Structure (Plan Section 2.5):
    - 4 Multiple Choice Questions (including 1 contrast comparison)
    - 2 Code Completion exercises (key API blanks)
    - 1 Practical coding task (<= 30 lines: multi-worker DataLoader with custom collate)
    - Self-scoring verification (pass threshold >= 60%)
    - Feynman Check #3 (Why can't h5py.File be opened in __init__? Everyday analogy)

Related concepts:
    P04-P07 (HDF5 mechanics, fork deadlocks, memory profiles), D03-D05 (Collation, workers, hooks)

Hub position:
    Review of Hub (4) Dataset/DataLoader high-concurrency pipeline engineering.

Constraints:
    - Headless: no cv2.imshow / waitKey / createTrackbar / setMouseCallback
    - Score >= 60% to pass
    - Pure functions called from main()
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]


# -----------------------------------------------------------------------------
# Part 1: Multiple Choice Questions (4 Questions)
# -----------------------------------------------------------------------------

@dataclass
class MCQ:
    question: str
    options: dict[str, str]
    correct_answer: str
    explanation: str


MCQ_QUESTIONS: list[MCQ] = [
    MCQ(
        question="1. What is the fundamental root cause of the HDF5 deadlock when h5py.File is opened inside Dataset.__init__?",
        options={
            "A": "HDF5 files automatically self-destruct when accessed by multiple threads.",
            "B": "When PyTorch DataLoader forks worker processes, open C-level file descriptors and locked internal mutexes are duplicated across address spaces, causing child processes to hang or corrupt file tables.",
            "C": "Python garbage collector automatically deletes all class attributes during fork.",
            "D": "Linux operating systems prohibit opening files larger than 1GB in subprocesses.",
        },
        correct_answer="B",
        explanation="fork() copies the process memory state including open C file descriptors and locked mutexes without restarting the HDF5 library subsystem in the child.",
    ),
    MCQ(
        question="2. How does lazy slicing (e.g. h5_dataset[idx]) differ from reading via h5_dataset[:] in terms of RAM utilization?",
        options={
            "A": "h5_dataset[idx] only transfers the bytes for that specific frame from disk to RAM; h5_dataset[:] loads the entire multi-gigabyte array into process memory.",
            "B": "h5_dataset[:] compresses data by 90% while h5_dataset[idx] is uncompressed.",
            "C": "There is no difference; h5py always buffers the whole file in RAM.",
            "D": "h5_dataset[idx] stores the data on GPU directly.",
        },
        correct_answer="A",
        explanation="h5py implements NumPy array-like slicing backed by HDF5 chunk cache, reading only the requested hyperslab from disk.",
    ),
    MCQ(
        question="3. Why must custom collate_fn be used instead of PyTorch default_collate for complex robot datasets?",
        options={
            "A": "default_collate only works with Python integers.",
            "B": "default_collate cannot handle variable length sequences or specialized multi-modal dictionaries with explicit temporal chunk stacking without falling back to suboptimal object lists.",
            "C": "PyTorch 2.x completely removed default_collate.",
            "D": "Custom collate_fn bypasses CPU and executes directly on NVLink.",
        },
        correct_answer="B",
        explanation="Default collation uses general heuristics; specialized robot data with varied dictionary keys and temporal chunk horizons requires explicit tensor assembly and shape assertions.",
    ),
    MCQ(
        question="4. In multi-worker DataLoader pipelines (num_workers=4), what is the proper architectural location to initialize worker-specific resources (e.g. file handles, database connections)?",
        options={
            "A": "Inside Dataset.__init__ before DataLoader is constructed.",
            "B": "Inside worker_init_fn or lazily inside Dataset.__getitem__ on the first read per worker PID.",
            "C": "In the global module scope of the training script.",
            "D": "Inside the neural network forward() method.",
        },
        correct_answer="B",
        explanation="worker_init_fn runs strictly inside the worker subprocess after forking/spawning, ensuring each worker establishes its own unshared handle.",
    ),
]


def run_mcqs() -> int:
    """Run MCQs and return total score out of 4.

    TODO(Student):
        Assign answers['Q1'] = 'A'/'B'/'C'/'D', etc.
    """
    # --- [TODO: Student Implementation Start] ---
    answers: dict[str, str] = {
        "Q1": "",
        "Q2": "",
        "Q3": "",
        "Q4": "",
    }
    # --- [TODO: Student Implementation End] ---

    score = 0
    for idx, mcq in enumerate(MCQ_QUESTIONS, start=1):
        key = f"Q{idx}"
        user_choice = answers.get(key, "").strip().upper()
        if user_choice == mcq.correct_answer:
            score += 1
            print(f"    {key}: CORRECT")
        else:
            print(f"    {key}: INCORRECT (Your answer: {user_choice}, Correct: {mcq.correct_answer})")
            print(f"      Explanation: {mcq.explanation}")
    return score


# -----------------------------------------------------------------------------
# Part 2: Code Completion (2 Questions)
# -----------------------------------------------------------------------------

def code_completion_1() -> bool:
    """Q5: Given a list of sample dicts [{'img': Tensor, 'action': Tensor}], stack into batch dict.

    TODO(Student):
        Stack batch items along dim 0 for both keys.
    """
    batch = [
        {"img": torch.zeros(3, 84, 84), "action": torch.zeros(7)},
        {"img": torch.zeros(3, 84, 84), "action": torch.zeros(7)},
    ]

    # --- [TODO: Student Implementation Start] ---
    raise NotImplementedError("Q5: Implement batch stacking for dictionary keys")
    # --- [TODO: Student Implementation End] ---

    return out_dict["img"].shape == (2, 3, 84, 84) and out_dict["action"].shape == (2, 7)


def code_completion_2() -> bool:
    """Q6: Implement process-safe lazy handle getter pattern.

    TODO(Student):
        If self._handle is None: initialize handle; return self._handle.
    """
    class LazyHandler:
        def __init__(self) -> None:
            self._handle: str | None = None

        def get_handle(self) -> str:
            # --- [TODO: Student Implementation Start] ---
            raise NotImplementedError("Q6: Implement lazy initialization getter")
            # --- [TODO: Student Implementation End] ---
            return self._handle

    handler = LazyHandler()
    h1 = handler.get_handle()
    h2 = handler.get_handle()
    return h1 == "initialized" and h2 == "initialized"


# -----------------------------------------------------------------------------
# Part 3: Practical Coding Challenge (1 Question)
# -----------------------------------------------------------------------------

def practical_safe_dataloader_pipeline() -> bool:
    """Q7 Practical: Multi-worker DataLoader with custom collate.

    TODO(Student):
        1. Define inline SyntheticDataset with length 16 returning {'obs': Tensor[7], 'act': Tensor[7]}.
        2. Define collate function stacking 'obs' and 'act'.
        3. Create DataLoader(ds, batch_size=4, num_workers=2, collate_fn=collate).
        4. Iterate 2 batches and verify batch['obs'].shape == (4, 7).
    """
    # --- [TODO: Student Implementation Start] ---
    raise NotImplementedError("Q7: Implement multi-worker DataLoader with custom collation")
    # --- [TODO: Student Implementation End] ---

    return True


# -----------------------------------------------------------------------------
# Part 4: Feynman Review Check #3
# -----------------------------------------------------------------------------

def feynman_review_check_3() -> dict[str, str]:
    """Feynman Check #3: Everyday analogy for why h5py.File cannot be opened in __init__.

    TODO(Student):
        Provide:
        1. An everyday analogy (e.g. shared physical keys, duplicate notary seals, bank vault passbooks).
        2. The technical explanation of what happens across the process boundary.
    """
    # --- [TODO: Student Implementation Start] ---
    feynman_response = {
        "analogy": "",
        "technical_explanation": "",
    }
    # --- [TODO: Student Implementation End] ---
    return feynman_response


def main() -> None:
    """Entry point for Day 21."""
    print("=== Day 21: Week 3 Quiz & Feynman Review ===")

    # Part 1: MCQs (4 points)
    print("\n--- Part 1: MCQs (4 Points) ---")
    mcq_score = run_mcqs()

    # Part 2: Code Completions (2 points)
    print("\n--- Part 2: Code Completion (2 Points) ---")
    c1_ok = code_completion_1()
    print(f"    Q5 Completion: {'PASS' if c1_ok else 'FAIL'}")
    c2_ok = code_completion_2()
    print(f"    Q6 Completion: {'PASS' if c2_ok else 'FAIL'}")
    completion_score = (1 if c1_ok else 0) + (1 if c2_ok else 0)

    # Part 3: Practical Challenge (1 point)
    print("\n--- Part 3: Practical Challenge (1 Point) ---")
    p_ok = practical_safe_dataloader_pipeline()
    print(f"    Q7 Practical: {'PASS' if p_ok else 'FAIL'}")
    practical_score = 1 if p_ok else 0

    total_score = mcq_score + completion_score + practical_score
    pct = (total_score / 7.0) * 100
    print(f"\nTotal Score: {total_score}/7 ({pct:.1f}%)")
    assert pct >= 60.0, f"Score {pct:.1f}% is below passing threshold (60.0%)"

    # Part 4: Feynman Check
    print("\n--- Part 4: Feynman Check #3 ---")
    feynman = feynman_review_check_3()
    print(f"    Analogy: {feynman.get('analogy', '<pending>')}")

    print("\n>>> Day 21 Acceptance: ALL PASS")


if __name__ == "__main__":
    main()
