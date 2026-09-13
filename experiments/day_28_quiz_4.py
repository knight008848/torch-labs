"""
Day 28 / 2026-10-08 / Week 4 Quiz & Feynman Review: Optimization, Action Chunking, and Profiling
Runtime: <fill in after completion>

Goal:
    Consolidate Week 4 topics (state tensorization, action normalization [-1, 1], temporal action
    chunking [K, 7], pin_memory host-to-device streaming under 8GB VRAM, parameter sweep tuning,
    and pure pipeline FPS benchmarking) through final weekly assessment with >= 60% threshold.

Assessment Structure (Plan Section 2.5):
    - 4 Multiple Choice Questions (including 1 contrast comparison)
    - 2 Code Completion exercises (key API blanks)
    - 1 Practical coding task (<= 30 lines: pipeline throughput benchmark loop)
    - Self-scoring verification (pass threshold >= 60%)
    - Feynman Check #4 (Where is the data pipeline bottleneck and how to prove it?)
    - Aggregate four-week error distribution table (C/A/E/K counts)

Related concepts:
    P08-P11 (Action chunking, normalization, prefetch, benchmarking), B05 (Pinned memory)

Hub position:
    Review of Hub (4) Dataset/DataLoader end-to-end performance and temporal engineering.

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
        question="1. In temporal action chunking (sliding window of size K=8), what happens at the end of an episode when fewer than K frames remain?",
        options={
            "A": "PyTorch automatically aborts the epoch.",
            "B": "A boundary padding strategy must be applied, such as repeating the final robot action or appending zero actions to maintain fixed tensor shape [K, 7].",
            "C": "The action tensor is reduced to [remaining, 7] and batch collation skips dimension checks.",
            "D": "The episode is discarded completely.",
        },
        correct_answer="B",
        explanation="Neural network policy architectures require static tensor shapes [Batch, K, Dim]. When remaining frames < K, explicit repeating or padding maintains shape invariants.",
    ),
    MCQ(
        question="2. How does pin_memory=True accelerate data loading when combined with tensor.to('cuda', non_blocking=True)?",
        options={
            "A": "It converts all floating-point numbers into 8-bit integers.",
            "B": "It allocates host tensors in non-pageable (pinned) physical RAM, enabling the GPU Direct Memory Access (DMA) controller to copy memory asynchronously without CPU intervention.",
            "C": "It doubles the GPU clock frequency.",
            "D": "It caches the entire dataset in the L2 cache of the CPU.",
        },
        correct_answer="B",
        explanation="Standard pageable host memory requires an intermediate CPU copy before GPU DMA transfer; pinned memory enables direct, non-blocking hardware DMA streaming.",
    ),
    MCQ(
        question="3. When tuning DataLoader hyperparameters for an 8GB VRAM GPU, what is the trade-off of setting prefetch_factor excessively high (e.g. prefetch_factor=16)?",
        options={
            "A": "The script fails with a syntax error.",
            "B": "Each worker pre-allocates and holds many batches in memory queues, potentially exceeding host RAM and triggering CPU swapping or out-of-memory crashes.",
            "C": "GPU compute speed drops to zero.",
            "D": "PyTorch automatically disables CUDA.",
        },
        correct_answer="B",
        explanation="prefetch_factor controls queue size per worker. High prefetch multipliers with large image batches cause RAM explosion and IPC socket saturation.",
    ),
    MCQ(
        question="4. In pure data pipeline FPS profiling (benchmarking DataLoader without neural network forward/backward), why is isolating data loading essential?",
        options={
            "A": "Because Python cannot run neural networks and data loaders simultaneously.",
            "B": "Because GPU compute time can mask slow I/O; isolating DataLoader reveals the true data ingestion ceiling and proves whether storage is the bottleneck.",
            "C": "Because backward pass permanently deletes the dataset from disk.",
            "D": "Because FPS profilers only support CPU tensors.",
        },
        correct_answer="B",
        explanation="Amortized training time blends compute and I/O. Benchmarking DataLoader alone measures the maximum throughput ceiling the input pipeline can feed to the model.",
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
    """Q5: Pad a trailing action slice of length 3 to chunk size K=8 by repeating the last frame.

    TODO(Student):
        Repeat trailing slice[-1:] for the missing (8 - 3 = 5) frames and concatenate.
    """
    trailing_slice = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # shape [3, 2]
    k = 8

    # --- [TODO: Student Implementation Start] ---
    raise NotImplementedError("Q5: Pad trailing slice to length K=8 by repeating last action")
    # --- [TODO: Student Implementation End] ---

    return padded_chunk.shape == (8, 2) and torch.equal(padded_chunk[-1], trailing_slice[-1])


def code_completion_2() -> bool:
    """Q6: Transfer a pinned CPU tensor to CUDA asynchronously without blocking host thread.

    TODO(Student):
        Verify tensor is pinned, transfer to device using non_blocking=True.
    """
    cpu_tensor = torch.zeros(10).pin_memory()

    # --- [TODO: Student Implementation Start] ---
    raise NotImplementedError("Q6: Asynchronous non-blocking transfer to device")
    # --- [TODO: Student Implementation End] ---

    return target_tensor.device.type in ("cuda", "cpu")


# -----------------------------------------------------------------------------
# Part 3: Practical Coding Challenge (1 Question)
# -----------------------------------------------------------------------------

def practical_fps_benchmark_loop() -> float:
    """Q7 Practical: Measure pure DataLoader iteration FPS across 20 batches.

    TODO(Student):
        1. Create synthetic dataset with 64 samples.
        2. Create DataLoader(batch_size=8).
        3. Measure elapsed time over 20 iterations.
        4. Return calculated FPS (samples / elapsed_seconds).
    """
    # --- [TODO: Student Implementation Start] ---
    raise NotImplementedError("Q7: Pure DataLoader throughput benchmark loop returning FPS")
    # --- [TODO: Student Implementation End] ---

    assert fps > 0.0
    return fps


# -----------------------------------------------------------------------------
# Part 4: Feynman Review Check #4 & Four-Week Error Aggregation
# -----------------------------------------------------------------------------

def feynman_review_check_4() -> dict[str, str]:
    """Feynman Check #4: Where is the data pipeline bottleneck and how to prove it?

    TODO(Student):
        Explain:
        1. How do you distinguish between disk I/O bound vs worker CPU bound vs GPU compute bound?
        2. What diagnostic experiment proves whether DataLoader is starving the GPU?
    """
    # --- [TODO: Student Implementation Start] ---
    feynman_response = {
        "bottleneck_identification": "",
        "proof_experiment": "",
    }
    # --- [TODO: Student Implementation End] ---
    return feynman_response


def aggregate_error_distribution() -> dict[str, int]:
    """Aggregate total error counts across Weeks 1-4 by category (C, A, E, K).

    TODO(Student):
        Audit docs/error_log.md and tally counts for:
        - C: Concept misunderstanding
        - A: Application failure
        - E: Expression ambiguity
        - K: Knowledge confusion
    """
    # --- [TODO: Student Implementation Start] ---
    error_counts = {
        "C": 0,
        "A": 0,
        "E": 0,
        "K": 0,
    }
    # --- [TODO: Student Implementation End] ---
    return error_counts


def main() -> None:
    """Entry point for Day 28."""
    print("=== Day 28: Week 4 Final Comprehensive Assessment ===")

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
    measured_fps = practical_fps_benchmark_loop()
    print(f"    Q7 Practical: Measured {measured_fps:.1f} FPS")
    practical_score = 1 if measured_fps > 0 else 0

    total_score = mcq_score + completion_score + practical_score
    pct = (total_score / 7.0) * 100
    print(f"\nTotal Score: {total_score}/7 ({pct:.1f}%)")
    assert pct >= 60.0, f"Score {pct:.1f}% is below passing threshold (60.0%)"

    # Part 4: Feynman Check & Error Summary
    print("\n--- Part 4: Feynman Check #4 & Error Distribution ---")
    feynman = feynman_review_check_4()
    print(f"    Bottleneck Diagnosis: {feynman.get('bottleneck_identification', '<pending>')}")
    errors = aggregate_error_distribution()
    print(f"    Four-Week Error Distribution: {errors}")

    print("\n>>> Day 28 Acceptance: ALL PASS")


if __name__ == "__main__":
    main()
