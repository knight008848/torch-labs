"""
Day 14 / 2026-09-24 / Week 2 Quiz & Feynman Review: Modules, Loops, and Dataset Contracts
Runtime: <fill in after completion>

Goal:
    Consolidate Week 2 foundations (nn.Module, parameters vs buffers, loss/optimizer steps,
    train vs eval modes, Dataset/DataLoader interface contracts, and vision tensor bridging)
    through standard assessment (4 MCQs, 2 Code Completions, 1 Practical) with >= 60% score
    threshold and Feynman review #2.

Assessment Structure (Plan Section 2.5):
    - 4 Multiple Choice Questions (including 1 contrast comparison)
    - 2 Code Completion exercises (key API blanks)
    - 1 Practical coding task (<= 30 lines: custom dataset + dataloader batch extraction)
    - Self-scoring verification (pass threshold >= 60%)
    - Feynman Check #2 (Dataset vs DataLoader division of labor)

Related concepts:
    M01-M07 (Modules, training loops, checkpoints), D01-D06 (Dataset & DataLoader contracts),
    P01-P03 (Vision tensor bridging)

Hub position:
    Review of Hub (3) nn.Module and Hub (4) Dataset/DataLoader foundations.

Constraints:
    - Headless: no cv2.imshow / waitKey / createTrackbar / setMouseCallback
    - Score >= 60% to pass
    - Pure functions called from main()
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
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
        question="1. What is the fundamental difference between parameters and buffers in an nn.Module?",
        options={
            "A": "Parameters are saved in state_dict, while buffers are completely discarded during torch.save.",
            "B": "Parameters require gradients and are updated by optimizers; buffers (like running_mean in BatchNorm) are non-trainable state updated during forward.",
            "C": "Buffers only exist on CPU, whereas parameters reside on GPU.",
            "D": "Buffers are deprecated in PyTorch 2.x and replaced by Python attributes.",
        },
        correct_answer="B",
        explanation="Both parameters and buffers are serialized in state_dict, but only parameters have requires_grad=True and receive gradient updates from optimizers.",
    ),
    MCQ(
        question="2. In PyTorch image processing for robotic vision, why must raw OpenCV images [H, W, C] be permuted to [C, H, W]?",
        options={
            "A": "Because PyTorch torch.nn.Conv2d expects channel dimension first: [Batch, Channel, Height, Width].",
            "B": "Because permute() saves 50% GPU memory compared to contiguous arrays.",
            "C": "Because OpenCV only supports 2D images.",
            "D": "Because Python cannot index the last dimension of a tensor.",
        },
        correct_answer="A",
        explanation="PyTorch standard convolutional operators follow the NCHW memory layout by default.",
    ),
    MCQ(
        question="3. What is the precise division of labor between Dataset and DataLoader in PyTorch?",
        options={
            "A": "Dataset handles multiprocessing; DataLoader defines how a single item is indexed.",
            "B": "Dataset abstracts sample-level indexing (__len__, __getitem__); DataLoader orchestrates batching, shuffling, multiprocessing, and memory pinning.",
            "C": "Dataset converts tensors to numpy; DataLoader converts numpy to tensors.",
            "D": "There is no difference; DataLoader inherits directly from Dataset.",
        },
        correct_answer="B",
        explanation="Dataset is the data source contract (random access by index). DataLoader is the stream consumer/assembler (batching, workers, pinning).",
    ),
    MCQ(
        question="4. Why should shuffle=True and drop_last=True be explicitly turned OFF during pipeline throughput benchmarking?",
        options={
            "A": "Because shuffling consumes 90% of GPU compute time.",
            "B": "Because drop_last=True discards the remainder batch causing sample count variance, and shuffle introduces random disk seek patterns that prevent repeatable cache comparisons.",
            "C": "Because DataLoader throws an error if both flags are enabled.",
            "D": "Because benchmark profilers only support single-threaded sequential execution.",
        },
        correct_answer="B",
        explanation="Benchmarking requires deterministic, full-dataset traversal so metrics represent steady-state hardware throughput without random seek latency or sample drops.",
    ),
]


def run_mcqs() -> int:
    """Run MCQs and return total score out of 4.

    TODO(Student):
        Answer each question by assigning answers['Q1'] = 'A'/'B'/'C'/'D', etc.
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
    """Q5: Transform a batch of HWC uint8 images [B, 84, 84, 3] to CHW float32 [B, 3, 84, 84] in [0, 1].

    TODO(Student):
        Permute axes (0, 3, 1, 2), convert to float32, and divide by 255.0.
    """
    raw_batch = torch.randint(0, 256, (4, 84, 84, 3), dtype=torch.uint8)

    # --- [TODO: Student Implementation Start] ---
    raise NotImplementedError("Q5: Permute batch HWC -> CHW and normalize to [0, 1]")
    # --- [TODO: Student Implementation End] ---

    return out_batch.shape == (4, 3, 84, 84) and out_batch.dtype == torch.float32 and 0.0 <= out_batch.min().item() <= out_batch.max().item() <= 1.0


def code_completion_2() -> bool:
    """Q6: Iterate through a DataLoader with model.eval() and torch.no_grad(), collecting batch sizes.

    TODO(Student):
        Set model to eval mode, wrap iteration with torch.no_grad(), and tally total samples.
    """
    class DummyDataset(Dataset[torch.Tensor]):
        def __len__(self) -> int:
            return 10
        def __getitem__(self, idx: int) -> torch.Tensor:
            return torch.tensor([idx], dtype=torch.float32)

    loader = DataLoader(DummyDataset(), batch_size=4, shuffle=False)
    model = nn.Linear(1, 1)

    # --- [TODO: Student Implementation Start] ---
    raise NotImplementedError("Q6: Iterate loader under eval mode and torch.no_grad()")
    # --- [TODO: Student Implementation End] ---

    return total_samples == 10 and not model.training


# -----------------------------------------------------------------------------
# Part 3: Practical Coding Challenge (1 Question)
# -----------------------------------------------------------------------------

def practical_multimodal_loader() -> bool:
    """Q7 Practical: Implement minimal custom Dataset returning multi-modal dict and batch via DataLoader.

    TODO(Student):
        1. Define inline MiniDataset returning {'img': [3, 84, 84], 'state': [7]} with length 8.
        2. Create DataLoader with batch_size=4.
        3. Fetch first batch, asserting batch['img'].shape == (4, 3, 84, 84) and batch['state'].shape == (4, 7).
        4. Return True if assertions pass.
    """
    # --- [TODO: Student Implementation Start] ---
    raise NotImplementedError("Q7: Implement MiniDataset and DataLoader batch assertion")
    # --- [TODO: Student Implementation End] ---

    return True


# -----------------------------------------------------------------------------
# Part 4: Feynman Review Check #2
# -----------------------------------------------------------------------------

def feynman_review_check_2() -> dict[str, str]:
    """Feynman Check #2: Division of labor between Dataset and DataLoader.

    TODO(Student):
        Explain in your own words (without looking at docs):
        1. What is the Dataset's sole responsibility?
        2. What does DataLoader do that Dataset should never do?
        3. An everyday analogy (e.g. restaurant kitchen, warehouse).
    """
    # --- [TODO: Student Implementation Start] ---
    feynman_response = {
        "dataset_role": "",
        "dataloader_role": "",
        "analogy": "",
    }
    # --- [TODO: Student Implementation End] ---
    return feynman_response


def main() -> None:
    """Entry point for Day 14."""
    print("=== Day 14: Week 2 Quiz & Feynman Review ===")

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
    p_ok = practical_multimodal_loader()
    print(f"    Q7 Practical: {'PASS' if p_ok else 'FAIL'}")
    practical_score = 1 if p_ok else 0

    total_score = mcq_score + completion_score + practical_score
    pct = (total_score / 7.0) * 100
    print(f"\nTotal Score: {total_score}/7 ({pct:.1f}%)")
    assert pct >= 60.0, f"Score {pct:.1f}% is below passing threshold (60.0%)"

    # Part 4: Feynman Check
    print("\n--- Part 4: Feynman Check #2 ---")
    feynman = feynman_review_check_2()
    print(f"    Analogy: {feynman.get('analogy', '<pending>')}")

    print("\n>>> Day 14 Acceptance: ALL PASS")


if __name__ == "__main__":
    main()
