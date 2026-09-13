"""
Day 07 / 2026-09-17 / Week 1 Quiz & Feynman Review: Tensor and Autograd Foundations
Runtime: <fill in after completion>

Goal:
    Consolidate Week 1 foundations (Tensor trinity, shape ops, indexing, broadcasting,
    in-place traps, autograd dynamic graph) through standard assessment (4 MCQs,
    2 Code Completions, 1 Practical) with >= 60% score threshold and Feynman review.

Assessment Structure (Plan Section 2.5):
    - 4 Multiple Choice Questions (including 1 contrast comparison)
    - 2 Code Completion exercises (key API blanks)
    - 1 Practical coding task (<= 30 lines: create tensor -> permute CHW -> normalize -> backward)
    - Self-scoring verification (pass threshold >= 60%)
    - Feynman Check #1 (Tensor & autograd hub explanation)

Related concepts:
    T01-T07 (Tensor foundations), G01-G05 (Autograd mechanics)

Hub position:
    Review of Hub (1) Tensor and Hub (2) Autograd.

Constraints:
    - Headless: no cv2.imshow / waitKey / createTrackbar / setMouseCallback
    - Score >= 60% to pass
    - Pure functions called from main()
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

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
        question="1. What is the fundamental difference between tensor.view() and tensor.reshape()?",
        options={
            "A": "view() always copies memory, while reshape() never copies memory.",
            "B": "view() requires the tensor to be contiguous; reshape() returns a view if contiguous or a copy if non-contiguous.",
            "C": "reshape() only works on CPU tensors, whereas view() works on CUDA.",
            "D": "There is no difference; they are exact aliases in PyTorch.",
        },
        correct_answer="B",
        explanation="view() strictly checks .is_contiguous() and errors out if false. reshape() returns a view if possible or copies data if needed.",
    ),
    MCQ(
        question="2. Why does calling backward() on a tensor modified by an in-place operation (e.g. x.add_(1)) often fail?",
        options={
            "A": "Because in-place operations delete the GPU memory immediately.",
            "B": "Because PyTorch does not allow addition on tensors requiring gradients.",
            "C": "Because the saved intermediate values needed by the backward pass were overwritten in-place.",
            "D": "Because autograd only works on integer tensors.",
        },
        correct_answer="C",
        explanation="Autograd computation graph saves intermediate tensors for gradient calculations. In-place mutation overwrites those values, corrupting the backward chain.",
    ),
    MCQ(
        question="3. When converting a NumPy array to a PyTorch tensor via torch.from_numpy(arr):",
        options={
            "A": "Memory is deeply duplicated; modifying arr never affects the tensor.",
            "B": "Memory is zero-copy shared on CPU; modifying arr directly modifies the tensor.",
            "C": "The tensor is automatically allocated on CUDA:0.",
            "D": "The data type is always cast to float32 regardless of NumPy dtype.",
        },
        correct_answer="B",
        explanation="torch.from_numpy shares the underlying CPU buffer pointers without copying.",
    ),
    MCQ(
        question="4. In robot data pipelines, why do raw image observations use uint8 while neural network inputs use float32?",
        options={
            "A": "uint8 saves 75% memory/bandwidth during storage and transfer; float32 provides continuous dynamic range for smooth gradient propagation.",
            "B": "PyTorch cannot store float32 on disk.",
            "C": "uint8 supports negative numbers while float32 does not.",
            "D": "Convolutional layers only accept uint8 inputs.",
        },
        correct_answer="A",
        explanation="Raw images in uint8 take 1 byte per channel vs 4 bytes for float32. Backpropagation requires continuous float representations.",
    ),
]


def run_mcqs() -> int:
    """Run MCQs and return total score out of 4.

    TODO(Student):
        Answer each question by assigning answers['Q1'] = 'A'/'B'/'C'/'D', etc.
    """
    # --- [TODO: Student Implementation Start] ---
    # Fill in your answers: "A", "B", "C", or "D"
    answers: dict[str, str] = {
        "Q1": "",  # fill with "A", "B", "C", or "D"
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
    """Q5: Given raw_img with shape [84, 84, 3] (HWC), transform it to [3, 84, 84] (CHW)
    and ensure it is contiguous in memory.

    TODO(Student):
        Replace NotImplementedError with the one-line transformation.
    """
    raw_img = torch.randint(0, 256, (84, 84, 3), dtype=torch.uint8)

    # --- [TODO: Student Implementation Start] ---
    # Hint: raw_img.permute(...).contiguous()
    raise NotImplementedError("Q5: Implement HWC -> CHW permute and make contiguous")
    # --- [TODO: Student Implementation End] ---

    return chw_img.shape == (3, 84, 84) and chw_img.is_contiguous()


def code_completion_2() -> bool:
    """Q6: Given parameter tensor w with requires_grad=True, perform forward pass,
    backward pass, and reset gradients to zero without memory leak.

    TODO(Student):
        Implement forward (loss = (w ** 2).sum()), backward, and gradient zeroing.
    """
    w = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

    # --- [TODO: Student Implementation Start] ---
    raise NotImplementedError("Q6: Implement forward, backward, verify grad, and zero_grad")
    # --- [TODO: Student Implementation End] ---

    return w.grad is not None and torch.all(w.grad == 0.0).item()


# -----------------------------------------------------------------------------
# Part 3: Practical Coding Challenge (1 Question)
# -----------------------------------------------------------------------------

def practical_pipeline_mini_sprint() -> float:
    """Q7 Practical: Mini data-to-loss pipeline in <= 30 lines.

    TODO(Student):
        1. Create synthetic image batch `images_hwc` [B=4, H=84, W=84, C=3] in uint8.
        2. Convert to float32 and permute to [B, C, H, W].
        3. Normalize pixel values to [0.0, 1.0] by dividing by 255.0.
        4. Define linear weight `weight` [3, 1] with requires_grad=True.
        5. Compute channel-wise weighted sum across pixels and compute mean loss.
        6. Execute loss.backward() and assert weight.grad is not None and has shape [3, 1].
        7. Return loss.item().
    """
    # --- [TODO: Student Implementation Start] ---
    raise NotImplementedError("Q7 Practical: Mini pipeline from raw images to backward pass")
    # --- [TODO: Student Implementation End] ---

    assert loss_val > 0, "Loss must be positive"
    return loss_val


# -----------------------------------------------------------------------------
# Part 4: Feynman Hub Check #1
# -----------------------------------------------------------------------------

def feynman_review_week_1() -> str:
    """Feynman Check #1:
    Answer without looking at reference docs:
    1. What is Hub (1) Tensor and why is it a hub?
    2. What is Hub (2) Autograd and its relationship with Hub (1)?
    3. State an everyday Chinese analogy for Tensor and autograd.
    """
    review_text = """
    [Student to fill in their own words during review]
    1. Hub (1) Tensor: The universal multi-dimensional data carrier of PyTorch.
       It is a hub because all data (images, states, actions, model weights) must enter
       and leave as Tensors. Downstream components (autograd, nn.Module, DataLoader) all operate on it.
    2. Hub (2) Autograd: The dynamic graph engine that tracks operations on Tensors with
       requires_grad=True. It connects Tensors to parameter optimization via backward().
    3. Analogy: Tensor is a multi-dimensional spreadsheet (dtype = cell format, device = local disk vs cloud server).
       Autograd is an automatic formula calculation audit trail: whenever you change an input cell,
       it automatically traces every formula back to calculate sensitivity.
    """
    return review_text.strip()


def main() -> None:
    """Entry point for Day 07 Week 1 Quiz."""
    print("=======================================================")
    print("           Day 07: Week 1 Assessment & Review          ")
    print("=======================================================")

    total_score = 0
    max_score = 7

    # Part 1: MCQs (4 pts)
    print("\n--- [Part 1] Multiple Choice Questions (4 pts) ---")
    mcq_score = run_mcqs()
    total_score += mcq_score
    print(f"Part 1 Score: {mcq_score} / 4")

    # Part 2: Code Completion (2 pts)
    print("\n--- [Part 2] Code Completion (2 pts) ---")
    cc1_passed = False
    try:
        cc1_passed = code_completion_1()
        print("  Q5 (HWC -> CHW Contiguous): PASS (+1)")
        total_score += 1
    except NotImplementedError:
        print("  Q5: NOT IMPLEMENTED")

    cc2_passed = False
    try:
        cc2_passed = code_completion_2()
        print("  Q6 (Autograd Loop & Zero): PASS (+1)")
        total_score += 1
    except NotImplementedError:
        print("  Q6: NOT IMPLEMENTED")

    # Part 3: Practical Challenge (1 pt)
    print("\n--- [Part 3] Practical Coding Sprint (1 pt) ---")
    try:
        loss = practical_pipeline_mini_sprint()
        print(f"  Q7 (Mini Pipeline Sprint): PASS (Loss={loss:.4f}) (+1)")
        total_score += 1
    except NotImplementedError:
        print("  Q7: NOT IMPLEMENTED")

    # Final tally
    percentage = (total_score / max_score) * 100.0
    print("\n=======================================================")
    print(f"Final Score: {total_score} / {max_score} ({percentage:.1f}%)")
    if percentage >= 60.0:
        print("Status: PASSED (>= 60%)")
    else:
        print("Status: NOT PASSED (< 60%). Please review Week 1 materials.")
    print("=======================================================")

    # Feynman review
    print("\n--- [Part 4] Feynman Review #1 ---")
    print(feynman_review_week_1())


if __name__ == "__main__":
    main()
