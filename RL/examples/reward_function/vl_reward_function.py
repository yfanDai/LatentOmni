# Pure VL reward function — no latent / Monet dependencies.
# Compatible with the existing verl reward pipeline:
#   worker.reward.reward_function = ./examples/reward_function/vl_reward_function.py:compute_score
#   worker.rule_based_judge.judge_function = ./examples/reward_function/vl_reward_function.py:rule_then_api_batch_judge

import re
from typing import Dict, List, Optional

import torch
from mathruler.grader import extract_boxed_content, grade_answer

from verl.workers.rollout.utils.util import extract_no_boxed_answer
from tools.api_judge import api_batch_judge

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BOXED_RE = re.compile(r"\\boxed\{.*?\}", re.DOTALL)


def format_reward(predict: str) -> float:
    """Return 1.0 if the prediction contains a \\boxed{} expression."""
    return 1.0 if BOXED_RE.search(predict) else 0.0


def extract_and_check(predict: str, ground_truth: str) -> float:
    """Rule-based correctness: extract answer and grade it."""
    answer = extract_boxed_content(predict)
    if answer == "None":
        answer = extract_no_boxed_answer(predict)
    return grade_answer(answer, ground_truth)


# ---------------------------------------------------------------------------
# Simple rule-based reward (no length penalty, no hash server needed)
# ---------------------------------------------------------------------------

def compute_score(
    predicts: List[str],
    ground_truths: List[str],
    format_weight: float = 0.1,
    # The VL pipeline does not use a hash server, so length-penalty args are
    # accepted for API compatibility but simply ignored.
    length_penalty_weight: float = 0.0,
    resp_lengths=None,
    ref_resp_lengths=None,
) -> List[Dict[str, float]]:
    scores = []
    for predict, ground_truth in zip(predicts, ground_truths):
        predict = re.sub(r"\s*(<|>|/)\s*", r"\1", predict)  # Qwen2.5-VL format fix
        fmt = format_reward(predict)
        acc = 1.0 if extract_and_check(predict, ground_truth) else 0.0
        scores.append(
            {
                "overall": (1 - format_weight) * acc + format_weight * fmt,
                "format": fmt,
                "accuracy": acc,
            }
        )
    return scores


# ---------------------------------------------------------------------------
# Rule-first then API judge (mirrors monet_reward_function.rule_then_api_batch_judge)
# ---------------------------------------------------------------------------

def rule_then_api_batch_judge(
    questions: List[Optional[str]],
    preds: List[Optional[str]],
    gts: List[Optional[str]],
    *,
    api_name: Optional[str] = "gemini-2.5-pro",
    api_max_workers: int = 32,
    api_kwargs: Optional[Dict] = None,
    client=None,
    dataset_name: str = "",
    repetition_penalty: bool = False,
):
    """First tries rule-based grading; calls API only for samples that fail."""
    correctness_list = [extract_and_check(p, g) for p, g in zip(preds, gts)]

    # Collect indices where rule-based grading returned False
    api_indices = [i for i, c in enumerate(correctness_list) if not c]

    if api_indices:
        api_results = api_batch_judge(
            [questions[i] for i in api_indices],
            [preds[i] for i in api_indices],
            [gts[i] for i in api_indices],
            api_name=api_name,
            api_max_workers=api_max_workers,
            api_kwargs=api_kwargs,
            client=client,
            repetition_penalty=repetition_penalty,
        )
        for rank, i in enumerate(api_indices):
            if api_results[rank] is not None:
                correctness_list[i] = api_results[rank]

    return correctness_list
