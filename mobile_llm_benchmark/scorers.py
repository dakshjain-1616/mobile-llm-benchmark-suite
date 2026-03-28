"""Per-benchmark scoring modules.

Each scorer takes a question dict and a model response string and returns
True (correct) or False (incorrect).

Prompts follow zero-shot instruction format compatible with small chat models.
"""

from __future__ import annotations

import re
import string
from abc import ABC, abstractmethod
from typing import Any


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseScorer(ABC):
    """Abstract base for benchmark scorers."""

    @abstractmethod
    def build_prompt(self, question: dict[str, Any]) -> str:
        """Build the prompt string to send to the model."""

    @abstractmethod
    def score(self, question: dict[str, Any], response: str) -> bool:
        """Return True if the response is correct."""

    def max_tokens(self) -> int:
        return 128

    def temperature(self) -> float:
        return 0.0


# ---------------------------------------------------------------------------
# GSM8K
# ---------------------------------------------------------------------------

class GSM8KScorer(BaseScorer):
    """Math word problems — extract a number from the response."""

    def build_prompt(self, question: dict) -> str:
        return (
            f"Solve the following math problem step by step.\n\n"
            f"Problem: {question['question']}\n\n"
            "Think through it carefully, then on the final line write:\n"
            "Answer: <number>"
        )

    def score(self, question: dict, response: str) -> bool:
        gt = str(question["answer"]).strip().replace(",", "").rstrip(".")
        # Look for "Answer: <number>" pattern first
        match = re.search(r"[Aa]nswer[:\s]+([0-9,.\-/]+)", response)
        if match:
            pred = match.group(1).replace(",", "").rstrip(".")
            return self._numbers_equal(pred, gt)
        # Fallback: find last number in response
        numbers = re.findall(r"-?[0-9,]+\.?[0-9]*", response.replace(",", ""))
        if numbers:
            return self._numbers_equal(numbers[-1], gt)
        return False

    @staticmethod
    def _numbers_equal(a: str, b: str) -> bool:
        try:
            return abs(float(a) - float(b)) < 1e-4
        except ValueError:
            return a.strip() == b.strip()

    def max_tokens(self) -> int:
        return 512


# ---------------------------------------------------------------------------
# Multiple choice helper
# ---------------------------------------------------------------------------

_LETTER_MAP = {"A": 0, "B": 1, "C": 2, "D": 3}
_LETTER_PATTERN = re.compile(r"\b([A-Da-d])\b")


def _extract_mc_answer(response: str) -> str:
    """Extract a single letter (A-D) from a model response."""
    # Prefer "Answer: X" or "The answer is X"
    m = re.search(r"(?:answer is|answer:|the answer)[:\s]*([A-Da-d])\b", response, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Find first standalone letter
    m = _LETTER_PATTERN.search(response)
    if m:
        return m.group(1).upper()
    # Just the first character if it's a letter
    stripped = response.strip()
    if stripped and stripped[0].upper() in "ABCD":
        return stripped[0].upper()
    return ""


def _build_mc_prompt(question: str, choices: list[str], context: str = "") -> str:
    labels = ["A", "B", "C", "D"]
    formatted = "\n".join(
        f"{labels[i]}. {c}" for i, c in enumerate(choices[:4])
    )
    ctx = f"\n{context}" if context else ""
    return (
        f"Question:{ctx}\n{question}\n\n"
        f"Options:\n{formatted}\n\n"
        "Reply with ONLY the letter of the correct answer (A, B, C, or D)."
    )


# ---------------------------------------------------------------------------
# ARC-Challenge
# ---------------------------------------------------------------------------

class ARCScorer(BaseScorer):

    def build_prompt(self, question: dict) -> str:
        return _build_mc_prompt(question["question"], question["choices"])

    def score(self, question: dict, response: str) -> bool:
        gt = str(question["answer"]).strip().upper()
        pred = _extract_mc_answer(response)
        return pred == gt and pred != ""

    def max_tokens(self) -> int:
        return 16


# ---------------------------------------------------------------------------
# MMLU
# ---------------------------------------------------------------------------

class MMLUScorer(BaseScorer):

    def build_prompt(self, question: dict) -> str:
        subject = question.get("subject", "general knowledge")
        ctx = f"Subject: {subject}"
        return _build_mc_prompt(question["question"], question["choices"], context=ctx)

    def score(self, question: dict, response: str) -> bool:
        gt = str(question["answer"]).strip().upper()
        pred = _extract_mc_answer(response)
        return pred == gt and pred != ""

    def max_tokens(self) -> int:
        return 16


# ---------------------------------------------------------------------------
# HellaSwag
# ---------------------------------------------------------------------------

class HellaSwagScorer(BaseScorer):

    def build_prompt(self, question: dict) -> str:
        ctx = question["ctx"]
        endings = question["endings"]
        choices = [f"...{e}" for e in endings[:4]]
        return _build_mc_prompt(
            f"Choose the most plausible continuation:\n\n{ctx}",
            choices,
        )

    def score(self, question: dict, response: str) -> bool:
        gt_idx = int(question["answer"])
        gt_letter = ["A", "B", "C", "D"][gt_idx]
        pred = _extract_mc_answer(response)
        return pred == gt_letter and pred != ""

    def max_tokens(self) -> int:
        return 16


# ---------------------------------------------------------------------------
# TruthfulQA
# ---------------------------------------------------------------------------

class TruthfulQAScorer(BaseScorer):

    def build_prompt(self, question: dict) -> str:
        choices = question["choices"][:4]
        return _build_mc_prompt(question["question"], choices)

    def score(self, question: dict, response: str) -> bool:
        gt = str(question["answer"]).strip().upper()
        # TruthfulQA answers are sometimes stored as numeric index
        if gt.isdigit():
            gt = ["A", "B", "C", "D"][int(gt)]
        pred = _extract_mc_answer(response)
        return pred == gt and pred != ""

    def max_tokens(self) -> int:
        return 16


# ---------------------------------------------------------------------------
# IFEval
# ---------------------------------------------------------------------------

class IFEvalScorer(BaseScorer):
    """Instruction-following evaluation via lightweight pattern checks."""

    def build_prompt(self, question: dict) -> str:
        return question["prompt"]

    def score(self, question: dict, response: str) -> bool:
        # Use pre-baked check_fn if present (mock data)
        check_fn = question.get("check_fn")
        if check_fn is not None:
            try:
                return bool(check_fn(response))
            except Exception:
                return False

        # For real IFEval data: apply instruction-type checks
        instruction_ids = question.get("instruction_id_list", [])
        kwargs_list = question.get("kwargs", [])
        if not instruction_ids:
            return len(response.strip()) > 0

        results = []
        for instr_id, kwargs in zip(instruction_ids, kwargs_list or [{}]):
            results.append(self._check_instruction(response, instr_id, kwargs or {}))

        # Strict: ALL instructions must be followed
        return all(results)

    @staticmethod
    def _check_instruction(response: str, instruction_id: str, kwargs: dict) -> bool:
        r = response.strip()
        if "forbidden_words" in instruction_id or "forbidden" in instruction_id:
            forbidden = kwargs.get("forbidden_words", [])
            return not any(w.lower() in r.lower() for w in forbidden)

        if "keyword" in instruction_id and "existence" in instruction_id:
            kw = kwargs.get("keyword", "")
            return kw.lower() in r.lower()

        if "english_capital" in instruction_id:
            return r == r.upper() and len(r) > 0

        if "number_sentences" in instruction_id:
            n_req = kwargs.get("num_sentences", 1)
            sentences = [s for s in re.split(r"[.!?]+", r) if s.strip()]
            return abs(len(sentences) - n_req) <= 1

        if "number_paragraphs" in instruction_id:
            n_req = kwargs.get("num_paragraphs", 1)
            paras = [p for p in r.split("\n\n") if p.strip()]
            return len(paras) >= n_req

        if "json" in instruction_id.lower():
            import json
            try:
                json.loads(r)
                return True
            except (json.JSONDecodeError, ValueError):
                return False

        # Default: non-empty response
        return len(r) > 10

    def max_tokens(self) -> int:
        return 256


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_SCORERS: dict[str, BaseScorer] = {
    "gsm8k": GSM8KScorer(),
    "arc_challenge": ARCScorer(),
    "mmlu": MMLUScorer(),
    "hellaswag": HellaSwagScorer(),
    "truthfulqa": TruthfulQAScorer(),
    "ifeval": IFEvalScorer(),
}


def get_scorer(benchmark_id: str) -> BaseScorer:
    """Return the scorer for the given benchmark ID."""
    if benchmark_id not in _SCORERS:
        raise ValueError(f"Unknown benchmark: {benchmark_id}. Valid: {list(_SCORERS)}")
    return _SCORERS[benchmark_id]
