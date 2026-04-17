"""Data models for the prompt evaluator."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EvaluationInput:
    """Raw input: the original prompt and two candidate responses."""
    prompt: str
    response_a: str
    response_b: str


@dataclass
class DimensionScore:
    """Score for a single evaluation dimension."""
    dimension: str
    score_a: float  # 0-10
    score_b: float  # 0-10
    explanation: str


@dataclass
class HeuristicResult:
    """Result from the local heuristic scoring engine."""
    dimensions: list[DimensionScore] = field(default_factory=list)
    total_a: float = 0.0
    total_b: float = 0.0
    winner: str = ""  # "A", "B", or "TIE"
    summary: str = ""


@dataclass
class LLMResult:
    """Result from LLM-based evaluation."""
    score_a: float = 0.0
    score_b: float = 0.0
    strengths_a: list[str] = field(default_factory=list)
    strengths_b: list[str] = field(default_factory=list)
    weaknesses_a: list[str] = field(default_factory=list)
    weaknesses_b: list[str] = field(default_factory=list)
    winner: str = ""  # "A", "B", or "TIE"
    reasoning: str = ""
    raw_response: str = ""


@dataclass
class FinalVerdict:
    """Combined final evaluation result."""
    winner: str  # "A", "B", or "TIE"
    confidence: float  # 0-100
    heuristic_result: HeuristicResult
    llm_result: Optional[LLMResult]
    reasoning: str
    score_a: float
    score_b: float
