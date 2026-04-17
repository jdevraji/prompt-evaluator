"""Combines heuristic and LLM scores into a final verdict."""

from typing import Optional
from evaluator.models import FinalVerdict, HeuristicResult, LLMResult


# Weights for combining scores
HEURISTIC_WEIGHT = 0.4
LLM_WEIGHT = 0.6


def combine(
    heuristic: HeuristicResult,
    llm: Optional[LLMResult],
) -> FinalVerdict:
    """
    Combine heuristic and LLM results into a final verdict.

    Weighting:
    - With LLM: 40% heuristic + 60% LLM
    - Without LLM: 100% heuristic
    """
    # Normalize heuristic scores to 0-10 scale (they're summed across 6 dimensions, max 60)
    h_norm_a = heuristic.total_a / 6.0
    h_norm_b = heuristic.total_b / 6.0

    if llm is not None:
        # Weighted combination
        final_a = (h_norm_a * HEURISTIC_WEIGHT) + (llm.score_a * LLM_WEIGHT)
        final_b = (h_norm_b * HEURISTIC_WEIGHT) + (llm.score_b * LLM_WEIGHT)

        # Build reasoning from both sources
        reasons = []
        if heuristic.winner == llm.winner:
            reasons.append(
                f"Both heuristic analysis and LLM evaluation agree: "
                f"Response {heuristic.winner} is stronger."
            )
        else:
            reasons.append(
                f"Heuristic analysis favors Response {heuristic.winner}, "
                f"while LLM evaluation favors Response {llm.winner}. "
                f"The weighted combination determines the final winner."
            )

        reasons.append(f"Heuristic scores: A={heuristic.total_a:.1f}/60, B={heuristic.total_b:.1f}/60")
        reasons.append(f"LLM scores: A={llm.score_a:.1f}/10, B={llm.score_b:.1f}/10")

        if llm.reasoning:
            reasons.append(f"LLM reasoning: {llm.reasoning}")

        reasoning = "\n".join(reasons)
    else:
        # Heuristic only
        final_a = h_norm_a
        final_b = h_norm_b
        reasoning = (
            f"Evaluation based on heuristic analysis only (LLM unavailable).\n"
            f"Heuristic scores: A={heuristic.total_a:.1f}/60, B={heuristic.total_b:.1f}/60\n"
            f"{heuristic.summary}"
        )

    # Determine winner
    diff = abs(final_a - final_b)
    if diff < 0.3:
        winner = "TIE"
    elif final_a > final_b:
        winner = "A"
    else:
        winner = "B"

    # Confidence based on score gap (0-100)
    max_possible_diff = 10.0  # maximum possible score difference
    confidence = min((diff / max_possible_diff) * 100, 100.0)
    if winner == "TIE":
        confidence = 100.0 - confidence  # high confidence in tie = small gap

    return FinalVerdict(
        winner=winner,
        confidence=round(confidence, 1),
        heuristic_result=heuristic,
        llm_result=llm,
        reasoning=reasoning,
        score_a=round(final_a, 2),
        score_b=round(final_b, 2),
    )
