"""LLM-based scoring engine — uses OpenRouter (DeepSeek R1 Free) for AI evaluation."""

import json
import os
import re
from typing import Optional

from evaluator.models import EvaluationInput, LLMResult


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "deepseek/deepseek-r1:free"

EVALUATION_PROMPT = """You are an expert AI response evaluator. You will be given an original prompt and two candidate responses (A and B). Evaluate which response is better.

## Original Prompt
{prompt}

## Response A
{response_a}

## Response B
{response_b}

## Your Task
Evaluate both responses and return a JSON object with this EXACT structure (no other text, only JSON):

{{
  "score_a": <float 1-10>,
  "score_b": <float 1-10>,
  "strengths_a": ["strength 1", "strength 2"],
  "strengths_b": ["strength 1", "strength 2"],
  "weaknesses_a": ["weakness 1"],
  "weaknesses_b": ["weakness 1"],
  "winner": "A" or "B" or "TIE",
  "reasoning": "One paragraph explaining your decision"
}}

Score on: helpfulness, accuracy, clarity, completeness, and relevance.
Be objective and fair. Return ONLY the JSON object."""


def _parse_llm_response(raw: str) -> Optional[LLMResult]:
    """Parse the LLM's JSON response into an LLMResult."""
    # Try to extract JSON from the response (it might have markdown fences or thinking tags)
    json_match = re.search(r'\{[\s\S]*\}', raw)
    if not json_match:
        return None

    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError:
        return None

    try:
        return LLMResult(
            score_a=float(data.get("score_a", 0)),
            score_b=float(data.get("score_b", 0)),
            strengths_a=data.get("strengths_a", []),
            strengths_b=data.get("strengths_b", []),
            weaknesses_a=data.get("weaknesses_a", []),
            weaknesses_b=data.get("weaknesses_b", []),
            winner=data.get("winner", "TIE"),
            reasoning=data.get("reasoning", ""),
            raw_response=raw,
        )
    except (ValueError, TypeError):
        return None


def score(evaluation_input: EvaluationInput, model_override: Optional[str] = None) -> Optional[LLMResult]:
    """
    Evaluate responses using the LLM via OpenRouter.

    Args:
        evaluation_input: The prompt and two responses to evaluate.
        model_override: Optional model ID to use instead of the default.

    Returns None if the API call fails (missing key, rate limit, network error).
    """
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    selected_model = model_override or MODEL
    if not api_key or api_key == "sk-or-v1-your-key-here":
        return None

    try:
        from openai import OpenAI

        client = OpenAI(
            base_url=OPENROUTER_BASE_URL,
            api_key=api_key,
        )

        prompt_text = EVALUATION_PROMPT.format(
            prompt=evaluation_input.prompt,
            response_a=evaluation_input.response_a,
            response_b=evaluation_input.response_b,
        )

        completion = client.chat.completions.create(
            model=selected_model,
            messages=[
                {"role": "user", "content": prompt_text},
            ],
            temperature=0.3,
            max_tokens=1500,
        )

        raw = completion.choices[0].message.content or ""
        return _parse_llm_response(raw)

    except Exception:
        # Gracefully handle any API errors — caller falls back to heuristic-only
        return None
