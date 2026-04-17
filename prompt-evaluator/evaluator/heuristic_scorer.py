"""Heuristic scoring engine — evaluates responses using local rules, no API needed."""

import re
import string
from evaluator.models import DimensionScore, HeuristicResult, EvaluationInput


# Common vague/filler words that reduce specificity
FILLER_WORDS = {
    "very", "really", "quite", "basically", "actually", "just", "simply",
    "stuff", "things", "something", "somehow", "somewhat", "kind of",
    "sort of", "a lot", "many", "various", "several", "numerous",
    "important", "interesting", "nice", "good", "great", "amazing",
}

# Transition words that indicate logical flow
TRANSITION_WORDS = {
    "however", "therefore", "furthermore", "moreover", "additionally",
    "consequently", "nevertheless", "meanwhile", "specifically",
    "for example", "for instance", "in contrast", "on the other hand",
    "as a result", "in conclusion", "first", "second", "third",
    "finally", "next", "then", "also", "because", "since", "although",
    "while", "whereas", "similarly", "likewise", "in addition",
}


def _word_count(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def _sentence_count(text: str) -> int:
    """Count sentences in text."""
    sentences = re.split(r'[.!?]+', text)
    return len([s for s in sentences if s.strip()])


def _extract_prompt_keywords(prompt: str) -> set[str]:
    """Extract meaningful keywords from the prompt (words > 3 chars, excluding stopwords)."""
    stopwords = {
        "what", "how", "does", "this", "that", "with", "from", "your",
        "about", "would", "could", "should", "will", "have", "been",
        "they", "them", "their", "there", "where", "when", "which",
        "were", "into", "than", "then", "also", "more", "some", "such",
        "only", "other", "most", "each", "both", "between", "through",
        "after", "before", "during", "without", "again", "further",
        "here", "once", "does", "doing", "being", "having",
        "the", "and", "for", "are", "but", "not", "you", "all",
        "can", "her", "was", "one", "our", "out",
    }
    words = re.findall(r'\b[a-z]+\b', prompt.lower())
    return {w for w in words if len(w) > 3 and w not in stopwords}


def _score_length_detail(response: str, prompt: str) -> tuple[float, str]:
    """Score based on response length relative to prompt complexity."""
    wc = _word_count(response)
    prompt_wc = _word_count(prompt)

    # Heuristic: a good response should be 3-20x the prompt length
    ratio = wc / max(prompt_wc, 1)

    if wc < 10:
        return 1.0, f"Very short ({wc} words) — likely insufficient detail"
    elif ratio < 1.5:
        return 3.0, f"Brief ({wc} words) — may lack sufficient detail"
    elif ratio < 3:
        return 5.0, f"Moderate length ({wc} words) — acceptable detail level"
    elif ratio < 8:
        return 8.0, f"Good length ({wc} words) — detailed and thorough"
    elif ratio < 20:
        return 9.0, f"Comprehensive ({wc} words) — very detailed"
    else:
        return 6.0, f"Very long ({wc} words) — possibly too verbose"


def _score_structure(response: str) -> tuple[float, str]:
    """Score based on structural elements (lists, headers, paragraphs)."""
    score = 4.0  # baseline
    reasons = []

    # Check for bullet points or numbered lists
    bullet_matches = re.findall(r'^\s*[-•*]\s+', response, re.MULTILINE)
    numbered_matches = re.findall(r'^\s*\d+[.)]\s+', response, re.MULTILINE)

    if bullet_matches:
        score += 1.5
        reasons.append(f"{len(bullet_matches)} bullet points")
    if numbered_matches:
        score += 2.0
        reasons.append(f"{len(numbered_matches)} numbered items")

    # Check for paragraphs (double newlines)
    paragraphs = len(re.split(r'\n\s*\n', response.strip()))
    if paragraphs >= 3:
        score += 1.5
        reasons.append(f"{paragraphs} paragraphs")
    elif paragraphs >= 2:
        score += 0.5
        reasons.append(f"{paragraphs} paragraphs")

    # Check for headers (markdown)
    headers = re.findall(r'^#+\s+', response, re.MULTILINE)
    if headers:
        score += 1.0
        reasons.append(f"{len(headers)} headers")

    # Check for bold/emphasis (markdown)
    bold = re.findall(r'\*\*[^*]+\*\*', response)
    if bold:
        score += 0.5
        reasons.append(f"{len(bold)} bold elements")

    score = min(score, 10.0)
    explanation = f"Structure: {', '.join(reasons)}" if reasons else "Minimal structure — plain text block"
    return score, explanation


def _score_keyword_relevance(response: str, prompt: str) -> tuple[float, str]:
    """Score how many prompt keywords appear in the response."""
    keywords = _extract_prompt_keywords(prompt)
    if not keywords:
        return 5.0, "No extractable keywords from prompt"

    response_lower = response.lower()
    found = {kw for kw in keywords if kw in response_lower}
    ratio = len(found) / len(keywords)

    score = min(ratio * 10, 10.0)
    missing = keywords - found
    explanation = f"Matched {len(found)}/{len(keywords)} prompt keywords"
    if missing and len(missing) <= 5:
        explanation += f" (missing: {', '.join(sorted(missing)[:5])})"

    return round(score, 1), explanation


def _score_formatting(response: str) -> tuple[float, str]:
    """Score formatting quality: punctuation, capitalization, code blocks."""
    score = 5.0
    reasons = []

    # Check for code blocks
    code_blocks = re.findall(r'```[\s\S]*?```', response)
    inline_code = re.findall(r'`[^`]+`', response)
    if code_blocks:
        score += 1.5
        reasons.append(f"{len(code_blocks)} code blocks")
    if inline_code:
        score += 0.5
        reasons.append(f"{len(inline_code)} inline code spans")

    # Check sentence capitalization
    sentences = re.split(r'[.!?]\s+', response)
    capitalized = sum(1 for s in sentences if s and s[0].isupper())
    if sentences and capitalized / len(sentences) > 0.8:
        score += 1.0
        reasons.append("good capitalization")

    # Check for proper ending punctuation
    stripped = response.strip()
    if stripped and stripped[-1] in '.!?':
        score += 0.5
        reasons.append("proper ending punctuation")

    # Penalize ALL CAPS segments
    caps_segments = re.findall(r'\b[A-Z]{5,}\b', response)
    if caps_segments:
        score -= 1.0
        reasons.append(f"{len(caps_segments)} ALL-CAPS words")

    score = max(min(score, 10.0), 0.0)
    explanation = f"Formatting: {', '.join(reasons)}" if reasons else "Basic formatting"
    return round(score, 1), explanation


def _score_specificity(response: str) -> tuple[float, str]:
    """Score specificity — penalize vague filler, reward concrete language."""
    words = re.findall(r'\b[a-z]+\b', response.lower())
    if not words:
        return 0.0, "Empty response"

    filler_count = sum(1 for w in words if w in FILLER_WORDS)
    filler_ratio = filler_count / len(words)

    # Check for numbers, percentages, specific terms
    numbers = re.findall(r'\b\d+\.?\d*%?\b', response)
    has_numbers = len(numbers)

    score = 7.0 - (filler_ratio * 20)  # penalize high filler ratio
    if has_numbers:
        score += min(has_numbers * 0.3, 2.0)

    score = max(min(score, 10.0), 1.0)
    explanation = f"{filler_count} filler words out of {len(words)} total"
    if has_numbers:
        explanation += f", {has_numbers} specific numbers/values"

    return round(score, 1), explanation


def _score_coherence(response: str) -> tuple[float, str]:
    """Score logical flow using transition words and sentence structure."""
    response_lower = response.lower()
    found_transitions = {t for t in TRANSITION_WORDS if t in response_lower}

    sentences = _sentence_count(response)
    words = _word_count(response)

    # Average sentence length (ideal: 15-25 words)
    avg_sentence_len = words / max(sentences, 1)

    score = 5.0
    reasons = []

    # Transition words
    if len(found_transitions) >= 5:
        score += 2.5
        reasons.append(f"{len(found_transitions)} transition words")
    elif len(found_transitions) >= 3:
        score += 1.5
        reasons.append(f"{len(found_transitions)} transition words")
    elif len(found_transitions) >= 1:
        score += 0.5
        reasons.append(f"{len(found_transitions)} transition words")
    else:
        reasons.append("no transition words")

    # Sentence length variation
    if 12 <= avg_sentence_len <= 28:
        score += 1.5
        reasons.append(f"good avg sentence length ({avg_sentence_len:.0f} words)")
    elif avg_sentence_len > 40:
        score -= 1.0
        reasons.append(f"very long sentences ({avg_sentence_len:.0f} words avg)")

    score = max(min(score, 10.0), 1.0)
    return round(score, 1), f"Coherence: {', '.join(reasons)}"


def score(evaluation_input: EvaluationInput) -> HeuristicResult:
    """Run all heuristic scoring dimensions on both responses."""
    prompt = evaluation_input.prompt
    response_a = evaluation_input.response_a
    response_b = evaluation_input.response_b

    scorers = [
        ("Length & Detail", _score_length_detail, True),
        ("Structure", _score_structure, False),
        ("Keyword Relevance", _score_keyword_relevance, True),
        ("Formatting", _score_formatting, False),
        ("Specificity", _score_specificity, False),
        ("Coherence", _score_coherence, False),
    ]

    dimensions = []
    total_a = 0.0
    total_b = 0.0

    for name, scorer_fn, needs_prompt in scorers:
        if needs_prompt:
            sa, ea = scorer_fn(response_a, prompt)
            sb, eb = scorer_fn(response_b, prompt)
        else:
            sa, ea = scorer_fn(response_a)
            sb, eb = scorer_fn(response_b)

        # Build explanation showing both sides
        explanation = f"A: {ea} | B: {eb}"
        dimensions.append(DimensionScore(
            dimension=name,
            score_a=sa,
            score_b=sb,
            explanation=explanation,
        ))
        total_a += sa
        total_b += sb

    if total_a > total_b + 1:
        winner = "A"
    elif total_b > total_a + 1:
        winner = "B"
    else:
        winner = "TIE"

    diff = abs(total_a - total_b)
    summary = (
        f"Heuristic analysis: Response {winner} scores higher "
        f"({total_a:.1f} vs {total_b:.1f}, Δ={diff:.1f})"
        if winner != "TIE"
        else f"Heuristic analysis: Responses are closely matched ({total_a:.1f} vs {total_b:.1f})"
    )

    return HeuristicResult(
        dimensions=dimensions,
        total_a=round(total_a, 1),
        total_b=round(total_b, 1),
        winner=winner,
        summary=summary,
    )
