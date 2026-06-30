from __future__ import annotations

from datetime import datetime
from typing import Callable


def calculate_confidence(
    chunk,
    avg_relevance,
    query=None,
    *,
    has_factual_data: Callable[[str], bool],
):
    """
    Calculate confidence score for a chunk based on relevance, freshness,
    factual density, source position, and text substance.

    The source-domain whitelist was intentionally removed to avoid
    hardcoded trust assumptions in library code.
    """
    score = 0
    now = datetime.now()

    rel = chunk["relevance"]
    if rel > 0.7:
        score += 3
    elif rel > 0.4:
        score += 2
    elif rel > 0.2:
        score += 1

    if avg_relevance > 0 and rel > avg_relevance * 1.3:
        score += 1

    if chunk.get("pub_date"):
        pub_date = chunk["pub_date"]
        age_hours = (now - pub_date).total_seconds() / 3600

        if age_hours < 24:
            score += 3
        elif age_hours < 24 * 7:
            score += 2
        elif age_hours < 24 * 30:
            score += 1
    else:
        score += 1

    if has_factual_data(chunk["text"]):
        score += 2

    if chunk["source_idx"] < 3:
        score += 1
    elif chunk["source_idx"] < 6:
        score += 0.5

    word_count = len(chunk["text"].split())
    if word_count > 100:
        score += 1
    elif word_count > 50:
        score += 0.5

    if score >= 7:
        return "HIGH"
    if score >= 4:
        return "MEDIUM"
    return "LOW"
