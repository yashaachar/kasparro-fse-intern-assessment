"""Validation and span offset location module for KASPARRO FSE Intern Pathway Assessment.

Locates exact character offsets for extracted quotes, verifies character slice equality,
validates label enums, ensures catalogue fact ID existence, checks sentence count in reasons,
and strictly enforces non-overlapping order and uniqueness.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Set

from src.models import PredictedClaim, RawExtractedClaim, VALID_LABELS


def sanitize_reason(reason: str) -> str:
    """Ensure reason is at most two sentences without breaking numbers."""
    trimmed = reason.strip()
    if not trimmed:
        return ""
    # Split on sentence terminals followed by whitespace
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", trimmed) if s.strip()]
    if len(sentences) <= 2:
        return " ".join(sentences)
    return " ".join(sentences[:2])


def locate_quote_offset(ai_answer: str, quote: str, search_start: int = 0) -> tuple[int, int]:
    """Locate exact start_char and end_char of quote in ai_answer.

    Resolves from source context starting at search_start when possible,
    falling back to full string search if out-of-order.
    Raises ValueError if quote is not found.
    """
    clean_quote = quote.strip()
    if not clean_quote:
        raise ValueError("Cannot locate empty quote in text")

    # Try searching from current monotonic pointer
    pos = ai_answer.find(clean_quote, search_start)
    if pos == -1:
        # Fall back to searching anywhere in the text
        pos = ai_answer.find(clean_quote, 0)

    if pos == -1:
        # Quote was altered or hallucinated
        snippet = clean_quote[:60] + ("..." if len(clean_quote) > 60 else "")
        raise ValueError(f"Extracted quote not found in ai_answer: {snippet!r}")

    start_char = pos
    end_char = pos + len(clean_quote)
    return start_char, end_char


def validate_and_build_claims(
    answer_id: str,
    ai_answer: str,
    raw_claims: Sequence[RawExtractedClaim | Dict[str, Any]],
    valid_fact_ids: Set[str],
) -> List[PredictedClaim]:
    """Convert raw extracted claims to validated PredictedClaims with exact offsets.

    Validates:
    - Exact character span bounds (0 <= start_char < end_char <= len(ai_answer))
    - Exact quote equality: ai_answer[start_char:end_char] == quote
    - No leading or trailing whitespace in quote
    - Derived claim_id: f"{answer_id}:{start_char}:{end_char}"
    - Unique claim IDs
    - Strictly non-overlapping spans sorted in increasing start_char order
    - Valid label in {"supported", "contradicted", "unknown"}
    - Fact IDs all exist in the catalogue
    - Reason is at most two sentences

    Raises ValueError on any validation violation.
    """
    if not raw_claims:
        return []

    unvalidated: List[PredictedClaim] = []
    current_search_idx = 0

    for item in raw_claims:
        if isinstance(item, dict):
            raw = RawExtractedClaim(**item)
        else:
            raw = item

        clean_quote = raw.quote.strip()
        if not clean_quote:
            raise ValueError("Claim quote cannot be empty")

        # Locate exact offsets in ai_answer
        start_char, end_char = locate_quote_offset(ai_answer, clean_quote, current_search_idx)
        current_search_idx = end_char

        # Span bounds check
        if not (0 <= start_char < end_char <= len(ai_answer)):
            raise ValueError(f"Span offsets ({start_char}, {end_char}) out of bounds for answer length {len(ai_answer)}")

        # Exact slice equality check
        actual_slice = ai_answer[start_char:end_char]
        if actual_slice != clean_quote:
            raise ValueError(f"Quote mismatch: slice '{actual_slice}' != quote '{clean_quote}'")

        # Label check
        label = raw.label.strip().lower()
        if label not in VALID_LABELS:
            raise ValueError(f"Invalid claim label: '{raw.label}'. Must be one of {VALID_LABELS}")

        # Fact IDs check: all must exist in catalogue
        fact_ids: List[str] = []
        for fid in raw.fact_ids:
            clean_fid = fid.strip()
            if not clean_fid:
                continue
            if clean_fid not in valid_fact_ids:
                raise ValueError(f"Invalid fact ID '{clean_fid}' not found in speaker catalogue")
            fact_ids.append(clean_fid)

        # Reason: at most two sentences
        reason_text = sanitize_reason(raw.reason)

        claim_id = f"{answer_id}:{start_char}:{end_char}"

        unvalidated.append(
            PredictedClaim(
                claim_id=claim_id,
                start_char=start_char,
                end_char=end_char,
                quote=clean_quote,
                label=label,
                fact_ids=fact_ids,
                reason=reason_text,
            )
        )

    # Sort strictly by start_char
    unvalidated.sort(key=lambda c: (c.start_char, c.end_char))

    # Validate uniqueness and non-overlapping order
    seen_ids: Set[str] = set()
    prev_end = -1

    for claim in unvalidated:
        if claim.claim_id in seen_ids:
            raise ValueError(f"Duplicate claim_id: {claim.claim_id}")
        seen_ids.add(claim.claim_id)

        if claim.start_char < prev_end:
            raise ValueError(
                f"Overlapping claim span: claim {claim.claim_id} starts at {claim.start_char} "
                f"which is before previous end {prev_end}"
            )
        prev_end = claim.end_char

    return unvalidated
