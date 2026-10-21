"""Tests for offset location, quote equality, span validity, and claim constraints."""

import pytest

from src.models import RawExtractedClaim
from src.validation import locate_quote_offset, sanitize_reason, validate_and_build_claims


def test_exact_offset_location():
    """Verify exact start_char and end_char derivation."""
    text = "The ClearTrail Mini weighs 350 grams. It costs INR 1,299."
    quote1 = "The ClearTrail Mini weighs 350 grams."
    quote2 = "It costs INR 1,299."

    s1, e1 = locate_quote_offset(text, quote1)
    assert s1 == 0
    assert e1 == 37
    assert text[s1:e1] == quote1

    s2, e2 = locate_quote_offset(text, quote2, search_start=e1)
    assert s2 == 38
    assert e2 == 57
    assert text[s2:e2] == quote2


def test_quote_not_found_raises():
    """Verify hallucinated or altered quote raises ValueError."""
    text = "The speaker has a 12 hour battery."
    with pytest.raises(ValueError, match="not found in ai_answer"):
        locate_quote_offset(text, "The speaker has an 18 hour battery.")


def test_validate_and_build_claims_valid(sample_catalogue):
    """Verify successful validation and derivation of claim IDs."""
    text = "ClearTrail Mini is 350g. ClearTrail Plus is 650g."
    valid_ids = sample_catalogue.get_valid_fact_ids()

    raw_claims = [
        RawExtractedClaim(
            quote="ClearTrail Mini is 350g.",
            label="supported",
            fact_ids=["cleartrail.mini.weight"],
            reason="Weight matches catalogue.",
        ),
        RawExtractedClaim(
            quote="ClearTrail Plus is 650g.",
            label="supported",
            fact_ids=["cleartrail.plus.weight"],
            reason="Plus weight matches catalogue.",
        ),
    ]

    claims = validate_and_build_claims("A01", text, raw_claims, valid_ids)
    assert len(claims) == 2
    assert claims[0].claim_id == "A01:0:24"
    assert claims[0].start_char == 0
    assert claims[0].end_char == 24
    assert claims[0].quote == "ClearTrail Mini is 350g."

    assert claims[1].claim_id == "A01:25:49"
    assert claims[1].start_char == 25
    assert claims[1].end_char == 49
    assert claims[1].quote == "ClearTrail Plus is 650g."


def test_overlapping_claims_rejected(sample_catalogue):
    """Verify overlapping spans are rejected."""
    text = "ClearTrail Mini speaker is compact and portable."
    valid_ids = sample_catalogue.get_valid_fact_ids()

    raw_claims = [
        {"quote": "ClearTrail Mini speaker is compact", "label": "supported", "fact_ids": [], "reason": "Reason."},
        {"quote": "speaker is compact and portable.", "label": "unknown", "fact_ids": [], "reason": "Reason."},
    ]

    with pytest.raises(ValueError, match="Overlapping claim span"):
        validate_and_build_claims("A01", text, raw_claims, valid_ids)


def test_duplicate_claims_rejected(sample_catalogue):
    """Verify duplicate claims for the same sentence are rejected."""
    text = "ClearTrail Mini has bluetooth 5.3."
    valid_ids = sample_catalogue.get_valid_fact_ids()

    # Two extractions of the single sentence resulting in the same span
    raw_claims = [
        {"quote": "ClearTrail Mini has bluetooth 5.3.", "label": "supported", "fact_ids": [], "reason": "R1."},
        {"quote": "ClearTrail Mini has bluetooth 5.3.", "label": "supported", "fact_ids": [], "reason": "R2."},
    ]
    with pytest.raises(ValueError, match="Duplicate claim_id"):
        validate_and_build_claims("A01", text, raw_claims, valid_ids)


def test_invalid_label_rejected(sample_catalogue):
    """Verify label outside supported/contradicted/unknown is rejected."""
    text = "ClearTrail Mini has bluetooth 5.3."
    valid_ids = sample_catalogue.get_valid_fact_ids()

    raw_claims = [
        {"quote": "ClearTrail Mini has bluetooth 5.3.", "label": "verified", "fact_ids": [], "reason": "Valid reason."}
    ]

    with pytest.raises(ValueError, match="Invalid claim label"):
        validate_and_build_claims("A01", text, raw_claims, valid_ids)


def test_invalid_fact_id_rejected(sample_catalogue):
    """Verify hallucinated fact IDs are rejected."""
    text = "ClearTrail Mini has bluetooth 5.3."
    valid_ids = sample_catalogue.get_valid_fact_ids()

    raw_claims = [
        {
            "quote": "ClearTrail Mini has bluetooth 5.3.",
            "label": "supported",
            "fact_ids": ["cleartrail.mini.hallucinated_fact_id"],
            "reason": "Valid reason.",
        }
    ]

    with pytest.raises(ValueError, match="Invalid fact ID"):
        validate_and_build_claims("A01", text, raw_claims, valid_ids)


def test_sanitize_reason():
    """Verify reasons are capped at two sentences without breaking numbers."""
    three_sentences = "Sentence one. Sentence two. Sentence three."
    assert sanitize_reason(three_sentences) == "Sentence one. Sentence two."

    number_sentence = "The price is INR 1,299.00 in the 2026 catalogue. That is confirmed."
    assert sanitize_reason(number_sentence) == number_sentence
