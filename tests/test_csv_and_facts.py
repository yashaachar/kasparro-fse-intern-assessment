"""Tests for CSV parsing, value conversions, and fact catalogue integrity."""

import csv
import io
import pytest

from src.fact_tool import BrandFactsCatalogue, parse_typed_value


def test_csv_parser_embedded_newlines_and_quoted_commas():
    """Verify that Python CSV parser preserves quoted commas and embedded newlines without corruption."""
    sample_csv = (
        'answer_id,split,brand_id,question,ai_answer\n'
        'T01,development,cleartrail,"What about pricing, features?","First line.\n\nSecond line with, comma."'
    )
    reader = csv.DictReader(io.StringIO(sample_csv))
    rows = list(reader)
    assert len(rows) == 1
    row = rows[0]
    assert row["answer_id"] == "T01"
    assert row["question"] == "What about pricing, features?"
    assert row["ai_answer"] == "First line.\n\nSecond line with, comma."
    # Embedded newline is preserved
    assert "\n\n" in row["ai_answer"]


def test_typed_value_parsing():
    """Verify boolean, number, and string parsing per specification."""
    # Numbers
    assert parse_typed_value("1299", "number") == 1299
    assert parse_typed_value("350.5", "number") == 350.5

    # Booleans: must be strict, never bool('false')
    assert parse_typed_value("true", "boolean") is True
    assert parse_typed_value("false", "boolean") is False
    assert parse_typed_value("True", "boolean") is True
    assert parse_typed_value("False", "boolean") is False
    with pytest.raises(ValueError):
        parse_typed_value("not_a_bool", "boolean")

    # Strings
    assert parse_typed_value("ClearTrail Mini", "string") == "ClearTrail Mini"


def test_catalogue_loading(sample_catalogue):
    """Verify catalogue loaded all 132 facts across 3 brands."""
    assert sample_catalogue.fact_version == "speaker-catalogue-v4"
    assert len(sample_catalogue.fact_id_map) == 132
    assert set(sample_catalogue.facts_by_brand.keys()) == {"cleartrail", "stonewave", "cloudnote"}

    # Test valid fact retrieval
    res = sample_catalogue.get_brand_facts("cleartrail")
    assert res["brand_id"] == "cleartrail"
    assert res["fact_version"] == "speaker-catalogue-v4"
    assert len(res["facts"]) > 0


def test_catalogue_unknown_brand_rejection(sample_catalogue):
    """Verify unknown brand is rejected with ValueError."""
    with pytest.raises(ValueError, match="Unknown brand_id"):
        sample_catalogue.get_brand_facts("unknown_speaker_brand")


def test_catalogue_illegal_tool_rejection(sample_catalogue):
    """Verify execution of arbitrary tool names is rejected."""
    with pytest.raises(ValueError, match="Illegal tool"):
        sample_catalogue.execute_tool("execute_bash_command", {"command": "dir"})

    with pytest.raises(ValueError, match="Illegal tool"):
        sample_catalogue.execute_tool("format_hard_drive", {})
