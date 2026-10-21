"""Tests for deterministic scoring arithmetic, exact span matching, and error accounting."""

import pytest

from src.evaluator import Evaluator, calculate_metrics_from_counts
from src.models import AnswerResult, GoldClaim, PredictedClaim


def test_required_arithmetic_example():
    """Verify exact specification arithmetic example: G=10, P=10, M=8, C=7."""
    metrics = calculate_metrics_from_counts(
        G=10,
        P=10,
        M=8,
        C=7,
        missed_gold_ids=["G01", "G02"],
        unmatched_prediction_ids=["P01", "P02"],
        matched_wrong_label_ids=[{"pred": "P03", "gold": "G03"}],
        error_count=0,
    )

    assert metrics.G == 10
    assert metrics.P == 10
    assert metrics.M == 8
    assert metrics.C == 7
    assert metrics.extraction_precision == 0.80
    assert metrics.extraction_recall == 0.80
    assert metrics.extraction_f1 == 0.80
    assert metrics.joint_precision == 0.70
    assert metrics.joint_recall == 0.70
    assert metrics.joint_f1 == 0.70
    assert metrics.coverage == 0.70


def test_perfect_evaluation():
    """Verify perfect output gives 1.0 on all metrics."""
    metrics = calculate_metrics_from_counts(
        G=5,
        P=5,
        M=5,
        C=5,
        missed_gold_ids=[],
        unmatched_prediction_ids=[],
        matched_wrong_label_ids=[],
        error_count=0,
    )
    assert metrics.extraction_precision == 1.0
    assert metrics.extraction_recall == 1.0
    assert metrics.extraction_f1 == 1.0
    assert metrics.joint_precision == 1.0
    assert metrics.joint_recall == 1.0
    assert metrics.joint_f1 == 1.0
    assert metrics.coverage == 1.0


def test_extra_claims_reduce_precision():
    """Verify extra predictions (e.g. advice mistakenly extracted) increase P and reduce precision."""
    metrics = calculate_metrics_from_counts(
        G=4,
        P=8,
        M=4,
        C=4,
        missed_gold_ids=[],
        unmatched_prediction_ids=["P05", "P06", "P07", "P08"],
        matched_wrong_label_ids=[],
        error_count=0,
    )
    # Ext precision = 4/8 = 0.5, recall = 4/4 = 1.0, F1 = 2*4/12 = 0.6667
    assert metrics.extraction_precision == 0.50
    assert metrics.extraction_recall == 1.00
    assert metrics.extraction_f1 == 0.6667
    assert metrics.joint_precision == 0.50
    assert metrics.joint_recall == 1.00


def test_wrong_labels_reduce_joint_f1():
    """Verify incorrect labels leave extraction F1 intact while reducing joint F1."""
    metrics = calculate_metrics_from_counts(
        G=4,
        P=4,
        M=4,
        C=2,
        missed_gold_ids=[],
        unmatched_prediction_ids=[],
        matched_wrong_label_ids=[{"pred": "P1", "gold": "G1"}, {"pred": "P2", "gold": "G2"}],
        error_count=0,
    )
    assert metrics.extraction_f1 == 1.0
    assert metrics.joint_f1 == 0.50
    assert metrics.coverage == 0.50


def test_answer_error_accounting():
    """Verify answer errors add 0 accepted predictions while gold claims remain in G as missed."""
    evaluator = Evaluator()
    # Mock gold claims for answers T01 and T02
    evaluator.gold_claims_by_answer = {
        "T01": [
            GoldClaim(
                answer_id="T01",
                claim_id="T01:0:10",
                start_char=0,
                end_char=10,
                quote="Quote 010.",
                expected_label="supported",
            )
        ],
        "T02": [
            GoldClaim(
                answer_id="T02",
                claim_id="T02:0:10",
                start_char=0,
                end_char=10,
                quote="Quote 020.",
                expected_label="supported",
            )
        ],
    }

    # T01 succeeded, T02 failed with provider error
    ans1 = AnswerResult(
        answer_id="T01",
        status="success",
        claims=[
            PredictedClaim(
                claim_id="T01:0:10",
                start_char=0,
                end_char=10,
                quote="Quote 010.",
                label="supported",
            )
        ],
    )
    ans2 = AnswerResult(
        answer_id="T02",
        status="error",
        error_message="Provider timeout",
        claims=[],
    )

    metrics = evaluator.evaluate_development_split([ans1, ans2])
    assert metrics.G == 2
    assert metrics.P == 1
    assert metrics.M == 1
    assert metrics.C == 1
    assert metrics.error_count == 1
    assert metrics.missed_gold_ids == ["T02:0:10"]
    assert metrics.extraction_recall == 0.50
    assert metrics.joint_recall == 0.50


def test_check_split_metrics_null():
    """Verify check split (A41-A50) keeps G, M, C, and all quality metrics null."""
    evaluator = Evaluator()
    ans1 = AnswerResult(
        answer_id="A41",
        status="success",
        claims=[
            PredictedClaim(
                claim_id="A41:0:10",
                start_char=0,
                end_char=10,
                quote="Quote 010.",
                label="supported",
            )
        ],
    )
    metrics = evaluator.evaluate_check_split([ans1])
    assert metrics.G is None
    assert metrics.M is None
    assert metrics.C is None
    assert metrics.extraction_f1 is None
    assert metrics.joint_f1 is None
    assert metrics.P == 1
    assert metrics.error_count == 0
