"""Deterministic evaluation harness module for KASPARRO FSE Intern Pathway Assessment.

Implements exact span matching and calculates extraction precision, recall, F1,
joint precision, recall, F1, coverage, missed gold claims, extra predictions,
and wrong labels. Ensures final-check splits have null quality metrics.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from src.models import AnswerResult, EvaluationMetrics, GoldClaim, PredictedClaim


EVALUATOR_VERSION = "sentence-spans-and-labels-v4"


def calculate_metrics_from_counts(
    G: int,
    P: int,
    M: int,
    C: int,
    missed_gold_ids: List[str],
    unmatched_prediction_ids: List[str],
    matched_wrong_label_ids: List[Dict[str, str]],
    error_count: int,
) -> EvaluationMetrics:
    """Calculate extraction and joint metrics from raw counts.

    Handles zero-denominators as specified: return 0.0 when split has gold claims.
    """
    if G == 0:
        return EvaluationMetrics(
            G=0,
            P=P,
            M=0,
            C=0,
            extraction_precision=None,
            extraction_recall=None,
            extraction_f1=None,
            joint_precision=None,
            joint_recall=None,
            joint_f1=None,
            coverage=None,
            missed_gold_ids=missed_gold_ids,
            unmatched_prediction_ids=unmatched_prediction_ids,
            matched_wrong_label_ids=matched_wrong_label_ids,
            error_count=error_count,
        )

    ext_p = (M / P) if P > 0 else 0.0
    ext_r = (M / G) if G > 0 else 0.0
    ext_f1 = (2 * M / (P + G)) if (P + G) > 0 else 0.0

    joint_p = (C / P) if P > 0 else 0.0
    joint_r = (C / G) if G > 0 else 0.0
    joint_f1 = (2 * C / (P + G)) if (P + G) > 0 else 0.0

    coverage = (C / G) if G > 0 else 0.0

    return EvaluationMetrics(
        G=G,
        P=P,
        M=M,
        C=C,
        extraction_precision=round(ext_p, 4),
        extraction_recall=round(ext_r, 4),
        extraction_f1=round(ext_f1, 4),
        joint_precision=round(joint_p, 4),
        joint_recall=round(joint_r, 4),
        joint_f1=round(joint_f1, 4),
        coverage=round(coverage, 4),
        missed_gold_ids=missed_gold_ids,
        unmatched_prediction_ids=unmatched_prediction_ids,
        matched_wrong_label_ids=matched_wrong_label_ids,
        error_count=error_count,
    )


class Evaluator:
    """Deterministic evaluator for development and check splits."""

    def __init__(self, gold_csv_path: str | Path = "development_gold.csv") -> None:
        self.gold_csv_path = Path(gold_csv_path)
        self.evaluator_version = EVALUATOR_VERSION
        self.gold_claims_by_answer: Dict[str, List[GoldClaim]] = {}
        self.all_gold_claims: Dict[str, GoldClaim] = {}
        self._load_gold()

    def _load_gold(self) -> None:
        if not self.gold_csv_path.exists():
            # Gold might not exist in check-only environments
            return

        with open(self.gold_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                evidence = [x.strip() for x in row.get("evidence_fact_ids", "").split("|") if x.strip()]
                tags = [x.strip() for x in row.get("difficulty_tags", "").split("|") if x.strip()]
                gold = GoldClaim(
                    answer_id=row["answer_id"].strip(),
                    claim_id=row["claim_id"].strip(),
                    start_char=int(row["start_char"]),
                    end_char=int(row["end_char"]),
                    quote=row["quote"],
                    expected_label=row["expected_label"].strip().lower(),
                    evidence_fact_ids=evidence,
                    explanation=row.get("explanation", "").strip(),
                    difficulty_tags=tags,
                )
                self.gold_claims_by_answer.setdefault(gold.answer_id, []).append(gold)
                self.all_gold_claims[gold.claim_id] = gold

    def evaluate_development_split(
        self,
        answer_results: Sequence[AnswerResult],
    ) -> EvaluationMetrics:
        """Score predictions against development gold claims across all scheduled answers.

        An answer with an error adds 0 accepted predictions, while its reference
        claims remain in G as missed.
        """
        all_gold_ids_scheduled: Set[str] = set()
        matched_gold_ids: Set[str] = set()
        accepted_predictions: List[Tuple[str, PredictedClaim]] = []

        unmatched_pred_ids: List[str] = []
        matched_wrong_label: List[Dict[str, str]] = []
        error_count = 0

        M = 0  # Exact span matches
        C = 0  # Exact span matches with correct label

        for ans in answer_results:
            aid = ans.answer_id
            golds = self.gold_claims_by_answer.get(aid, [])
            for g in golds:
                all_gold_ids_scheduled.add(g.claim_id)

            if ans.status != "success":
                error_count += 1
                continue

            for claim in ans.claims:
                accepted_predictions.append((aid, claim))

                # Look for matching gold claim in this answer
                matched_gold: Optional[GoldClaim] = None
                for g in golds:
                    if g.claim_id in matched_gold_ids:
                        continue
                    # Exact span matching: answer_id, start_char, end_char identical, quote matches
                    if (
                        g.start_char == claim.start_char
                        and g.end_char == claim.end_char
                        and g.quote == claim.quote
                    ):
                        matched_gold = g
                        break

                if matched_gold is not None:
                    M += 1
                    matched_gold_ids.add(matched_gold.claim_id)
                    if claim.label == matched_gold.expected_label:
                        C += 1
                    else:
                        matched_wrong_label.append(
                            {
                                "prediction_id": claim.claim_id,
                                "gold_id": matched_gold.claim_id,
                                "predicted_label": claim.label,
                                "expected_label": matched_gold.expected_label,
                            }
                        )
                else:
                    unmatched_pred_ids.append(claim.claim_id)

        G = len(all_gold_ids_scheduled)
        P = len(accepted_predictions)
        missed_gold_ids = sorted(all_gold_ids_scheduled - matched_gold_ids)

        return calculate_metrics_from_counts(
            G=G,
            P=P,
            M=M,
            C=C,
            missed_gold_ids=missed_gold_ids,
            unmatched_prediction_ids=sorted(unmatched_pred_ids),
            matched_wrong_label_ids=matched_wrong_label,
            error_count=error_count,
        )

    def evaluate_check_split(
        self,
        answer_results: Sequence[AnswerResult],
    ) -> EvaluationMetrics:
        """Evaluate final-check split (A41-A50).

        As mandated by the assessment: G, M, C, matched gold IDs, expected labels,
        and all quality metrics must remain null. Only error counts and prediction
        counts are tracked.
        """
        error_count = sum(1 for a in answer_results if a.status != "success")
        total_p = sum(len(a.claims) for a in answer_results if a.status == "success")

        return EvaluationMetrics(
            G=None,
            P=total_p,
            M=None,
            C=None,
            extraction_precision=None,
            extraction_recall=None,
            extraction_f1=None,
            joint_precision=None,
            joint_recall=None,
            joint_f1=None,
            coverage=None,
            missed_gold_ids=None,
            unmatched_prediction_ids=None,
            matched_wrong_label_ids=None,
            error_count=error_count,
        )
