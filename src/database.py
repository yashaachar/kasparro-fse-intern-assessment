"""Database access module for KASPARRO FSE Intern Pathway Assessment.

Handles PostgreSQL connection, schema creation, parameterized inserts, and joined retrieval.
Enforces unique constraints, foreign keys with cascade delete, and preserves partial evidence.
"""

from __future__ import annotations

import json
import os
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional
import psycopg
from psycopg.rows import dict_row

DEFAULT_DB_URL = "postgresql://postgres:postgres@localhost:5433/kasparro_eval"


def get_db_url() -> str:
    return os.environ.get("DATABASE_URL", DEFAULT_DB_URL)


@contextmanager
def get_db_connection() -> Generator[psycopg.Connection, None, None]:
    """Provide a transactional database connection."""
    conn = psycopg.connect(get_db_url(), row_factory=dict_row)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Create tables, foreign keys, and unique constraints."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    id UUID PRIMARY KEY,
                    split VARCHAR(32) NOT NULL,
                    agent_version VARCHAR(16) NOT NULL,
                    prompt_version VARCHAR(16) NOT NULL,
                    dataset_version VARCHAR(64) NOT NULL,
                    fact_version VARCHAR(64) NOT NULL,
                    evaluator_version VARCHAR(64) NOT NULL,
                    provider VARCHAR(32) NOT NULL,
                    model VARCHAR(64) NOT NULL,
                    status VARCHAR(32) NOT NULL,
                    settings JSONB,
                    metrics JSONB,
                    answer_count INT DEFAULT 0,
                    accepted_prediction_count INT DEFAULT 0,
                    error_count INT DEFAULT 0,
                    duration_seconds FLOAT DEFAULT 0.0,
                    file_hashes JSONB,
                    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS answer_results (
                    id UUID PRIMARY KEY,
                    run_id UUID NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                    answer_id VARCHAR(16) NOT NULL,
                    status VARCHAR(16) NOT NULL,
                    error_message TEXT,
                    latency_ms FLOAT DEFAULT 0.0,
                    turns_used INT DEFAULT 0,
                    trace_ref TEXT,
                    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT uq_run_answer UNIQUE (run_id, answer_id)
                );

                CREATE TABLE IF NOT EXISTS claim_results (
                    id UUID PRIMARY KEY,
                    run_id UUID NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                    answer_result_id UUID NOT NULL REFERENCES answer_results(id) ON DELETE CASCADE,
                    answer_id VARCHAR(16) NOT NULL,
                    claim_id VARCHAR(64) NOT NULL,
                    start_char INT NOT NULL,
                    end_char INT NOT NULL,
                    quote TEXT NOT NULL,
                    label VARCHAR(16) NOT NULL,
                    fact_ids JSONB NOT NULL,
                    reason TEXT,
                    is_matched BOOLEAN,
                    is_label_correct BOOLEAN,
                    matched_gold_id VARCHAR(64),
                    CONSTRAINT uq_run_answer_span UNIQUE (run_id, answer_id, start_char, end_char)
                );
                """
            )


def save_evaluation_run(
    run_id: str | uuid.UUID,
    split: str,
    agent_version: str,
    prompt_version: str,
    dataset_version: str,
    fact_version: str,
    evaluator_version: str,
    provider: str,
    model: str,
    status: str,
    settings: Dict[str, Any],
    metrics: Optional[Dict[str, Any]],
    answer_results: List[Dict[str, Any]],
    duration_seconds: float,
    file_hashes: Dict[str, str],
) -> None:
    """Persist run metadata, answer outcomes, and accepted claims using parameterized SQL."""
    run_uuid = uuid.UUID(str(run_id))
    answer_count = len(answer_results)
    accepted_claims_count = sum(len(a.get("claims", [])) for a in answer_results if a.get("status") == "success")
    error_count = sum(1 for a in answer_results if a.get("status") != "success")

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO runs (
                    id, split, agent_version, prompt_version, dataset_version,
                    fact_version, evaluator_version, provider, model, status,
                    settings, metrics, answer_count, accepted_prediction_count,
                    error_count, duration_seconds, file_hashes
                ) VALUES (
                    %(id)s, %(split)s, %(agent_version)s, %(prompt_version)s, %(dataset_version)s,
                    %(fact_version)s, %(evaluator_version)s, %(provider)s, %(model)s, %(status)s,
                    %(settings)s, %(metrics)s, %(answer_count)s, %(accepted_prediction_count)s,
                    %(error_count)s, %(duration_seconds)s, %(file_hashes)s
                );
                """,
                {
                    "id": run_uuid,
                    "split": split,
                    "agent_version": agent_version,
                    "prompt_version": prompt_version,
                    "dataset_version": dataset_version,
                    "fact_version": fact_version,
                    "evaluator_version": evaluator_version,
                    "provider": provider,
                    "model": model,
                    "status": status,
                    "settings": json.dumps(settings),
                    "metrics": json.dumps(metrics) if metrics is not None else None,
                    "answer_count": answer_count,
                    "accepted_prediction_count": accepted_claims_count,
                    "error_count": error_count,
                    "duration_seconds": duration_seconds,
                    "file_hashes": json.dumps(file_hashes),
                },
            )

            for ans in answer_results:
                ans_uuid = uuid.uuid4()
                cur.execute(
                    """
                    INSERT INTO answer_results (
                        id, run_id, answer_id, status, error_message, latency_ms, turns_used, trace_ref
                    ) VALUES (
                        %(id)s, %(run_id)s, %(answer_id)s, %(status)s, %(error_message)s, %(latency_ms)s, %(turns_used)s, %(trace_ref)s
                    );
                    """,
                    {
                        "id": ans_uuid,
                        "run_id": run_uuid,
                        "answer_id": ans["answer_id"],
                        "status": ans["status"],
                        "error_message": ans.get("error_message"),
                        "latency_ms": ans.get("latency_ms", 0.0),
                        "turns_used": ans.get("turns_used", 0),
                        "trace_ref": ans.get("trace_ref"),
                    },
                )

                if ans.get("status") == "success":
                    for claim in ans.get("claims", []):
                        claim_uuid = uuid.uuid4()
                        cur.execute(
                            """
                            INSERT INTO claim_results (
                                id, run_id, answer_result_id, answer_id, claim_id,
                                start_char, end_char, quote, label, fact_ids, reason,
                                is_matched, is_label_correct, matched_gold_id
                            ) VALUES (
                                %(id)s, %(run_id)s, %(answer_result_id)s, %(answer_id)s, %(claim_id)s,
                                %(start_char)s, %(end_char)s, %(quote)s, %(label)s, %(fact_ids)s, %(reason)s,
                                %(is_matched)s, %(is_label_correct)s, %(matched_gold_id)s
                            );
                            """,
                            {
                                "id": claim_uuid,
                                "run_id": run_uuid,
                                "answer_result_id": ans_uuid,
                                "answer_id": ans["answer_id"],
                                "claim_id": claim["claim_id"],
                                "start_char": claim["start_char"],
                                "end_char": claim["end_char"],
                                "quote": claim["quote"],
                                "label": claim["label"],
                                "fact_ids": json.dumps(claim.get("fact_ids", [])),
                                "reason": claim.get("reason", ""),
                                "is_matched": claim.get("is_matched"),
                                "is_label_correct": claim.get("is_label_correct"),
                                "matched_gold_id": claim.get("matched_gold_id"),
                            },
                        )


def get_evaluation_run(run_id: str | uuid.UUID) -> Optional[Dict[str, Any]]:
    """Retrieve joined run metadata, answer outcomes, and predicted claims in stable order."""
    run_uuid = uuid.UUID(str(run_id))
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM runs WHERE id = %s;", (run_uuid,))
            run_row = cur.fetchone()
            if not run_row:
                return None

            # Retrieve answers sorted by answer_id
            cur.execute(
                "SELECT * FROM answer_results WHERE run_id = %s ORDER BY answer_id ASC;",
                (run_uuid,),
            )
            answers = cur.fetchall()

            # Retrieve claims sorted by answer_id and start_char
            cur.execute(
                "SELECT * FROM claim_results WHERE run_id = %s ORDER BY answer_id ASC, start_char ASC;",
                (run_uuid,),
            )
            claims = cur.fetchall()

            # Group claims under respective answers
            claims_by_ans: Dict[str, List[Dict[str, Any]]] = {}
            for c in claims:
                claims_by_ans.setdefault(c["answer_id"], []).append(c)

            formatted_answers = []
            for a in answers:
                ans_dict = dict(a)
                ans_dict["claims"] = claims_by_ans.get(a["answer_id"], [])
                formatted_answers.append(ans_dict)

            result = dict(run_row)
            result["answers"] = formatted_answers
            return result
