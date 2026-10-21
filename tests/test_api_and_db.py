"""Tests for FastAPI endpoints, status codes, and PostgreSQL database persistence."""

import uuid
import pytest
from fastapi.testclient import TestClient
import psycopg

from src.api import app
from src.database import get_db_connection, get_evaluation_run, init_db, save_evaluation_run


@pytest.fixture(scope="module")
def api_client():
    init_db()
    return TestClient(app)


def test_classify_forbidden_gold_fields(api_client):
    """Verify POST /classify rejects gold spans or claim list fields with 422."""
    payload = {
        "answer_id": "T01",
        "brand_id": "cleartrail",
        "question": "What is the price?",
        "ai_answer": "ClearTrail Mini is INR 1,299.",
        "expected_label": "supported",  # Forbidden!
    }
    response = api_client.post("/classify", json=payload)
    assert response.status_code == 422
    assert response.json()["detail"]["type"] == "forbidden_fields"

    payload2 = {
        "answer_id": "T01",
        "brand_id": "cleartrail",
        "question": "What is the price?",
        "ai_answer": "ClearTrail Mini is INR 1,299.",
        "claims": [{"quote": "...", "label": "supported"}],  # Forbidden!
    }
    response2 = api_client.post("/classify", json=payload2)
    assert response2.status_code == 422


def test_classify_unknown_brand_404(api_client):
    """Verify POST /classify with unknown primary brand returns 404."""
    payload = {
        "answer_id": "T01",
        "brand_id": "nonexistent_brand",
        "question": "What is the price?",
        "ai_answer": "It is INR 1,299.",
    }
    response = api_client.post("/classify", json=payload)
    assert response.status_code == 404
    assert response.json()["detail"]["type"] == "not_found"


def test_classify_missing_fields_422(api_client):
    """Verify POST /classify with missing required fields returns 422."""
    payload = {
        "answer_id": "T01",
        # Missing brand_id, question, ai_answer
    }
    response = api_client.post("/classify", json=payload)
    assert response.status_code == 422


def test_evaluation_runs_invalid_split_422(api_client):
    """Verify POST /evaluation-runs with invalid split returns 422."""
    payload = {"agent_version": "v1", "split": "invalid_split"}
    response = api_client.post("/evaluation-runs", json=payload)
    assert response.status_code == 422


def test_get_run_invalid_uuid_422(api_client):
    """Verify GET /evaluation-runs/{id} with invalid UUID returns 422."""
    response = api_client.get("/evaluation-runs/not-a-valid-uuid")
    assert response.status_code == 422


def test_get_run_missing_404(api_client):
    """Verify GET /evaluation-runs/{id} with missing UUID returns 404."""
    random_uuid = str(uuid.uuid4())
    response = api_client.get(f"/evaluation-runs/{random_uuid}")
    assert response.status_code == 404


def test_database_persistence_and_constraints():
    """Verify PostgreSQL run persistence, foreign keys, and unique constraints."""
    init_db()
    run_id = uuid.uuid4()

    # Insert a test run
    save_evaluation_run(
        run_id=run_id,
        split="development",
        agent_version="v1",
        prompt_version="v1.0",
        dataset_version="long-brand-answers-50-v4",
        fact_version="speaker-catalogue-v4",
        evaluator_version="sentence-spans-and-labels-v4",
        provider="pytest",
        model="fake-model",
        status="completed",
        settings={"temp": 0.0},
        metrics={"f1": 0.85},
        answer_results=[
            {
                "answer_id": "A01",
                "status": "success",
                "claims": [
                    {
                        "claim_id": "A01:10:30",
                        "start_char": 10,
                        "end_char": 30,
                        "quote": "Quote one.",
                        "label": "supported",
                        "fact_ids": ["cleartrail.mini.price_2026"],
                        "reason": "Reason one.",
                    },
                    {
                        "claim_id": "A01:40:60",
                        "start_char": 40,
                        "end_char": 60,
                        "quote": "Quote two.",
                        "label": "contradicted",
                        "fact_ids": ["cleartrail.mini.warranty_india"],
                        "reason": "Reason two.",
                    },
                ],
            }
        ],
        duration_seconds=1.2,
        file_hashes={"test": "hash"},
    )

    # Joined retrieval
    retrieved = get_evaluation_run(run_id)
    assert retrieved is not None
    assert str(retrieved["id"]) == str(run_id)
    assert len(retrieved["answers"]) == 1
    assert retrieved["answers"][0]["answer_id"] == "A01"
    assert len(retrieved["answers"][0]["claims"]) == 2
    # Check stable start_char ordering
    assert retrieved["answers"][0]["claims"][0]["start_char"] == 10
    assert retrieved["answers"][0]["claims"][1]["start_char"] == 40

    # Test unique constraint (run_id, answer_id)
    with pytest.raises(psycopg.errors.UniqueViolation):
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO answer_results (id, run_id, answer_id, status)
                    VALUES (%s, %s, %s, %s);
                    """,
                    (uuid.uuid4(), run_id, "A01", "error"),
                )

    # Test cascade delete: deleting run deletes answers and claims
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM runs WHERE id = %s;", (run_id,))
            cur.execute("SELECT COUNT(*) FROM answer_results WHERE run_id = %s;", (run_id,))
            assert cur.fetchone()["count"] == 0
            cur.execute("SELECT COUNT(*) FROM claim_results WHERE run_id = %s;", (run_id,))
            assert cur.fetchone()["count"] == 0
