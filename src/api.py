"""FastAPI web service for KASPARRO FSE Intern Pathway Assessment.

Exposes:
- POST /classify: single answer classification
- POST /evaluation-runs: batch execution of development or check splits
- GET /evaluation-runs/{id}: joined PostgreSQL run retrieval
"""

from __future__ import annotations

import csv
import hashlib
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from src.agent import ClaimExtractionAgent
from src.database import get_evaluation_run, init_db, save_evaluation_run
from src.evaluator import Evaluator, EVALUATOR_VERSION
from src.fact_tool import FACT_VERSION, get_default_catalogue
from src.models import (
    AnswerRecord,
    AnswerResult,
    ClassifyResponse,
    EvaluationRunRequest,
    EvaluationRunResponse,
    VALID_BRANDS,
    VALID_SPLITS,
    VALID_VERSIONS,
)

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Ensure database schema is created on startup."""
    init_db()
    yield


app = FastAPI(
    title="KASPARRO Claim Extraction & Verification Service",
    description="FSE Intern Pathway Assessment API for speaker product factual claim extraction and evaluation.",
    version="1.0.0",
    lifespan=lifespan,
)

DATASET_VERSION = "long-brand-answers-50-v4"
PROMPT_VERSION_MAP = {"v1": "v1.0", "v2": "v2.0"}

FORBIDDEN_CLASSIFY_FIELDS = {
    "claims",
    "start_char",
    "end_char",
    "expected_label",
    "evidence_fact_ids",
    "explanation",
    "difficulty_tags",
    "gold_spans",
}


def get_file_hashes() -> Dict[str, str]:
    """Calculate SHA256 hashes for dataset CSV files."""
    hashes = {}
    for fname in ["answers.csv", "brand_facts.csv", "development_gold.csv"]:
        fpath = Path(fname)
        if fpath.exists():
            with open(fpath, "rb") as f:
                hashes[fname] = hashlib.sha256(f.read()).hexdigest()
    return hashes


@app.post(
    "/classify",
    response_model=ClassifyResponse,
    status_code=status.HTTP_200_OK,
    summary="Extract and classify claims for a single answer",
)
async def classify(request: Request) -> ClassifyResponse:
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"type": "validation_error", "message": "Invalid JSON body"},
        )

    # Reject if gold or claim-list input fields are present
    forbidden_present = [f for f in FORBIDDEN_CLASSIFY_FIELDS if f in body]
    if forbidden_present:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "type": "forbidden_fields",
                "message": f"Input contains forbidden gold/claim fields: {forbidden_present}",
            },
        )

    # Validate required fields
    required = ["answer_id", "brand_id", "question", "ai_answer"]
    missing = [f for f in required if f not in body or not str(body[f]).strip()]
    if missing:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"type": "validation_error", "message": f"Missing required fields: {missing}"},
        )

    answer_id = str(body["answer_id"]).strip()
    brand_id = str(body["brand_id"]).strip().lower()
    question = str(body["question"])
    ai_answer = str(body["ai_answer"])
    agent_version = str(body.get("agent_version", "v1")).strip().lower()

    if agent_version not in VALID_VERSIONS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"type": "validation_error", "message": f"agent_version must be one of {VALID_VERSIONS}"},
        )

    # Unknown primary brand -> 404
    catalogue = get_default_catalogue()
    if brand_id not in catalogue.known_brands:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"type": "not_found", "message": f"Unknown primary brand '{brand_id}'"},
        )

    agent = ClaimExtractionAgent(catalogue=catalogue)
    result: AnswerResult = agent.classify_answer(
        answer_id=answer_id,
        brand_id=brand_id,
        question=question,
        ai_answer=ai_answer,
        agent_version=agent_version,
    )

    if result.status != "success":
        err_msg = result.error_message or "Unknown classification error"
        if "timeout" in err_msg.lower():
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail={"type": "timeout_error", "message": err_msg},
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail={"type": "provider_error", "message": err_msg},
            )

    return ClassifyResponse(
        answer_id=answer_id,
        agent_version=agent_version,
        claims=result.claims,
        latency_ms=result.latency_ms,
    )


@app.post(
    "/evaluation-runs",
    response_model=EvaluationRunResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Execute evaluation run on development or check split",
)
async def create_evaluation_run(request_body: EvaluationRunRequest) -> EvaluationRunResponse:
    agent_version = request_body.agent_version.strip().lower()
    split = request_body.split.strip().lower()

    if agent_version not in VALID_VERSIONS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"type": "validation_error", "message": f"agent_version must be one of {VALID_VERSIONS}"},
        )

    if split not in VALID_SPLITS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"type": "validation_error", "message": f"split must be one of {VALID_SPLITS}"},
        )

    answers_path = Path("answers.csv")
    if not answers_path.exists():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"type": "internal_error", "message": "answers.csv not found"},
        )

    scheduled_records: List[AnswerRecord] = []
    with open(answers_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"].strip().lower() == split:
                scheduled_records.append(
                    AnswerRecord(
                        answer_id=row["answer_id"].strip(),
                        split=split,
                        brand_id=row["brand_id"].strip().lower(),
                        question=row["question"],
                        ai_answer=row["ai_answer"],
                    )
                )

    if not scheduled_records:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"type": "not_found", "message": f"No answers found for split '{split}'"},
        )

    run_id = uuid.uuid4()
    start_run_time = time.time()
    catalogue = get_default_catalogue()
    agent = ClaimExtractionAgent(catalogue=catalogue)
    evaluator = Evaluator()

    answer_results: List[AnswerResult] = []
    for rec in scheduled_records:
        ans_res = agent.classify_answer(
            answer_id=rec.answer_id,
            brand_id=rec.brand_id,
            question=rec.question,
            ai_answer=rec.ai_answer,
            agent_version=agent_version,
        )
        answer_results.append(ans_res)

    duration_seconds = round(time.time() - start_run_time, 2)

    # Compute metrics
    if split == "development":
        metrics = evaluator.evaluate_development_split(answer_results)
    else:
        metrics = evaluator.evaluate_check_split(answer_results)

    serialized_answers: List[Dict[str, Any]] = []
    accepted_pred_count = 0
    error_count = 0

    for ans in answer_results:
        if ans.status == "success":
            accepted_pred_count += len(ans.claims)
        else:
            error_count += 1

        claims_list = [c.model_dump() for c in ans.claims]
        serialized_answers.append(
            {
                "answer_id": ans.answer_id,
                "status": ans.status,
                "error_message": ans.error_message,
                "latency_ms": ans.latency_ms,
                "turns_used": ans.turns_used,
                "trace_ref": ans.trace_ref,
                "claims": claims_list,
            }
        )

    if split == "development":
        golds_by_aid = evaluator.gold_claims_by_answer
        for ans in serialized_answers:
            aid = ans["answer_id"]
            gold_list = golds_by_aid.get(aid, [])
            for c in ans.get("claims", []):
                for g in gold_list:
                    if g.start_char == c["start_char"] and g.end_char == c["end_char"]:
                        c["is_matched"] = True
                        c["is_label_correct"] = c["label"] == g.expected_label
                        c["matched_gold_id"] = g.claim_id
                        break

    file_hashes = get_file_hashes()
    provider_name = getattr(agent._get_provider(), "__class__", type("x", (), {"__name__": "provider"}))
    provider_str = provider_name.__name__ if hasattr(provider_name, "__name__") else "GeminiProvider"
    model_str = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")

    save_evaluation_run(
        run_id=run_id,
        split=split,
        agent_version=agent_version,
        prompt_version=PROMPT_VERSION_MAP.get(agent_version, "v1.0"),
        dataset_version=DATASET_VERSION,
        fact_version=FACT_VERSION,
        evaluator_version=EVALUATOR_VERSION,
        provider=provider_str,
        model=model_str,
        status="completed",
        settings={"temperature": 0.0, "max_turns": 3},
        metrics=metrics.model_dump() if metrics else None,
        answer_results=serialized_answers,
        duration_seconds=duration_seconds,
        file_hashes=file_hashes,
    )

    return EvaluationRunResponse(
        run_id=str(run_id),
        status="completed",
        agent_version=agent_version,
        split=split,
        answer_count=len(scheduled_records),
        accepted_prediction_count=accepted_pred_count,
        error_count=error_count,
        metrics=metrics,
        duration_seconds=duration_seconds,
    )


@app.get(
    "/evaluation-runs/{id}",
    status_code=status.HTTP_200_OK,
    summary="Retrieve joined evaluation run by UUID",
)
async def get_run_by_id(id: str) -> Dict[str, Any]:
    try:
        run_uuid = uuid.UUID(id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"type": "validation_error", "message": f"Invalid UUID: '{id}'"},
        )

    data = get_evaluation_run(run_uuid)
    if not data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"type": "not_found", "message": f"Evaluation run '{id}' not found"},
        )

    data["id"] = str(data["id"])
    if data.get("created_at"):
        data["created_at"] = str(data["created_at"])
    for ans in data.get("answers", []):
        ans["id"] = str(ans["id"])
        ans["run_id"] = str(ans["run_id"])
        if ans.get("created_at"):
            ans["created_at"] = str(ans["created_at"])
        for c in ans.get("claims", []):
            c["id"] = str(c["id"])
            c["run_id"] = str(c["run_id"])
            c["answer_result_id"] = str(c["answer_result_id"])

    return data
