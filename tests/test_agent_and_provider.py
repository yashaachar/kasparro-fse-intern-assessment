"""Controlled tests for ClaimExtractionAgent with FakeLLMProvider.

Tests malformed output, invalid fact IDs, illegal tools, timeouts, turn exhaustion,
and ensures that gold references never enter model payloads.
"""

import pytest

from src.agent import ClaimExtractionAgent, ModelTurnResponse, ToolCallRequest
from tests.conftest import FakeLLMProvider


def test_agent_successful_multi_turn_flow(sample_catalogue):
    """Verify standard 2-turn execution: tool call then final extraction."""
    text = "ClearTrail Mini is 350g."
    provider = FakeLLMProvider(
        script=[
            # Turn 1: tool call
            ModelTurnResponse(
                tool_calls=[ToolCallRequest(name="get_brand_facts", args={"brand_id": "cleartrail"})]
            ),
            # Turn 2: final output
            ModelTurnResponse(
                text='{"claims": [{"quote": "ClearTrail Mini is 350g.", "label": "supported", "fact_ids": ["cleartrail.mini.weight"], "reason": "Weight matches catalogue."}]}'
            ),
        ]
    )

    agent = ClaimExtractionAgent(catalogue=sample_catalogue, provider=provider)
    res = agent.classify_answer(
        answer_id="T01",
        brand_id="cleartrail",
        question="What is the weight?",
        ai_answer=text,
    )

    assert res.status == "success"
    assert res.error_message is None
    assert len(res.claims) == 1
    assert res.claims[0].claim_id == "T01:0:24"
    assert res.turns_used == 2


def test_agent_malformed_json_preserved_as_error(sample_catalogue):
    """Verify malformed JSON results in answer-level error without crashing or inventing predictions."""
    text = "ClearTrail Mini is 350g."
    provider = FakeLLMProvider(
        script=[
            ModelTurnResponse(text="This is not JSON at all: {bad_json")
        ]
    )

    agent = ClaimExtractionAgent(catalogue=sample_catalogue, provider=provider)
    res = agent.classify_answer(
        answer_id="T01",
        brand_id="cleartrail",
        question="Q?",
        ai_answer=text,
    )

    assert res.status == "error"
    assert res.claims == []
    assert "JSON" in res.error_message


def test_agent_illegal_tool_call_rejected(sample_catalogue):
    """Verify model requesting an illegal tool produces an answer-level error."""
    text = "ClearTrail Mini is 350g."
    provider = FakeLLMProvider(
        script=[
            ModelTurnResponse(
                tool_calls=[ToolCallRequest(name="execute_untrusted_command", args={"cmd": "whoami"})]
            )
        ]
    )

    agent = ClaimExtractionAgent(catalogue=sample_catalogue, provider=provider)
    res = agent.classify_answer(
        answer_id="T01",
        brand_id="cleartrail",
        question="Q?",
        ai_answer=text,
    )

    assert res.status == "error"
    assert res.claims == []
    assert "Illegal tool" in res.error_message


def test_agent_unknown_brand_requested_rejected(sample_catalogue):
    """Verify model requesting an unknown brand produces an answer-level error."""
    text = "ClearTrail Mini is 350g."
    provider = FakeLLMProvider(
        script=[
            ModelTurnResponse(
                tool_calls=[ToolCallRequest(name="get_brand_facts", args={"brand_id": "unregistered_brand"})]
            )
        ]
    )

    agent = ClaimExtractionAgent(catalogue=sample_catalogue, provider=provider)
    res = agent.classify_answer(
        answer_id="T01",
        brand_id="cleartrail",
        question="Q?",
        ai_answer=text,
    )

    assert res.status == "error"
    assert "Unknown brand_id" in res.error_message


def test_agent_turn_exhaustion(sample_catalogue):
    """Verify exhausting the 3-turn limit produces an answer-level error."""
    text = "ClearTrail Mini is 350g."
    # 3 turns of repeated tool calls without final predictions
    provider = FakeLLMProvider(
        script=[
            ModelTurnResponse(
                tool_calls=[ToolCallRequest(name="get_brand_facts", args={"brand_id": "cleartrail"})]
            ),
            ModelTurnResponse(
                tool_calls=[ToolCallRequest(name="get_brand_facts", args={"brand_id": "cleartrail"})]
            ),
            ModelTurnResponse(
                tool_calls=[ToolCallRequest(name="get_brand_facts", args={"brand_id": "cleartrail"})]
            ),
        ]
    )

    agent = ClaimExtractionAgent(catalogue=sample_catalogue, provider=provider)
    res = agent.classify_answer(
        answer_id="T01",
        brand_id="cleartrail",
        question="Q?",
        ai_answer=text,
    )

    assert res.status == "error"
    assert "Turn limit" in res.error_message
    assert res.claims == []


def test_gold_never_sent_to_model(sample_catalogue):
    """Confirm gold spans or labels never enter model or tool payloads."""
    text = "ClearTrail Mini has bluetooth 5.3."
    provider = FakeLLMProvider(
        script=[
            ModelTurnResponse(
                text='{"claims": [{"quote": "ClearTrail Mini has bluetooth 5.3.", "label": "supported", "fact_ids": ["cleartrail.mini.bluetooth"], "reason": "Matches."}]}'
            )
        ]
    )

    agent = ClaimExtractionAgent(catalogue=sample_catalogue, provider=provider)
    agent.classify_answer(
        answer_id="A01",
        brand_id="cleartrail",
        question="Is bluetooth supported?",
        ai_answer=text,
    )

    # Check the recorded prompt sent to FakeSession
    recorded_prompt = provider.last_session.recorded_prompts[0]
    forbidden_tokens = ["gold", "expected_label", "difficulty_tags", "A01:306:460"]
    for token in forbidden_tokens:
        assert token not in recorded_prompt
