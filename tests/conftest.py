"""Pytest fixtures and FakeLLMProvider for controlled testing.

Allows simulating multi-turn exchanges, timeouts, errors, and turn exhaustion
without making live network calls or counting towards live evaluation runs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import pytest

from src.agent import AgentSession, BaseLLMProvider, ModelTurnResponse, ToolCallRequest
from src.fact_tool import BrandFactsCatalogue


class FakeSession(AgentSession):
    """Scripted session simulating turns, tool calls, and model outputs."""

    def __init__(self, script: List[ModelTurnResponse]) -> None:
        self.script = list(script)
        self.recorded_prompts: List[str] = []
        self.recorded_tool_responses: List[List[Tuple[str, Dict[str, Any]]]] = []

    def send_initial_prompt(
        self,
        system_instruction: str,
        user_text: str,
        timeout_seconds: float,
    ) -> ModelTurnResponse:
        self.recorded_prompts.append(user_text)
        if not self.script:
            return ModelTurnResponse(text='{"claims": []}')
        return self.script.pop(0)

    def send_tool_responses(
        self,
        responses: List[Tuple[str, Dict[str, Any]]],
        timeout_seconds: float,
    ) -> ModelTurnResponse:
        self.recorded_tool_responses.append(responses)
        if not self.script:
            return ModelTurnResponse(text='{"claims": []}')
        return self.script.pop(0)


class FakeLLMProvider(BaseLLMProvider):
    """Fake provider yielding scripted responses for testing."""

    def __init__(self, script: Optional[List[ModelTurnResponse]] = None) -> None:
        self.script = script or []
        self.last_session: Optional[FakeSession] = None

    def set_script(self, script: List[ModelTurnResponse]) -> None:
        self.script = list(script)

    def create_session(self) -> AgentSession:
        self.last_session = FakeSession(self.script)
        return self.last_session


@pytest.fixture
def sample_catalogue() -> BrandFactsCatalogue:
    """Fixture providing the speaker brand catalogue."""
    return BrandFactsCatalogue("brand_facts.csv")
