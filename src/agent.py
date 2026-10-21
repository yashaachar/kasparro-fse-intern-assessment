"""AI agent module for KASPARRO FSE Intern Pathway Assessment.

Orchestrates multi-turn model interaction, tool calling for catalogue facts,
turn limits, provider timeouts, output parsing, and strict answer error preservation.
"""

from __future__ import annotations

import json
import os
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from src.fact_tool import BrandFactsCatalogue, get_default_catalogue
from src.models import AnswerResult, PredictedClaim
from src.validation import validate_and_build_claims


MAX_TURNS = 3
DEFAULT_TIMEOUT_SECONDS = 45.0
DEFAULT_MODEL = "gemini-3.5-flash-lite"


def extract_json_block(text: str) -> Any:
    """Extract and parse JSON object/array from text, supporting markdown blocks."""
    trimmed = text.strip()
    pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    match = re.search(pattern, trimmed)
    candidate = match.group(1).strip() if match else trimmed

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        start_brace = candidate.find("{")
        end_brace = candidate.rfind("}")
        if start_brace != -1 and end_brace > start_brace:
            try:
                return json.loads(candidate[start_brace : end_brace + 1])
            except json.JSONDecodeError:
                pass

        start_bracket = candidate.find("[")
        end_bracket = candidate.rfind("]")
        if start_bracket != -1 and end_bracket > start_bracket:
            try:
                return json.loads(candidate[start_bracket : end_bracket + 1])
            except json.JSONDecodeError:
                pass

        # Try repairing cut-off JSON if closed bracket was truncated
        if start_brace != -1:
            last_brace = candidate.rfind("}")
            if last_obj_end := candidate.rfind("}"):
                sub = candidate[start_brace : last_obj_end + 1]
                for end_suffix in ["]}", "}"]:
                    try:
                        return json.loads(sub + end_suffix)
                    except json.JSONDecodeError:
                        pass

        raise ValueError(f"Could not parse valid JSON from model response: {trimmed[:200]}")


class ToolCallRequest:
    """Represents a tool call requested by the model."""

    def __init__(self, name: str, args: Dict[str, Any], call_id: Optional[str] = None) -> None:
        self.name = name
        self.args = args
        self.call_id = call_id


class ModelTurnResponse:
    """Response from one model turn."""

    def __init__(
        self,
        text: Optional[str] = None,
        tool_calls: Optional[List[ToolCallRequest]] = None,
        raw_model_content: Any = None,
    ) -> None:
        self.text = text
        self.tool_calls = tool_calls or []
        self.raw_model_content = raw_model_content


class AgentSession(ABC):
    """Abstract conversation session for multi-turn execution."""

    @abstractmethod
    def send_initial_prompt(
        self,
        system_instruction: str,
        user_text: str,
        timeout_seconds: float,
    ) -> ModelTurnResponse:
        pass

    @abstractmethod
    def send_tool_responses(
        self,
        responses: List[Tuple[str, Dict[str, Any]]],
        timeout_seconds: float,
    ) -> ModelTurnResponse:
        pass


class BaseLLMProvider(ABC):
    """Factory interface for creating model conversation sessions."""

    @abstractmethod
    def create_session(self) -> AgentSession:
        pass


class GeminiSession(AgentSession):
    """Session handling live multi-turn interactions with Google Gemini."""

    def __init__(self, client: Any, model: str, tool: Any) -> None:
        from google.genai import types

        self.client = client
        self.model = model
        self.tool = tool
        self.contents: List[types.Content] = []
        self.system_instruction: str = ""

    def send_initial_prompt(
        self,
        system_instruction: str,
        user_text: str,
        timeout_seconds: float,
    ) -> ModelTurnResponse:
        from google.genai import types

        self.system_instruction = system_instruction
        self.contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_text)],
            )
        ]
        return self._execute_generate(timeout_seconds)

    def send_tool_responses(
        self,
        responses: List[Tuple[str, Dict[str, Any]]],
        timeout_seconds: float,
    ) -> ModelTurnResponse:
        from google.genai import types

        resp_parts = []
        for name, res in responses:
            resp_parts.append(
                types.Part.from_function_response(
                    name=name,
                    response={"result": res},
                )
            )

        self.contents.append(types.Content(role="user", parts=resp_parts))
        return self._execute_generate(timeout_seconds)

    def _execute_generate(self, timeout_seconds: float) -> ModelTurnResponse:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=self.system_instruction,
            tools=[self.tool],
            temperature=0.0,
            max_output_tokens=4096,
            http_options=types.HttpOptions(timeout=int(timeout_seconds * 1000)),
        )

        max_retries = 3
        base_wait = 4.0

        for attempt in range(max_retries + 1):
            try:
                res = self.client.models.generate_content(
                    model=self.model,
                    contents=self.contents,
                    config=config,
                )
                break
            except Exception as e:
                err_msg = str(e)
                is_transient = any(
                    code in err_msg for code in ["429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "500", "INTERNAL"]
                )
                if is_transient and attempt < max_retries:
                    wait_time = base_wait * (2**attempt)
                    time.sleep(wait_time)
                    continue
                if "deadline" in err_msg.lower() or "timeout" in err_msg.lower():
                    raise TimeoutError(f"Provider request timed out after {timeout_seconds}s: {e}")
                raise RuntimeError(f"Gemini API request failed: {e}") from e

        if not res.candidates:
            return ModelTurnResponse(text="")

        candidate = res.candidates[0]
        model_content = candidate.content
        if model_content is not None:
            self.contents.append(model_content)

        text_parts = []
        tool_calls: List[ToolCallRequest] = []

        if model_content and model_content.parts:
            for part in model_content.parts:
                if part.text:
                    text_parts.append(part.text)
                if part.function_call:
                    fc = part.function_call
                    tool_calls.append(
                        ToolCallRequest(
                            name=fc.name,
                            args=dict(fc.args) if fc.args else {},
                        )
                    )

        full_text = "\n".join(text_parts).strip() if text_parts else None
        return ModelTurnResponse(text=full_text, tool_calls=tool_calls, raw_model_content=model_content)


class GeminiProvider(BaseLLMProvider):
    """Live Google Gemini provider using google-genai SDK."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        from google import genai
        from google.genai import types

        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")

        self.model = model or os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)
        self.client = genai.Client(api_key=self.api_key)

        func_decl = types.FunctionDeclaration(
            name="get_brand_facts",
            description="Retrieve official speaker catalogue facts for a specific brand.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "brand_id": types.Schema(
                        type=types.Type.STRING,
                        description="Brand ID: cleartrail, stonewave, or cloudnote",
                    )
                },
                required=["brand_id"],
            ),
        )
        self.tool = types.Tool(function_declarations=[func_decl])

    def create_session(self) -> AgentSession:
        return GeminiSession(self.client, self.model, self.tool)


class ClaimExtractionAgent:
    """Agent for factual claim sentence extraction and catalogue classification."""

    def __init__(
        self,
        catalogue: Optional[BrandFactsCatalogue] = None,
        provider: Optional[BaseLLMProvider] = None,
        prompt_dir: str | Path = "prompts",
    ) -> None:
        self.catalogue = catalogue or get_default_catalogue()
        self.provider = provider
        self.prompt_dir = Path(prompt_dir)
        self.valid_fact_ids: Set[str] = self.catalogue.get_valid_fact_ids()
        self.prompts: Dict[str, str] = {}
        self._load_prompts()

    def _load_prompts(self) -> None:
        for v in ["v1", "v2"]:
            pfile = self.prompt_dir / f"{v}.md"
            if pfile.exists():
                with open(pfile, "r", encoding="utf-8") as f:
                    self.prompts[v] = f.read()

    def _get_provider(self) -> BaseLLMProvider:
        if self.provider is not None:
            return self.provider
        self.provider = GeminiProvider()
        return self.provider

    def classify_answer(
        self,
        answer_id: str,
        brand_id: str,
        question: str,
        ai_answer: str,
        agent_version: str = "v1",
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> AnswerResult:
        """Run up to 3 turns to extract and classify claims for an answer.

        Preserves answer-level errors without inventing predictions.
        """
        start_time = time.time()
        turns_used = 0
        model_exchange: List[Dict[str, Any]] = []

        try:
            clean_brand = brand_id.strip().lower()
            if clean_brand not in self.catalogue.known_brands:
                raise ValueError(f"Unknown primary brand_id '{brand_id}'")

            system_instruction = self.prompts.get(agent_version)
            if not system_instruction:
                raise ValueError(f"Prompt template for agent_version '{agent_version}' not found")

            provider = self._get_provider()
            session = provider.create_session()

            user_initial_prompt = (
                f"Extract and classify factual claim sentences for the following answer:\n\n"
                f"Answer ID: {answer_id}\n"
                f"Primary Brand ID: {clean_brand}\n"
                f"Question: {question}\n\n"
                f"AI Answer:\n{ai_answer}\n\n"
                f"Call get_brand_facts to retrieve catalogue facts for '{clean_brand}' "
                f"and any comparison brands referenced, then output the final JSON claims."
            )

            # Turn 1
            turns_used += 1
            turn_response = session.send_initial_prompt(
                system_instruction=system_instruction,
                user_text=user_initial_prompt,
                timeout_seconds=timeout_seconds,
            )

            final_text: Optional[str] = None

            while True:
                if turn_response.tool_calls:
                    executed_responses: List[Tuple[str, Dict[str, Any]]] = []
                    turn_record = {
                        "turn": turns_used,
                        "tool_calls": [],
                        "tool_responses": [],
                    }

                    for call in turn_response.tool_calls:
                        turn_record["tool_calls"].append({"name": call.name, "args": call.args})

                        tool_result = self.catalogue.execute_tool(call.name, call.args)
                        executed_responses.append((call.name, tool_result))
                        turn_record["tool_responses"].append(
                            {"name": call.name, "fact_count": len(tool_result.get("facts", []))}
                        )

                    model_exchange.append(turn_record)

                    if turns_used >= MAX_TURNS:
                        raise TimeoutError(
                            f"Turn limit of {MAX_TURNS} model turns exhausted without final predictions"
                        )

                    turns_used += 1
                    turn_response = session.send_tool_responses(
                        responses=executed_responses,
                        timeout_seconds=timeout_seconds,
                    )

                elif turn_response.text:
                    final_text = turn_response.text
                    model_exchange.append(
                        {
                            "turn": turns_used,
                            "final_text_snippet": final_text[:300],
                        }
                    )
                    break
                else:
                    raise ValueError("Model returned neither text nor tool calls")

            if final_text is None:
                raise TimeoutError(f"Turn limit of {MAX_TURNS} model turns exhausted without final predictions")

            parsed_data = extract_json_block(final_text)
            if isinstance(parsed_data, dict):
                raw_claims = parsed_data.get("claims", [])
            elif isinstance(parsed_data, list):
                raw_claims = parsed_data
            else:
                raise ValueError(f"Unexpected JSON format from model: {type(parsed_data)}")

            validated_claims = validate_and_build_claims(
                answer_id=answer_id,
                ai_answer=ai_answer,
                raw_claims=raw_claims,
                valid_fact_ids=self.valid_fact_ids,
            )

            latency_ms = round((time.time() - start_time) * 1000, 2)
            return AnswerResult(
                answer_id=answer_id,
                status="success",
                error_message=None,
                latency_ms=latency_ms,
                claims=validated_claims,
                turns_used=turns_used,
                model_exchange=model_exchange,
            )

        except Exception as e:
            latency_ms = round((time.time() - start_time) * 1000, 2)
            return AnswerResult(
                answer_id=answer_id,
                status="error",
                error_message=str(e),
                latency_ms=latency_ms,
                claims=[],
                turns_used=turns_used,
                model_exchange=model_exchange,
            )
