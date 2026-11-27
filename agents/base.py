"""BaseAgent — shared LLM invocation + JSON validation + retry/repair (AGENTS.md §18).

Every analysis agent extends BaseAgent and overrides:
- `name`: short identifier (e.g. "metadata")
- `system_prompt`: the role-specific system prompt
- `output_model`: the Pydantic model class to validate against

BaseAgent.run(context) does the heavy lifting:
1. Build the [system, user] message pair from context.
2. Invoke the role-specific LLM via `make_llm(role)`.
3. Parse the response as JSON.
4. Validate against the Pydantic model.
5. On failure: retry up to N times with a "fix your JSON" follow-up.

If all retries fail, `run()` returns an instance of `output_model` populated
with safe defaults (None / [] / "Not available") so downstream code never
crashes (AGENTS.md §27 graceful error handling).

Prompt-injection guard (INJECTION_GUARD) is appended to every system
prompt automatically; subclasses don't need to include it.
"""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any, Callable, Type

from pydantic import BaseModel

from config.logging import get_logger
from config.settings import make_llm
from models.paper import AgentContext, RetrievedChunk, SourceReference
from rag.prompts import INJECTION_GUARD, build_user_prompt

logger = get_logger(__name__)

# Type alias: an LLM that responds to `.invoke(messages)` and returns an
# object with a `.content` attribute. LangChain chat models satisfy this.
LLMCallable = Any


# ============================================================================
# JSON extraction helpers
# ============================================================================
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _extract_json(text: str) -> dict | list | None:
    """Best-effort JSON extraction from an LLM response.

    Tries, in order:
    1. The whole string as JSON.
    2. The first ```json ... ``` fenced block.
    3. The first balanced {...} substring.
    Returns None if all attempts fail.
    """
    text = text.strip()
    # 1. Whole-string attempt
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Fenced block
    m = _FENCE_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 3. First balanced JSON object
    start = text.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    snippet = text[start : i + 1]
                    try:
                        return json.loads(snippet)
                    except json.JSONDecodeError:
                        break
    return None


def _looks_truncated(text: str) -> bool:
    """Heuristically detect a response that was cut off mid-JSON.

    A response truncated by the output-token cap typically has unbalanced
    braces/brackets or an unterminated string. Detecting this lets us
    report "raise LLM_NUM_PREDICT" instead of the generic "no JSON found".
    """
    t = text.strip()
    if not t:
        return False
    if t.endswith("}") and t.endswith("}"):
        # Balanced-looking end — but the object could still be malformed.
        pass
    if t.count("{") > t.count("}"):
        return True
    if t.count("[") > t.count("]"):
        return True
    # An odd number of unescaped quotes means the last string never closed.
    if t.count('"') % 2 == 1:
        return True
    # Truncation often leaves a dangling key with no value.
    return t.rstrip().endswith(":") or t.rstrip().endswith(",")


def _default_for(model: Type[BaseModel]) -> BaseModel:
    """Build an instance of `model` with all fields set to safe defaults."""
    fields = model.model_fields
    defaults: dict[str, Any] = {}
    for name, field in fields.items():
        if field.is_required():
            # Required-but-no-value fallback by type
            annotation = field.annotation
            if annotation in (list, set, tuple):
                defaults[name] = []
            elif annotation is str:
                defaults[name] = "Not available"
            else:
                defaults[name] = None
        else:
            defaults[name] = field.default
    return model(**defaults)


# ============================================================================
# BaseAgent
# ============================================================================
class BaseAgent(ABC):
    """Abstract base for all analysis agents."""

    #: Short identifier (used in logs + for `make_llm(role)`).
    name: str = "base"

    #: The Pydantic model the agent's output must validate against.
    output_model: Type[BaseModel]

    #: The role passed to `make_llm(role)` (e.g. "metadata").
    llm_role: str = "default"

    #: Max retry attempts when LLM produces malformed JSON.
    max_retries: int = 2

    def __init__(
        self,
        *,
        llm_factory: Callable[[str], Any] | None = None,
        max_retries: int | None = None,
    ) -> None:
        self._llm_factory = llm_factory or (lambda role: make_llm(role))
        if max_retries is not None:
            self.max_retries = max_retries
        logger.debug("Agent %s initialized", self.name)

    # ----- subclasses override these --------------------------------------
    @abstractmethod
    def system_prompt(self) -> str:
        """Return the agent-specific system prompt."""

    @abstractmethod
    def user_prompt(self, context: AgentContext) -> str:
        """Build the user-role message from the AgentContext."""

    # ----- shared execution ----------------------------------------------
    def build_messages(self, context: AgentContext) -> list[dict[str, str]]:
        system = self.system_prompt() + "\n\n" + INJECTION_GUARD
        # Always include a JSON instruction at the end of the system prompt
        # so the model reliably emits parseable output.
        system += (
            "\n\nOUTPUT FORMAT: Reply with a single JSON object matching the "
            "schema described above. Do not include prose outside the JSON."
        )
        user = self.user_prompt(context)
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def sources_from_context(self, context: AgentContext) -> list[SourceReference]:
        """Build deduplicated SourceReferences from context.retrieved_chunks."""
        seen: set[tuple[str, int | None]] = set()
        out: list[SourceReference] = []
        for c in context.retrieved_chunks:
            key = (c.paper_id, c.page)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                SourceReference(
                    paper_id=c.paper_id,
                    page=c.page,
                    section=c.section,
                    chunk_id=c.chunk_id,
                    text=(c.text[:300] + "…") if len(c.text) > 300 else c.text,
                )
            )
        return out

    def run(self, context: AgentContext) -> BaseModel:
        """Execute the agent end-to-end; always returns a valid output_model."""
        messages = self.build_messages(context)
        llm = self._llm_factory(self.llm_role)
        last_error: str | None = None
        for attempt in range(self.max_retries + 1):
            current_messages = list(messages)
            if attempt > 0:
                # Repair-mode follow-up: previous response was bad JSON.
                current_messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Your previous response could not be parsed as JSON "
                            f"(error: {last_error}). Reply ONLY with a valid JSON "
                            f"object matching the schema. No prose, no markdown "
                            f"fences, no commentary."
                        ),
                    }
                )
            try:
                response = llm.invoke(current_messages)
                content = getattr(response, "content", None) or str(response)
            except TimeoutError as exc:
                # Don't retry on timeout — the model is broken or the
                # network is wedged. Fail fast so the user sees the
                # error within ~timeout seconds instead of waiting for
                # (max_retries+1) * timeout = 30+ minutes.
                logger.error(
                    "Agent %s: LLM call timed out: %s", self.name, exc
                )
                fallback = _default_for(self.output_model)
                try:
                    fallback = fallback.model_copy(
                        update={"paper_id": context.paper_id}
                    )
                except Exception:  # noqa: BLE001
                    pass
                return fallback
            except Exception as exc:  # noqa: BLE001
                logger.warning("Agent %s LLM call failed: %s", self.name, exc)
                last_error = f"LLM error: {exc}"
                continue

            data = _extract_json(content)
            if data is None:
                last_error = "no JSON found in response"
                # Distinguish "model returned prose" from "response was
                # cut off mid-JSON" (output-token cap). The second case
                # needs a bigger num_predict, not a better prompt.
                truncated = _looks_truncated(content)
                if truncated:
                    last_error = (
                        "response truncated mid-JSON (likely hit the "
                        "output token cap)"
                    )
                logger.warning(
                    "Agent %s: %s (attempt %d/%d) len=%d",
                    self.name,
                    last_error,
                    attempt + 1,
                    self.max_retries + 1,
                    len(content),
                )
                # Log a short excerpt so the failure is diagnosable
                # without dumping megabytes into the log file.
                logger.debug(
                    "Agent %s raw response (first 400 chars): %s",
                    self.name,
                    content[:400],
                )
                continue

            try:
                obj = self.output_model.model_validate(data)
                # Attach sources from context if the model has a `sources` field.
                if "sources" in obj.model_fields:
                    obj = obj.model_copy(update={"sources": self.sources_from_context(context)})
                logger.info("Agent %s succeeded (attempt %d)", self.name, attempt + 1)
                return obj
            except Exception as exc:  # noqa: BLE001
                last_error = f"validation error: {exc}"
                logger.warning(
                    "Agent %s: %s (attempt %d/%d)",
                    self.name,
                    last_error,
                    attempt + 1,
                    self.max_retries + 1,
                )

        # All retries exhausted: return safe defaults so downstream code
        # never crashes (AGENTS.md §27).
        logger.error(
            "Agent %s: returning safe defaults after %d attempts",
            self.name,
            self.max_retries + 1,
        )
        fallback = _default_for(self.output_model)
        # paper_id is always required; populate from context
        try:
            fallback = fallback.model_copy(update={"paper_id": context.paper_id})
        except Exception:  # noqa: BLE001
            pass
        return fallback


__all__ = ["BaseAgent", "LLMCallable", "_extract_json", "_default_for"]
