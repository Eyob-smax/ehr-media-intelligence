"""Claude-backed summarizer using structured outputs.

- Structured outputs (``output_config.format``) guarantee schema-valid JSON, so
  there is no brittle parsing of free text.
- Server-side refusal fallbacks are enabled: if a safety classifier declines
  (clinical text occasionally trips them), the API retries on a fallback model
  inside the same call instead of returning nothing.
- If the summary comes back over the word limit we ask once for a shorter
  version in the same conversation; the quality layer trims as a last resort.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import anthropic

from .context import PatientContext
from .prompt import SYSTEM_PROMPT, shorten_message, user_message
from .schema import CONTENT_JSON_SCHEMA, MAX_WORDS, SummaryContent, word_count

log = logging.getLogger(__name__)

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class SummarizerError(RuntimeError):
    """Raised when the LLM could not produce a usable summary; callers fall back to offline."""

    def __init__(self, message: str, *, permanent: bool = False):
        super().__init__(message)
        self.permanent = permanent  # e.g. bad credentials: stop trying for the rest of the run


def credentials_available() -> bool:
    """Cheap check so 'auto' mode doesn't attempt a doomed network call per patient."""
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    return (Path.home() / ".config" / "anthropic").exists()


class ClaudeSummarizer:
    name = "claude"

    def __init__(self, model: str, client: anthropic.Anthropic | None = None, effort: str = "medium"):
        self.model = model
        self.effort = effort
        self.client = client or anthropic.Anthropic(max_retries=3, timeout=120.0)

    def _call(self, messages: list[dict]) -> tuple[SummaryContent, list, str]:
        try:
            resp = self.client.beta.messages.create(
                model=self.model,
                max_tokens=16000,
                betas=[FALLBACK_BETA],
                fallbacks="default",
                system=SYSTEM_PROMPT,
                output_config={
                    "effort": self.effort,
                    "format": {"type": "json_schema", "schema": CONTENT_JSON_SCHEMA},
                },
                messages=messages,
            )
        except anthropic.AuthenticationError as e:
            raise SummarizerError(f"authentication failed: {e.message}", permanent=True) from e
        except anthropic.PermissionDeniedError as e:
            raise SummarizerError(f"permission denied: {e.message}", permanent=True) from e
        except anthropic.NotFoundError as e:
            raise SummarizerError(f"model not found: {self.model}", permanent=True) from e
        except anthropic.RateLimitError as e:
            raise SummarizerError("rate limited after retries") from e
        except anthropic.APIStatusError as e:
            raise SummarizerError(f"API error {e.status_code}: {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise SummarizerError("could not reach the Anthropic API") from e
        except TypeError as e:  # SDK raises this when no credentials can be resolved at all
            raise SummarizerError(f"no Anthropic credentials: {e}", permanent=True) from e

        if resp.stop_reason == "refusal":
            category = getattr(resp.stop_details, "category", None) if resp.stop_details else None
            raise SummarizerError(f"model declined to summarize (category: {category})")
        if resp.stop_reason == "max_tokens":
            raise SummarizerError("response truncated at max_tokens")
        text = next((b.text for b in resp.content if b.type == "text"), None)
        if not text:
            raise SummarizerError("response contained no text block")
        try:
            content = SummaryContent.model_validate_json(text)
        except ValueError as e:
            raise SummarizerError(f"response did not match schema: {e}") from e
        return content, resp.content, resp.model

    def summarize(self, ctx: PatientContext) -> tuple[SummaryContent, str]:
        """Return (content, model that actually served the request)."""
        messages: list[dict] = [{"role": "user", "content": user_message(ctx.render())}]
        content, raw_content, served_by = self._call(messages)
        words = word_count(content)
        if words > MAX_WORDS:
            log.info("summary for %s was %d words; asking for a shorter version", ctx.patient_id, words)
            messages += [
                {"role": "assistant", "content": raw_content},  # pass blocks back unchanged
                {"role": "user", "content": shorten_message(words)},
            ]
            content, _, served_by = self._call(messages)
        return content, served_by
