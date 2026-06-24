from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from typing import Iterator
from urllib.parse import urlparse

import httpx

from .config import Settings


class LLMError(RuntimeError):
    pass


def is_llm_available(settings: Settings) -> bool:
    if not settings.enable_assistant:
        return False
    if settings.llm_api_key:
        return True
    host = (urlparse(settings.llm_base_url).hostname or "").lower()
    return host in {"localhost", "127.0.0.1", "::1"}


@dataclass(frozen=True)
class ExpandedQuery:
    intent: str
    search_terms: list[str]
    notes: str


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def expand_query(self, question: str, max_terms: int) -> ExpandedQuery:
        system = (
            "You are a Chinese video material retrieval assistant. "
            "Convert the user's natural-language request into practical search terms for transcript retrieval. "
            "Return JSON only."
        )
        user = {
            "task": "expand_video_material_query",
            "user_request": question,
            "requirements": [
                "Generate short Chinese transcript search phrases and emotional synonyms.",
                "Include likely spoken words, not only abstract labels.",
                "Do not invent file names or timestamps.",
                f"Return at most {max_terms} search_terms.",
            ],
            "json_schema": {
                "intent": "one short Chinese summary",
                "search_terms": ["term1", "term2"],
                "notes": "short Chinese note",
            },
        }
        payload = self._chat_json(system, json.dumps(user, ensure_ascii=False))
        terms = [
            str(term).strip()
            for term in payload.get("search_terms", [])
            if str(term).strip()
        ]
        return ExpandedQuery(
            intent=str(payload.get("intent") or question).strip(),
            search_terms=terms[:max_terms],
            notes=str(payload.get("notes") or "").strip(),
        )

    def expand_query_stream(self, question: str, max_terms: int) -> ExpandedQuery:
        system, user = self._expand_query_messages(question, max_terms)
        payload = None
        for event in self._chat_json_stream_events(system, user):
            if event.get("type") == "final":
                payload = event.get("payload")
        if not isinstance(payload, dict):
            raise LLMError("LLM returned no streamed JSON payload.")
        terms = [
            str(term).strip()
            for term in payload.get("search_terms", [])
            if str(term).strip()
        ]
        return ExpandedQuery(
            intent=str(payload.get("intent") or question).strip(),
            search_terms=terms[:max_terms],
            notes=str(payload.get("notes") or "").strip(),
        )

    def rerank(self, question: str, candidates: list[dict[str, Any]], limit: int) -> dict[str, Any]:
        system, user = self._rerank_messages(question, candidates, limit)
        return self._chat_json(system, user)

    def rerank_events(
        self,
        question: str,
        candidates: list[dict[str, Any]],
        limit: int,
    ) -> Iterator[dict[str, Any]]:
        system, user = self._rerank_messages(question, candidates, limit)
        last_answer = ""
        for event in self._chat_json_stream_events(system, user):
            if event.get("type") == "content":
                answer = _extract_partial_json_string(str(event.get("content") or ""), "answer")
                if len(answer) > len(last_answer):
                    delta = answer[len(last_answer) :]
                    last_answer = answer
                    if delta:
                        yield {"type": "answer_delta", "text": delta}
                continue
            yield event

    def _expand_query_messages(self, question: str, max_terms: int) -> tuple[str, str]:
        system = (
            "You are a Chinese video material retrieval assistant. "
            "Convert the user's natural-language request into practical search terms for transcript retrieval. "
            "Return JSON only."
        )
        user = {
            "task": "expand_video_material_query",
            "user_request": question,
            "requirements": [
                "Generate short Chinese transcript search phrases and emotional synonyms.",
                "Include likely spoken words, not only abstract labels.",
                "Do not invent file names or timestamps.",
                f"Return at most {max_terms} search_terms.",
            ],
            "json_schema": {
                "intent": "one short Chinese summary",
                "search_terms": ["term1", "term2"],
                "notes": "short Chinese note",
            },
        }
        return system, json.dumps(user, ensure_ascii=False)

    def _rerank_messages(
        self,
        question: str,
        candidates: list[dict[str, Any]],
        limit: int,
    ) -> tuple[str, str]:
        system = (
            "You are a senior video editing material search assistant. "
            "Pick the transcript candidates that best satisfy the user's request. "
            "You must only choose IDs from the provided candidates. Return JSON only."
        )
        compact_candidates = []
        for item in candidates:
            compact_candidates.append(
                {
                    "segment_id": item["segment_id"],
                    "filename": item["filename"],
                    "time": f"{item['start_seconds']:.1f}-{item['end_seconds']:.1f}s",
                    "match_type": item.get("match_type"),
                    "text": str(item.get("text") or "")[:360],
                }
            )

        user = {
            "task": "rerank_video_material_candidates",
            "user_request": question,
            "candidates": compact_candidates,
            "requirements": [
                f"Choose at most {limit} candidates.",
                "Prefer candidates that match the user's described intent, mood, action, or spoken content.",
                "If the candidates are weak, say so in answer and return the least bad useful candidates.",
                "Do not fabricate timestamps, files, or text.",
            ],
            "json_schema": {
                "answer": "short Chinese answer for the editor",
                "items": [
                    {
                        "segment_id": 123,
                        "confidence": 0.0,
                        "reason": "short Chinese reason",
                    }
                ],
            },
        }
        return system, json.dumps(user, ensure_ascii=False)

    def _chat_json(self, system: str, user: str) -> dict[str, Any]:
        body = {
            "model": self.settings.llm_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
        }
        try:
            return self._post_chat_json(body)
        except LLMError as exc:
            if "response_format" not in str(exc):
                raise
            body.pop("response_format", None)
            return self._post_chat_json(body)

    def _chat_json_stream_events(self, system: str, user: str) -> Iterator[dict[str, Any]]:
        body = {
            "model": self.settings.llm_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
        }
        try:
            yield from self._post_chat_json_stream_events(body)
            return
        except LLMError as exc:
            message = str(exc).lower()
            if "stream" in message:
                yield {"type": "final", "payload": self._post_chat_json(body)}
                return
            if "response_format" not in message:
                raise

        body.pop("response_format", None)
        try:
            yield from self._post_chat_json_stream_events(body)
        except LLMError as exc:
            if "stream" not in str(exc).lower():
                raise
            yield {"type": "final", "payload": self._post_chat_json(body)}

    def _post_chat_json(self, body: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.settings.llm_api_key:
            headers["Authorization"] = f"Bearer {self.settings.llm_api_key}"

        with httpx.Client(timeout=httpx.Timeout(120.0, connect=20.0)) as client:
            response = client.post(
                f"{self.settings.llm_base_url}/chat/completions",
                headers=headers,
                json=body,
            )

        if response.status_code >= 400:
            raise LLMError(f"LLM request failed: {response.status_code} {response.text[:500]}")

        payload = response.json()
        choices = payload.get("choices") or []
        if not choices:
            raise LLMError("LLM returned no choices.")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            content = "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
        if not isinstance(content, str) or not content.strip():
            raise LLMError("LLM returned empty content.")
        return self._parse_json_object(content)

    def _post_chat_json_stream_events(self, body: dict[str, Any]) -> Iterator[dict[str, Any]]:
        headers = {"Content-Type": "application/json"}
        if self.settings.llm_api_key:
            headers["Authorization"] = f"Bearer {self.settings.llm_api_key}"

        stream_body = dict(body)
        stream_body["stream"] = True
        content_parts: list[str] = []
        with httpx.Client(timeout=httpx.Timeout(120.0, connect=20.0)) as client:
            with client.stream(
                "POST",
                f"{self.settings.llm_base_url}/chat/completions",
                headers=headers,
                json=stream_body,
            ) as response:
                if response.status_code >= 400:
                    error_text = response.read().decode("utf-8", errors="replace")
                    raise LLMError(
                        f"LLM stream request failed: {response.status_code} {error_text[:500]}"
                    )
                for line in response.iter_lines():
                    data = line.strip()
                    if not data:
                        continue
                    if data.startswith("data:"):
                        data = data[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        payload = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    choices = payload.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    content = delta.get("content")
                    if isinstance(content, list):
                        content = "".join(
                            str(part.get("text", "")) for part in content if isinstance(part, dict)
                        )
                    if not isinstance(content, str) or not content:
                        continue
                    content_parts.append(content)
                    full_content = "".join(content_parts)
                    yield {"type": "content", "delta": content, "content": full_content}

        content = "".join(content_parts)
        if not content.strip():
            raise LLMError("LLM returned empty streamed content.")
        yield {"type": "final", "payload": self._parse_json_object(content)}

    def _parse_json_object(self, content: str) -> dict[str, Any]:
        text = content.strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start < 0 or end <= start:
                raise LLMError(f"LLM did not return JSON: {text[:200]}")
            parsed = json.loads(text[start : end + 1])
        if not isinstance(parsed, dict):
            raise LLMError("LLM JSON payload must be an object.")
        return parsed


def _extract_partial_json_string(content: str, key: str) -> str:
    key_token = json.dumps(key, ensure_ascii=False)
    key_index = content.find(key_token)
    if key_index < 0:
        return ""
    colon_index = content.find(":", key_index + len(key_token))
    if colon_index < 0:
        return ""
    start_quote = content.find('"', colon_index + 1)
    if start_quote < 0:
        return ""

    index = start_quote + 1
    escaped = False
    end_quote = -1
    while index < len(content):
        char = content[index]
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == '"':
            end_quote = index
            break
        index += 1

    raw_value = content[start_quote + 1 : end_quote if end_quote >= 0 else len(content)]
    if raw_value.endswith("\\"):
        raw_value = raw_value[:-1]
    try:
        return json.loads(f'"{raw_value}"')
    except json.JSONDecodeError:
        return raw_value.replace('\\"', '"')
