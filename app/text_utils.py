from __future__ import annotations

import re
import unicodedata


_PUNCT_RE = re.compile(r"[\s\W_]+", re.UNICODE)


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value or "").lower()
    return _PUNCT_RE.sub("", normalized)


def snippet(text: str, needle: str, radius: int = 26) -> str:
    if not needle:
        return text
    index = text.lower().find(needle.lower())
    if index < 0:
        return text
    start = max(0, index - radius)
    end = min(len(text), index + len(needle) + radius)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return f"{prefix}{text[start:end]}{suffix}"

