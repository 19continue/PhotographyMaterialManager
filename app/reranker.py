from __future__ import annotations

import importlib.util
import math
from typing import Protocol

from .config import Settings
from .embeddings import model_log
from .embeddings import resolve_local_model_path


class Reranker(Protocol):
    def score_pairs(self, query: str, texts: list[str]) -> list[float]:
        ...


class NoopReranker:
    def score_pairs(self, query: str, texts: list[str]) -> list[float]:
        return []


class LocalCrossEncoderReranker:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model = None

    def score_pairs(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        model = self._get_model()
        scores = model.predict(
            [(query, text) for text in texts],
            batch_size=16,
            show_progress_bar=False,
        )
        return [float(score) for score in scores]

    def _get_model(self):
        if self._model is not None:
            return self._model
        from sentence_transformers import CrossEncoder

        model_path = resolve_local_model_path(self.settings.local_reranker_model)
        model_log(
            "Loading local reranker model "
            f"{self.settings.local_reranker_model!r} on {self.settings.local_reranker_device!r}; "
            f"resolved={model_path}"
        )
        self._model = CrossEncoder(
            model_path,
            device=self.settings.local_reranker_device,
        )
        model_log(f"Local reranker model loaded: {self.settings.local_reranker_model}")
        return self._model


def is_local_reranker_available(settings: Settings) -> bool:
    importlib.invalidate_caches()
    return (
        settings.enable_local_reranker
        and importlib.util.find_spec("sentence_transformers") is not None
    )


def build_reranker(settings: Settings) -> Reranker:
    if is_local_reranker_available(settings):
        return LocalCrossEncoderReranker(settings)
    return NoopReranker()


def normalize_scores(scores: list[float]) -> list[float]:
    finite_scores = [score for score in scores if math.isfinite(score)]
    if not finite_scores:
        return [0.0 for _ in scores]
    low = min(finite_scores)
    high = max(finite_scores)
    if high <= low:
        return [0.5 if math.isfinite(score) else 0.0 for score in scores]
    return [
        (score - low) / (high - low) if math.isfinite(score) else 0.0
        for score in scores
    ]
