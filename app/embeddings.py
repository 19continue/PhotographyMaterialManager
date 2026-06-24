from __future__ import annotations

import array
import importlib.util
import logging
import math
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from typing import Protocol

import httpx

from .config import Settings


class EmbeddingError(RuntimeError):
    pass


EMBEDDING_INDEX_VERSION = "v2"
HUGGINGFACE_DOWNLOAD_LOGGER = "huggingface_hub.file_download"
MODELSCOPE_MODEL_ALIASES = {
    "BAAI/bge-small-zh-v1.5": "AI-ModelScope/bge-small-zh-v1.5",
    "BAAI/bge-reranker-base": "BAAI/bge-reranker-base",
}
MODEL_WEIGHT_FILES = {
    "model.safetensors",
    "pytorch_model.bin",
    "tf_model.h5",
    "model.onnx",
}


@dataclass(frozen=True)
class StoredVector:
    values: list[float]
    norm: float


class EmbeddingClient(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...


def embedding_model_key(settings: Settings) -> str:
    if settings.embedding_backend == "local":
        return f"local:{settings.local_embedding_model}:{EMBEDDING_INDEX_VERSION}"
    return f"openai:{settings.openai_embedding_model}:{EMBEDDING_INDEX_VERSION}"


def is_embedding_available(settings: Settings) -> bool:
    if not settings.enable_embeddings:
        return False
    if settings.embedding_backend == "local":
        importlib.invalidate_caches()
        return importlib.util.find_spec("sentence_transformers") is not None
    if settings.embedding_backend == "openai":
        return bool(settings.openai_api_key)
    return False


def embedding_status(settings: Settings) -> dict:
    if not settings.enable_embeddings:
        return {
            "backend": settings.embedding_backend,
            "available": False,
            "model": "",
            "detail": "Embeddings are disabled",
        }
    if settings.embedding_backend == "local":
        available = is_embedding_available(settings)
        return {
            "backend": "local",
            "available": available,
            "model": settings.local_embedding_model,
            "device": settings.local_embedding_device,
            "detail": "sentence-transformers is installed"
            if available
            else "Install optional dependency: pip install sentence-transformers",
        }
    if settings.embedding_backend == "openai":
        return {
            "backend": "openai",
            "available": bool(settings.openai_api_key),
            "model": settings.openai_embedding_model,
            "detail": "OPENAI_API_KEY configured"
            if settings.openai_api_key
            else "OPENAI_API_KEY is not configured",
        }
    return {
        "backend": settings.embedding_backend,
        "available": False,
        "model": "",
        "detail": "Unsupported embedding backend",
    }


class OpenAIEmbeddingClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if not self.settings.openai_api_key:
            raise EmbeddingError("OPENAI_API_KEY is not configured.")

        headers = {
            "Authorization": f"Bearer {self.settings.openai_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.settings.openai_embedding_model,
            "input": texts,
        }
        with httpx.Client(timeout=httpx.Timeout(300.0, connect=30.0)) as client:
            response = client.post(
                "https://api.openai.com/v1/embeddings",
                headers=headers,
                json=payload,
            )

        if response.status_code >= 400:
            raise EmbeddingError(
                f"OpenAI embeddings failed: {response.status_code} {response.text[:500]}"
            )
        body = response.json()
        data = sorted(body.get("data") or [], key=lambda item: int(item.get("index", 0)))
        vectors = [item.get("embedding") for item in data]
        if len(vectors) != len(texts) or not all(isinstance(vector, list) for vector in vectors):
            raise EmbeddingError("OpenAI embeddings returned an unexpected payload.")
        return vectors


class LocalEmbeddingClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model = None

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._get_model()
        vectors = model.encode(
            texts,
            normalize_embeddings=False,
            show_progress_bar=False,
            batch_size=32,
        )
        return [[float(value) for value in row] for row in vectors]

    def _get_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingError(
                "Local embedding backend requires 'sentence-transformers'. "
                "Install it with: pip install sentence-transformers"
            ) from exc
        cache_folder = embedding_cache_folder()
        model_path = resolve_local_model_path(self.settings.local_embedding_model)
        _model_log(
            "Loading local embedding model "
            f"{self.settings.local_embedding_model!r} on {self.settings.local_embedding_device!r}; "
            f"resolved={model_path}; cache={cache_folder}"
        )
        logging.getLogger(HUGGINGFACE_DOWNLOAD_LOGGER).setLevel(logging.CRITICAL)
        try:
            self._model = SentenceTransformer(
                model_path,
                device=self.settings.local_embedding_device,
                cache_folder=cache_folder,
            )
            _model_log(f"Local embedding model loaded: {self.settings.local_embedding_model}")
        except Exception as first_exc:
            _model_log(f"Local embedding model load failed: {first_exc}")
            removed = clear_local_embedding_cache(self.settings.local_embedding_model)
            if not removed:
                _model_log("No matching local embedding cache was found to clear.")
                raise
            _model_log("Cleared local embedding cache: " + "; ".join(removed))
            model_path = resolve_local_model_path(self.settings.local_embedding_model, force_download=True)
            try:
                self._model = SentenceTransformer(
                    model_path,
                    device=self.settings.local_embedding_device,
                    cache_folder=cache_folder,
                )
                _model_log(f"Local embedding model loaded after cache clear: {self.settings.local_embedding_model}")
            except Exception as second_exc:
                _model_log(f"Local embedding model retry failed: {second_exc}")
                raise EmbeddingError(
                    "Local embedding model failed after clearing an incomplete cache. "
                    f"Original error: {first_exc}; retry error: {second_exc}"
                ) from second_exc
        return self._model


def build_embedding_client(settings: Settings) -> EmbeddingClient:
    if settings.embedding_backend == "local":
        return LocalEmbeddingClient(settings)
    if settings.embedding_backend == "openai":
        return OpenAIEmbeddingClient(settings)
    raise EmbeddingError(
        f"Unsupported embedding backend {settings.embedding_backend!r}. "
        "Use 'local' or 'openai'."
    )


def embedding_cache_folder() -> str:
    raw_path = (
        os.getenv("SENTENCE_TRANSFORMERS_HOME")
        or os.getenv("HF_HOME")
        or str(Path(".pmm_data") / "st")
    )
    path = Path(raw_path)
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def resolve_local_model_path(model_name: str, force_download: bool = False) -> str:
    path = Path(model_name)
    if path.exists():
        return str(path)

    backend = os.getenv("PMM_MODEL_DOWNLOAD_BACKEND", "modelscope").strip().lower()
    if backend not in {"modelscope", "auto", "huggingface"}:
        backend = "modelscope"
    if backend in {"modelscope", "auto"}:
        try:
            local_path = download_model_from_modelscope(model_name, force_download=force_download)
            if is_complete_transformer_model(Path(local_path)):
                return local_path
            _model_log(f"Downloaded model is incomplete: {local_path}")
            if not force_download:
                clear_modelscope_cache(MODELSCOPE_MODEL_ALIASES.get(model_name, model_name))
                local_path = download_model_from_modelscope(model_name, force_download=True)
                if is_complete_transformer_model(Path(local_path)):
                    return local_path
            raise EmbeddingError(f"Downloaded model is incomplete: {local_path}")
        except Exception as exc:
            _model_log(f"ModelScope download failed for {model_name}: {exc}")
            if backend == "modelscope":
                raise EmbeddingError(
                    f"ModelScope download failed for {model_name}. See .pmm_data/logs/model.log for details."
                ) from exc
    return model_name


def download_model_from_modelscope(model_name: str, force_download: bool = False) -> str:
    model_id = MODELSCOPE_MODEL_ALIASES.get(model_name, model_name)
    cache_dir = modelscope_cache_folder()
    if force_download:
        clear_modelscope_cache(model_id)
    _model_log(f"Downloading model from ModelScope: {model_id}; cache={cache_dir}")
    try:
        from modelscope import snapshot_download
    except ImportError as exc:
        raise EmbeddingError("ModelScope package is not installed.") from exc
    local_path = snapshot_download(model_id=model_id, cache_dir=str(cache_dir))
    _model_log(f"ModelScope model path: {local_path}")
    return str(local_path)


def modelscope_cache_folder() -> Path:
    path = Path(os.getenv("MODELSCOPE_CACHE") or Path(".pmm_data") / "ms")
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_complete_transformer_model(path: Path) -> bool:
    if not path.exists() or not (path / "config.json").exists():
        return False
    for file in path.rglob("*"):
        if file.is_file() and file.name in MODEL_WEIGHT_FILES:
            return True
    return False


def clear_local_embedding_cache(model_name: str) -> list[str]:
    repo_cache_name = "models--" + model_name.replace("/", "--")
    removed: list[str] = []
    for root in embedding_cache_roots():
        for path in (root / repo_cache_name, root / ".locks" / repo_cache_name):
            if not path.exists():
                continue
            if not is_path_inside(path, root):
                continue
            shutil.rmtree(path, ignore_errors=True)
            removed.append(str(path))
    removed.extend(clear_modelscope_cache(MODELSCOPE_MODEL_ALIASES.get(model_name, model_name)))
    return removed


def clear_modelscope_cache(model_id: str) -> list[str]:
    removed: list[str] = []
    cache_root = modelscope_cache_folder()
    candidates = [
        cache_root / "models" / model_id,
        cache_root / model_id,
    ]
    for path in candidates:
        if not path.exists() or not is_path_inside(path, cache_root):
            continue
        shutil.rmtree(path, ignore_errors=True)
        removed.append(str(path))
    return removed


def embedding_cache_roots() -> list[Path]:
    roots: list[Path] = []
    for value in (
        os.getenv("SENTENCE_TRANSFORMERS_HOME"),
        os.getenv("HF_HUB_CACHE"),
        os.getenv("TRANSFORMERS_CACHE"),
        os.getenv("HF_HOME"),
        str(Path(".pmm_data") / "st"),
        str(Path(".pmm_data") / "huggingface_cache" / "sentence_transformers"),
        str(Path(".pmm_data") / "huggingface_cache" / "hub"),
        str(Path(".pmm_data") / "huggingface_cache" / "transformers"),
    ):
        if not value:
            continue
        root = Path(value)
        roots.append(root)
        if root.name != "hub":
            roots.append(root / "hub")
    unique: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        try:
            key = str(root.resolve()).lower()
        except OSError:
            key = str(root).lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(root)
    return unique


def is_path_inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


def _model_log(message: str) -> None:
    data_dir = Path(os.getenv("PMM_DATA_DIR", ".pmm_data"))
    log_path = data_dir / "logs" / "model.log"
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as file:
            file.write(line + "\n")
    except OSError:
        pass


def model_log(message: str) -> None:
    _model_log(message)


def vector_to_blob(values: Iterable[float]) -> tuple[bytes, int, float]:
    vector = array.array("f", (float(value) for value in values))
    norm = math.sqrt(sum(float(value) * float(value) for value in vector))
    return vector.tobytes(), len(vector), norm


def blob_to_vector(blob: bytes) -> StoredVector:
    vector = array.array("f")
    vector.frombytes(blob)
    values = [float(value) for value in vector]
    norm = math.sqrt(sum(value * value for value in values))
    return StoredVector(values=values, norm=norm)


def cosine_similarity(a: list[float], a_norm: float, b: list[float], b_norm: float) -> float:
    if not a or not b or a_norm <= 0 or b_norm <= 0 or len(a) != len(b):
        return 0.0
    dot = sum(left * right for left, right in zip(a, b))
    return dot / (a_norm * b_norm)
