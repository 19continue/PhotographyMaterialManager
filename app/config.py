from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {raw!r}") from exc


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a number, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    db_path: Path
    audio_dir: Path
    chunk_seconds: int
    chunk_overlap_seconds: int
    audio_bitrate: str
    openai_api_key: str | None
    openai_model: str
    transcription_backend: str
    local_whisper_model: str
    local_whisper_device: str
    local_whisper_language: str
    local_whisper_task: str
    output_simplified_chinese: bool
    funasr_model: str
    funasr_vad_model: str
    funasr_punc_model: str
    funasr_device: str
    funasr_language: str
    funasr_batch_size_s: int
    funasr_merge_length_s: int
    enable_embeddings: bool
    embedding_backend: str
    local_embedding_model: str
    local_embedding_device: str
    openai_embedding_model: str
    semantic_candidates: int
    semantic_min_score: float
    embedding_query_instruction: str
    search_candidate_multiplier: int
    enable_local_reranker: bool
    local_reranker_model: str
    local_reranker_device: str
    reranker_candidates: int
    enable_assistant: bool
    llm_base_url: str
    llm_api_key: str | None
    llm_model: str
    assistant_candidates: int
    assistant_query_terms: int
    ffmpeg_bin: str
    ffprobe_bin: str


def load_settings() -> Settings:
    _load_dotenv(Path(".env"))
    data_dir = Path(os.getenv("PMM_DATA_DIR", ".pmm_data")).resolve()
    return Settings(
        data_dir=data_dir,
        db_path=data_dir / "materials.db",
        audio_dir=data_dir / "audio_chunks",
        chunk_seconds=_int_env("PMM_CHUNK_SECONDS", 900),
        chunk_overlap_seconds=_int_env("PMM_CHUNK_OVERLAP_SECONDS", 5),
        audio_bitrate=os.getenv("PMM_AUDIO_BITRATE", "32k"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_model=os.getenv("PMM_OPENAI_MODEL", "whisper-1"),
        transcription_backend=os.getenv("PMM_TRANSCRIPTION_BACKEND", "funasr-sensevoice").strip().lower(),
        local_whisper_model=os.getenv("PMM_LOCAL_WHISPER_MODEL", "small"),
        local_whisper_device=os.getenv("PMM_LOCAL_WHISPER_DEVICE", "cpu"),
        local_whisper_language=os.getenv("PMM_LOCAL_WHISPER_LANGUAGE", "zh").strip(),
        local_whisper_task=os.getenv("PMM_LOCAL_WHISPER_TASK", "transcribe").strip().lower(),
        output_simplified_chinese=os.getenv("PMM_OUTPUT_SIMPLIFIED_CHINESE", "true").strip().lower()
        not in {"0", "false", "no", "off"},
        funasr_model=os.getenv("PMM_FUNASR_MODEL", "iic/SenseVoiceSmall").strip(),
        funasr_vad_model=os.getenv("PMM_FUNASR_VAD_MODEL", "fsmn-vad").strip(),
        funasr_punc_model=os.getenv("PMM_FUNASR_PUNC_MODEL", "ct-punc").strip(),
        funasr_device=os.getenv("PMM_FUNASR_DEVICE", "cpu").strip().lower(),
        funasr_language=os.getenv("PMM_FUNASR_LANGUAGE", "zh").strip(),
        funasr_batch_size_s=_int_env("PMM_FUNASR_BATCH_SIZE_S", 60),
        funasr_merge_length_s=_int_env("PMM_FUNASR_MERGE_LENGTH_S", 15),
        enable_embeddings=os.getenv("PMM_ENABLE_EMBEDDINGS", "true").strip().lower()
        not in {"0", "false", "no", "off"},
        embedding_backend=os.getenv("PMM_EMBEDDING_BACKEND", "local").strip().lower(),
        local_embedding_model=os.getenv("PMM_LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"),
        local_embedding_device=os.getenv("PMM_LOCAL_EMBEDDING_DEVICE", "cpu"),
        openai_embedding_model=os.getenv("PMM_OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        semantic_candidates=_int_env("PMM_SEMANTIC_CANDIDATES", 400),
        semantic_min_score=_float_env("PMM_SEMANTIC_MIN_SCORE", 0.36),
        embedding_query_instruction=os.getenv("PMM_EMBEDDING_QUERY_INSTRUCTION", "").strip(),
        search_candidate_multiplier=_int_env("PMM_SEARCH_CANDIDATE_MULTIPLIER", 4),
        enable_local_reranker=os.getenv("PMM_ENABLE_LOCAL_RERANKER", "false").strip().lower()
        not in {"0", "false", "no", "off"},
        local_reranker_model=os.getenv("PMM_LOCAL_RERANKER_MODEL", "BAAI/bge-reranker-base"),
        local_reranker_device=os.getenv("PMM_LOCAL_RERANKER_DEVICE", "cpu"),
        reranker_candidates=_int_env("PMM_RERANKER_CANDIDATES", 80),
        enable_assistant=os.getenv("PMM_ENABLE_ASSISTANT", "true").strip().lower()
        not in {"0", "false", "no", "off"},
        llm_base_url=os.getenv("PMM_LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1").rstrip("/"),
        llm_api_key=os.getenv("PMM_LLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
        llm_model=os.getenv("PMM_LLM_MODEL", "deepseek-v4-flash"),
        assistant_candidates=_int_env("PMM_ASSISTANT_CANDIDATES", 50),
        assistant_query_terms=_int_env("PMM_ASSISTANT_QUERY_TERMS", 12),
        ffmpeg_bin=os.getenv("PMM_FFMPEG_BIN", "ffmpeg"),
        ffprobe_bin=os.getenv("PMM_FFPROBE_BIN", "ffprobe"),
    )
