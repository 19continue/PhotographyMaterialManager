from __future__ import annotations

import importlib.util
import os
import re
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Protocol

import httpx

from .config import Settings


class TranscriptionError(RuntimeError):
    pass


class Transcriber(Protocol):
    def transcribe(
        self,
        audio_path: Path,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> dict[str, Any]:
        ...


LOCAL_WHISPER_BACKENDS = {"local-whisper", "whisper-local", "local"}
FUNASR_BACKENDS = {"funasr", "funasr-sensevoice", "sensevoice", "sensevoice-small"}


_OPENCC_CONVERTER = None


def is_local_whisper_backend(settings: Settings) -> bool:
    return settings.transcription_backend in LOCAL_WHISPER_BACKENDS


def is_local_whisper_available() -> bool:
    return importlib.util.find_spec("whisper") is not None


def is_funasr_backend(settings: Settings) -> bool:
    return settings.transcription_backend in FUNASR_BACKENDS


def is_funasr_available() -> bool:
    return (
        importlib.util.find_spec("funasr") is not None
        and importlib.util.find_spec("torchaudio") is not None
    )


def transcription_backend_status(settings: Settings) -> dict[str, Any]:
    if settings.transcription_backend == "openai":
        return {
            "backend": "openai",
            "available": bool(settings.openai_api_key),
            "detail": "OPENAI_API_KEY configured"
            if settings.openai_api_key
            else "OPENAI_API_KEY is not configured",
        }
    if is_local_whisper_backend(settings):
        available = is_local_whisper_available()
        return {
            "backend": "local-whisper",
            "available": available,
            "model": settings.local_whisper_model,
            "device": settings.local_whisper_device,
            "language": settings.local_whisper_language,
            "task": settings.local_whisper_task,
            "simplified_chinese": settings.output_simplified_chinese,
            "detail": "openai-whisper package is installed"
            if available
            else "Install optional dependency: pip install -U openai-whisper",
        }
    if is_funasr_backend(settings):
        available = is_funasr_available()
        return {
            "backend": "funasr-sensevoice",
            "available": available,
            "model": settings.funasr_model,
            "device": settings.funasr_device,
            "language": settings.funasr_language,
            "vad_model": settings.funasr_vad_model,
            "punc_model": settings.funasr_punc_model,
            "simplified_chinese": settings.output_simplified_chinese,
            "detail": "funasr and torchaudio are installed"
            if available
            else "Install optional dependency: pip install funasr modelscope torchaudio",
        }
    return {
        "backend": settings.transcription_backend,
        "available": False,
        "detail": "Unsupported transcription backend",
    }


class OpenAITranscriber:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def transcribe(
        self,
        audio_path: Path,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> dict[str, Any]:
        if not self.settings.openai_api_key:
            raise TranscriptionError("OPENAI_API_KEY is not configured.")

        with audio_path.open("rb") as audio_file:
            files = {"file": (audio_path.name, audio_file, "audio/mpeg")}
            data = [
                ("model", self.settings.openai_model),
                ("response_format", "verbose_json"),
                ("timestamp_granularities[]", "word"),
            ]
            headers = {"Authorization": f"Bearer {self.settings.openai_api_key}"}
            with httpx.Client(timeout=httpx.Timeout(600.0, connect=30.0)) as client:
                response = client.post(
                    "https://api.openai.com/v1/audio/transcriptions",
                    headers=headers,
                    data=data,
                    files=files,
                )

        if response.status_code >= 400:
            raise TranscriptionError(
                f"OpenAI transcription failed: {response.status_code} {response.text[:500]}"
            )
        payload = response.json()
        if not isinstance(payload, dict):
            raise TranscriptionError("OpenAI transcription returned an unexpected payload.")
        return payload


class LocalWhisperTranscriber:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._ensure_ffmpeg_path()
        try:
            import whisper  # type: ignore
        except ImportError as exc:
            raise TranscriptionError(
                "Local Whisper backend requires the optional package 'openai-whisper'. "
                "Install it with: pip install -U openai-whisper"
            ) from exc
        self._whisper = whisper
        self._model = whisper.load_model(
            settings.local_whisper_model,
            device=settings.local_whisper_device,
        )

    def _ensure_ffmpeg_path(self) -> None:
        ffmpeg_parent = Path(self.settings.ffmpeg_bin).expanduser().parent
        if ffmpeg_parent.exists():
            current_path = os.environ.get("PATH", "")
            paths = current_path.split(os.pathsep) if current_path else []
            if str(ffmpeg_parent) not in paths:
                os.environ["PATH"] = str(ffmpeg_parent) + os.pathsep + current_path

    def transcribe(
        self,
        audio_path: Path,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> dict[str, Any]:
        fp16 = self.settings.local_whisper_device.lower() not in {"cpu", "mps"}
        options: dict[str, Any] = {
            "verbose": False,
            "fp16": fp16,
            "word_timestamps": True,
            "task": self.settings.local_whisper_task,
        }
        if self.settings.local_whisper_language:
            options["language"] = self.settings.local_whisper_language
        result = self._model.transcribe(
            str(audio_path),
            **options,
        )
        if not isinstance(result, dict):
            raise TranscriptionError("Local Whisper returned an unexpected payload.")
        if self.settings.output_simplified_chinese:
            result = simplify_chinese_payload(result)
        return result


class FunASRSenseVoiceTranscriber:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._ensure_ffmpeg_path()
        try:
            from funasr import AutoModel
        except ImportError as exc:
            raise TranscriptionError(
                "FunASR backend requires optional dependencies. "
                "Install them with: pip install funasr modelscope torchaudio"
            ) from exc

        kwargs: dict[str, Any] = {
            "model": settings.funasr_model,
            "device": settings.funasr_device,
            "disable_update": True,
        }
        if settings.funasr_vad_model:
            kwargs["vad_model"] = settings.funasr_vad_model
        if settings.funasr_punc_model:
            kwargs["punc_model"] = settings.funasr_punc_model
        self._model = AutoModel(**kwargs)

    def _ensure_ffmpeg_path(self) -> None:
        ffmpeg_parent = Path(self.settings.ffmpeg_bin).expanduser().parent
        if ffmpeg_parent.exists():
            current_path = os.environ.get("PATH", "")
            paths = current_path.split(os.pathsep) if current_path else []
            if str(ffmpeg_parent) not in paths:
                os.environ["PATH"] = str(ffmpeg_parent) + os.pathsep + current_path

    def transcribe(
        self,
        audio_path: Path,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> dict[str, Any]:
        raw_result = self._model.generate(
            input=str(audio_path),
            progress_callback=progress_callback,
            language=self.settings.funasr_language or "auto",
            use_itn=True,
            output_timestamp=True,
            sentence_timestamp=True,
            merge_vad=True,
            merge_length_s=self.settings.funasr_merge_length_s,
            batch_size_s=self.settings.funasr_batch_size_s,
            disable_pbar=True,
        )
        payload = self._to_whisper_payload(raw_result)
        if self.settings.output_simplified_chinese:
            payload = simplify_chinese_payload(payload)
        return payload

    def _to_whisper_payload(self, raw_result: Any) -> dict[str, Any]:
        item = raw_result[0] if isinstance(raw_result, list) and raw_result else raw_result
        if not isinstance(item, dict):
            raise TranscriptionError("FunASR returned an unexpected payload.")

        text = clean_asr_text(str(item.get("text") or ""))
        segments = self._segments_from_words(item)
        if segments:
            text = "".join(segment["text"] for segment in segments)
        else:
            segments = self._segments_from_sentence_info(item)
        if not segments:
            segments = self._segments_from_timestamps(item, text)
        if not segments and text:
            segments = [{"start": 0.0, "end": 0.0, "text": text}]
        return {"text": text, "segments": segments, "words": []}

    def _segments_from_words(self, item: dict[str, Any]) -> list[dict[str, Any]]:
        words = item.get("words") or []
        timestamps = item.get("timestamp") or []
        if not isinstance(words, list) or not isinstance(timestamps, list):
            return []
        valid_timestamps = [
            stamp
            for stamp in timestamps
            if isinstance(stamp, (list, tuple)) and len(stamp) >= 2
        ]
        if not words or len(words) != len(valid_timestamps):
            return []

        output: list[dict[str, Any]] = []
        buffer: list[str] = []
        start_ms: Any = None
        end_ms: Any = None

        def flush() -> None:
            nonlocal start_ms, end_ms
            text = _strip_asr_markup("".join(buffer))
            buffer.clear()
            if not text or not _has_content_text(text):
                start_ms = None
                end_ms = None
                return
            start = _milliseconds_to_seconds(start_ms, 0.0)
            end = _milliseconds_to_seconds(end_ms, start)
            output.append({"start": start, "end": max(start, end), "text": text})
            start_ms = None
            end_ms = None

        for word, timestamp in zip(words, valid_timestamps):
            part = _strip_asr_markup(str(word))
            if not part:
                continue
            if start_ms is None:
                start_ms = timestamp[0]
            end_ms = timestamp[1]
            buffer.append(part)

            current_text = "".join(buffer).strip()
            current_length = len(current_text)
            has_sentence_end = any(char in _SENTENCE_END_CHARS for char in part)
            has_soft_break = any(char in _SOFT_BREAK_CHARS for char in part)
            if (
                (has_sentence_end and current_length >= _MIN_SEGMENT_CHARS)
                or (has_soft_break and current_length >= _SOFT_SPLIT_CHARS)
                or current_length >= _HARD_SPLIT_CHARS
            ):
                flush()

        if buffer:
            flush()
        return output

    def _segments_from_sentence_info(self, item: dict[str, Any]) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for sentence in item.get("sentence_info") or []:
            if not isinstance(sentence, dict):
                continue
            text_value = sentence.get("text")
            if isinstance(text_value, list):
                text = clean_asr_text("".join(str(part) for part in text_value))
            else:
                text = clean_asr_text(str(text_value or ""))
            if not text:
                continue
            start = _milliseconds_to_seconds(sentence.get("start"), 0.0)
            end = _milliseconds_to_seconds(sentence.get("end"), start)
            output.append({"start": start, "end": max(start, end), "text": text})
        return output

    def _segments_from_timestamps(self, item: dict[str, Any], text: str) -> list[dict[str, Any]]:
        timestamps = item.get("timestamp") or []
        if not text or not isinstance(timestamps, list) or not timestamps:
            return []
        valid = [
            stamp
            for stamp in timestamps
            if isinstance(stamp, (list, tuple)) and len(stamp) >= 2
        ]
        if not valid:
            return []
        start = _milliseconds_to_seconds(valid[0][0], 0.0)
        end = _milliseconds_to_seconds(valid[-1][1], start)
        return [{"start": start, "end": max(start, end), "text": text}]


def simplify_chinese_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return _simplify_value(payload)


def _simplify_value(value: Any) -> Any:
    if isinstance(value, str):
        return _to_simplified(value)
    if isinstance(value, list):
        return [_simplify_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _simplify_value(item) for key, item in value.items()}
    return value


def _to_simplified(text: str) -> str:
    global _OPENCC_CONVERTER
    try:
        if _OPENCC_CONVERTER is None:
            from opencc import OpenCC

            _OPENCC_CONVERTER = OpenCC("t2s")
        return _OPENCC_CONVERTER.convert(text)
    except Exception:
        return text


_SENTENCE_END_CHARS = set("。！？!?；;")
_SOFT_BREAK_CHARS = set("，,、")
_MIN_SEGMENT_CHARS = 4
_SOFT_SPLIT_CHARS = 45
_HARD_SPLIT_CHARS = 90
_CONTENT_TEXT_RE = re.compile(r"[\w\u3400-\u9fff]")
_SENSEVOICE_TAG_RE = re.compile(r"<\s*\|\s*[^<>]*?\s*\|\s*>")
_SENSEVOICE_EMOJI_RE = re.compile(
    "["
    "\U0001f300-\U0001f5ff"
    "\U0001f600-\U0001f64f"
    "\U0001f680-\U0001f6ff"
    "\U0001f700-\U0001f77f"
    "\U0001f780-\U0001f7ff"
    "\U0001f800-\U0001f8ff"
    "\U0001f900-\U0001f9ff"
    "\U0001fa00-\U0001fa6f"
    "\U0001fa70-\U0001faff"
    "\u2600-\u27bf"
    "]+"
)
_ASR_WHITESPACE_RE = re.compile(r"[ \t\r\n]+")


def clean_asr_text(text: str) -> str:
    clean = _strip_asr_markup(text)
    try:
        from funasr.utils.postprocess_utils import rich_transcription_postprocess

        clean = rich_transcription_postprocess(clean)
    except Exception:
        pass
    return _strip_asr_markup(clean)


def _strip_asr_markup(text: str) -> str:
    clean = _SENSEVOICE_TAG_RE.sub("", text.strip())
    clean = _SENSEVOICE_EMOJI_RE.sub("", clean)
    clean = _ASR_WHITESPACE_RE.sub(" ", clean)
    return clean.strip()


def _has_content_text(text: str) -> bool:
    return _CONTENT_TEXT_RE.search(text) is not None


def _milliseconds_to_seconds(value: Any, fallback: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    if number < 0:
        return fallback
    return number / 1000.0


def build_transcriber(settings: Settings) -> Transcriber:
    if settings.transcription_backend == "openai":
        return OpenAITranscriber(settings)
    if is_local_whisper_backend(settings):
        return LocalWhisperTranscriber(settings)
    if is_funasr_backend(settings):
        return FunASRSenseVoiceTranscriber(settings)
    raise TranscriptionError(
        f"Unsupported transcription backend {settings.transcription_backend!r}. "
        "Use 'openai', 'local-whisper', or 'funasr-sensevoice'."
    )
