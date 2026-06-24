from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import load_settings
from .database import Database
from .embeddings import embedding_status
from .llm_assistant import is_llm_available
from .media import check_media_tools
from .media import SUPPORTED_VIDEO_EXTENSIONS
from .reranker import is_local_reranker_available
from .service import MaterialService
from .transcription import transcription_backend_status


settings = load_settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
settings.audio_dir.mkdir(parents=True, exist_ok=True)

db = Database(settings.db_path)
service = MaterialService(settings, db)

app = FastAPI(title="Photography Material Manager", version="0.1.0")
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.on_event("startup")
def warm_ready_runtime() -> None:
    if _ai_dependency_status() != "ready":
        return
    threading.Thread(target=service.warm_runtime, daemon=True).start()


class IngestRequest(BaseModel):
    directory: str = Field(..., min_length=1)
    limit: Optional[int] = Field(default=None, ge=1)
    start_processing: bool = True


class IngestFilesRequest(BaseModel):
    paths: list[str] = Field(..., min_length=1)
    limit: Optional[int] = Field(default=None, ge=1)
    start_processing: bool = True


class AssistantSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    limit: int = Field(default=12, ge=1, le=50)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (static_dir / "index.html").read_text(encoding="utf-8")


@app.get("/api/health")
def health() -> dict:
    tools = check_media_tools(settings)
    backend_status = transcription_backend_status(settings)
    embed_status = embedding_status(settings)
    transcription_ok = bool(backend_status["available"])
    semantic_ok = (not settings.enable_embeddings) or bool(embed_status["available"])
    assistant_ok = is_llm_available(settings)
    return {
        "ok": tools.ffmpeg_available and tools.ffprobe_available and transcription_ok,
        "ffmpeg_available": tools.ffmpeg_available,
        "ffprobe_available": tools.ffprobe_available,
        "ffmpeg_path": tools.ffmpeg_path,
        "ffprobe_path": tools.ffprobe_path,
        "openai_api_key_configured": bool(settings.openai_api_key),
        "openai_model": settings.openai_model,
        "transcription_backend": settings.transcription_backend,
        "transcription_available": transcription_ok,
        "transcription_detail": backend_status["detail"],
        "local_whisper_model": settings.local_whisper_model,
        "local_whisper_device": settings.local_whisper_device,
        "local_whisper_language": settings.local_whisper_language,
        "local_whisper_task": settings.local_whisper_task,
        "output_simplified_chinese": settings.output_simplified_chinese,
        "funasr_model": settings.funasr_model,
        "funasr_device": settings.funasr_device,
        "funasr_language": settings.funasr_language,
        "funasr_vad_model": settings.funasr_vad_model,
        "funasr_punc_model": settings.funasr_punc_model,
        "enable_embeddings": settings.enable_embeddings,
        "embedding_backend": settings.embedding_backend,
        "embedding_available": bool(embed_status["available"]),
        "embedding_model": embed_status["model"],
        "embedding_detail": embed_status["detail"],
        "semantic_available": semantic_ok,
        "openai_embedding_model": settings.openai_embedding_model,
        "assistant_enabled": settings.enable_assistant,
        "assistant_available": assistant_ok,
        "llm_base_url": settings.llm_base_url,
        "llm_model": settings.llm_model,
        "local_reranker_enabled": settings.enable_local_reranker,
        "local_reranker_available": is_local_reranker_available(settings),
        "local_reranker_model": settings.local_reranker_model,
        "ai_dependency_status": _ai_dependency_status(),
        "data_dir": str(settings.data_dir),
        "db_path": str(settings.db_path),
        "chunk_seconds": settings.chunk_seconds,
        "chunk_overlap_seconds": settings.chunk_overlap_seconds,
        "audio_bitrate": settings.audio_bitrate,
    }


@app.get("/api/startup-status")
def startup_status() -> dict:
    progress = _ai_dependency_progress()
    status = progress.get("status") or _ai_dependency_status()
    return {
        "ai_dependency_status": status,
        "runtime_ready": status == "ready",
        "ai_dependency_progress": progress,
        "log_path": str(settings.data_dir / "logs" / "startup.log"),
        "model_log_path": str(settings.data_dir / "logs" / "model.log"),
        "dependency_install_dir": progress.get("dependency_install_dir") or _dependency_install_dir(),
        "model_cache_dir": progress.get("model_cache_dir") or _model_cache_dir(),
    }


@app.post("/api/ingest")
def ingest(request: IngestRequest, background_tasks: BackgroundTasks) -> dict:
    if request.start_processing:
        _require_runtime_ready()
    try:
        source = Path(request.directory)
        if source.exists() and source.is_file():
            result = service.scan_files([source], request.limit)
        else:
            result = service.scan_directory(source, request.limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if request.start_processing:
        background_tasks.add_task(service.process_pending)
    return result


@app.post("/api/ingest-files")
def ingest_files(request: IngestFilesRequest, background_tasks: BackgroundTasks) -> dict:
    if request.start_processing:
        _require_runtime_ready()
    try:
        result = service.scan_files([Path(path) for path in request.paths], request.limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if request.start_processing:
        background_tasks.add_task(service.process_pending)
    return result


@app.get("/api/pick-directory")
def pick_directory() -> dict:
    selected = _pick_directory_dialog()
    return {"path": selected}


@app.get("/api/pick-files")
def pick_files() -> dict:
    selected = _pick_files_dialog()
    return {"paths": selected}


@app.post("/api/process")
def process(background_tasks: BackgroundTasks) -> dict:
    _require_runtime_ready()
    background_tasks.add_task(service.process_pending)
    return {"started": True}


@app.post("/api/retry-failed")
def retry_failed(background_tasks: BackgroundTasks) -> dict:
    _require_runtime_ready()
    count = service.retry_failed()
    if count:
        background_tasks.add_task(service.process_pending)
    return {"reset": count, "started": bool(count)}


@app.post("/api/semantic-index")
def semantic_index(background_tasks: BackgroundTasks) -> dict:
    _require_runtime_ready()
    background_tasks.add_task(service.embed_missing_segments)
    return {"started": True}


@app.get("/api/progress")
def progress() -> dict:
    return service.progress()


@app.get("/api/media")
def list_media(limit: int = Query(default=100, ge=1, le=500)) -> list[dict]:
    return service.list_media(limit=limit)


@app.get("/api/search")
def search(q: str = Query(default="", min_length=0), limit: int = Query(default=50, ge=1, le=200)) -> dict:
    _require_runtime_ready()
    results = service.search(q, limit=limit)
    return {"query": q, "count": len(results), "results": [result.__dict__ for result in results]}


@app.post("/api/assistant-search")
def assistant_search(request: AssistantSearchRequest) -> dict:
    _require_runtime_ready()
    return service.assistant_search(request.query, limit=request.limit)


@app.post("/api/smart-search/stream")
@app.post("/api/assistant-search/stream")
def smart_search_stream(request: AssistantSearchRequest) -> StreamingResponse:
    _require_runtime_ready()
    def event_stream():
        for event in service.assistant_search_stream(request.query, limit=request.limit):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/export.csv")
def export_csv(q: str = Query(default="", min_length=0)) -> Response:
    _require_runtime_ready()
    csv_text = service.export_csv(q)
    return Response(
        csv_text,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="material-search-results.csv"'},
    )


@app.get("/api/media/{media_id}/file")
def media_file(media_id: int) -> FileResponse:
    media = service.media_by_id(media_id)
    if media is None:
        raise HTTPException(status_code=404, detail="Media not found")
    path = Path(str(media["path"]))
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Media file no longer exists")
    return FileResponse(path)


def _ai_dependency_status() -> str:
    status_path = settings.data_dir / "ai-deps.status.txt"
    if not status_path.exists():
        return "unknown"
    try:
        return (status_path.read_text(encoding="utf-8").splitlines() or ["unknown"])[0].strip() or "unknown"
    except OSError:
        return "unknown"


def _require_runtime_ready() -> None:
    status = _ai_dependency_status()
    if status in {"ready", "unknown"}:
        return
    progress = _ai_dependency_progress()
    message = progress.get("message") or "本地 AI 环境仍在准备中，请等待首次启动自检完成。"
    raise HTTPException(
        status_code=503,
        detail=f"本地 AI 环境未就绪（{status}）：{message}",
    )


def _dependency_install_dir() -> str:
    executable = Path(sys.executable)
    if executable.parent.name.lower() == "python":
        return str(executable.parent / "Lib" / "site-packages")
    return str(executable.parent.parent / "Lib" / "site-packages")


def _model_cache_dir() -> str:
    return (
        os.getenv("MODELSCOPE_CACHE")
        or os.getenv("SENTENCE_TRANSFORMERS_HOME")
        or os.getenv("HF_HOME")
        or str(settings.data_dir)
    )


def _ai_dependency_progress() -> dict:
    progress_path = settings.data_dir / "ai-deps.progress.json"
    if not progress_path.exists():
        return {
            "status": _ai_dependency_status(),
            "stage": "unknown",
            "message": "",
            "current": 0,
            "total": 0,
            "package": "",
            "percent": 0,
            "dependency_install_dir": _dependency_install_dir(),
            "model_cache_dir": _model_cache_dir(),
        }
    try:
        payload = json.loads(progress_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "status": _ai_dependency_status(),
            "stage": "unknown",
            "message": "",
            "current": 0,
            "total": 0,
            "package": "",
            "percent": 0,
            "dependency_install_dir": _dependency_install_dir(),
            "model_cache_dir": _model_cache_dir(),
        }
    if not isinstance(payload, dict):
        return {}
    payload.setdefault("status", _ai_dependency_status())
    payload.setdefault("dependency_install_dir", _dependency_install_dir())
    payload.setdefault("model_cache_dir", _model_cache_dir())
    return payload


def _startup_log_tail(limit: int = 8) -> list[str]:
    return _log_tail(settings.data_dir / "logs" / "startup.log", limit)


def _model_log_tail(limit: int = 8) -> list[str]:
    return _log_tail(settings.data_dir / "logs" / "model.log", limit)


def _log_tail(log_path: Path, limit: int) -> list[str]:
    if not log_path.exists():
        return []
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    return lines[-limit:]


def _pick_directory_dialog() -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"无法打开文件夹选择窗口：{exc}") from exc

    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
        root.update()
        selected = filedialog.askdirectory(title="选择素材文件夹", mustexist=True)
        return str(selected or "")
    finally:
        root.destroy()


def _pick_files_dialog() -> list[str]:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"无法打开文件选择窗口：{exc}") from exc

    video_patterns = " ".join(f"*{extension}" for extension in sorted(SUPPORTED_VIDEO_EXTENSIONS))
    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
        root.update()
        selected = filedialog.askopenfilenames(
            title="选择素材文件",
            filetypes=(("视频文件", video_patterns), ("所有文件", "*.*")),
        )
        return [str(path) for path in selected]
    finally:
        root.destroy()
