from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .config import Settings


SUPPORTED_VIDEO_EXTENSIONS = {
    ".3gp",
    ".avi",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".mxf",
    ".webm",
}


@dataclass(frozen=True)
class ToolStatus:
    ffmpeg_available: bool
    ffprobe_available: bool
    ffmpeg_path: str | None
    ffprobe_path: str | None


def check_media_tools(settings: Settings) -> ToolStatus:
    return ToolStatus(
        ffmpeg_available=shutil.which(settings.ffmpeg_bin) is not None,
        ffprobe_available=shutil.which(settings.ffprobe_bin) is not None,
        ffmpeg_path=shutil.which(settings.ffmpeg_bin),
        ffprobe_path=shutil.which(settings.ffprobe_bin),
    )


def iter_video_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS:
            yield path


def probe_duration(path: Path, settings: Settings) -> float | None:
    command = [
        settings.ffprobe_bin,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    payload = json.loads(result.stdout or "{}")
    raw_duration = payload.get("format", {}).get("duration")
    if raw_duration is None:
        return None
    return float(raw_duration)


def extract_audio_chunk(
    media_path: Path,
    output_path: Path,
    start_seconds: float,
    duration_seconds: float,
    settings: Settings,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        settings.ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_seconds:.3f}",
        "-t",
        f"{duration_seconds:.3f}",
        "-i",
        str(media_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-b:a",
        settings.audio_bitrate,
        str(output_path),
    ]
    subprocess.run(command, check=True, capture_output=True)

