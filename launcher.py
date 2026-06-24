from __future__ import annotations

import importlib.util
import json
import os
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
import webbrowser
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PID_FILE = ROOT / ".pmm_data" / "server.pid"
LOG_FILE = ROOT / ".pmm_data" / "logs" / "startup.log"
MODEL_LOG_FILE = ROOT / ".pmm_data" / "logs" / "model.log"
AI_STATUS_FILE = ROOT / ".pmm_data" / "ai-deps.status.txt"
AI_PROGRESS_FILE = ROOT / ".pmm_data" / "ai-deps.progress.json"
AI_REQUIREMENTS = ROOT / "requirements-ai.txt"
PIP_INDEX_URL = "https://pypi.tuna.tsinghua.edu.cn/simple"
PIP_INDEX_URLS = [
    PIP_INDEX_URL,
    "https://mirrors.aliyun.com/pypi/simple",
    "https://mirrors.cloud.tencent.com/pypi/simple",
]
HF_ENDPOINT = "https://hf-mirror.com"
PACKAGE_INSTALL_ORDER = [
    "torch",
    "torchaudio",
    "funasr",
    "modelscope",
    "sentence-transformers",
    "opencc-python-reimplemented",
]
MAX_SAFE_ROOT_LENGTH = 70
MAX_SAFE_MODEL_PATH_LENGTH = 240


def set_env_path(name: str, path: Path) -> None:
    os.environ.setdefault(name, str(path))


def set_forced_env_path(name: str, path: Path) -> None:
    os.environ[name] = str(path)


def log(message: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with LOG_FILE.open("a", encoding="utf-8") as file:
            file.write(line + "\n")
    except OSError:
        pass


def model_log(message: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    try:
        MODEL_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with MODEL_LOG_FILE.open("a", encoding="utf-8") as file:
            file.write(line + "\n")
    except OSError:
        pass


def write_status(message: str) -> None:
    try:
        AI_STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
        AI_STATUS_FILE.write_text(message + "\n", encoding="utf-8")
    except OSError:
        pass


def read_status() -> str:
    try:
        return (AI_STATUS_FILE.read_text(encoding="utf-8").splitlines() or [""])[0].strip()
    except OSError:
        return ""


def write_ai_progress(
    status: str,
    stage: str,
    message: str,
    current: int = 0,
    total: int = 0,
    package: str = "",
    percent: float | None = None,
) -> None:
    if percent is None:
        percent = 100.0 if status == "ready" else (current / total * 100.0 if total else 0.0)
    payload = {
        "status": status,
        "stage": stage,
        "message": message,
        "current": current,
        "total": total,
        "package": package,
        "percent": max(0.0, min(100.0, float(percent))),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dependency_install_dir": str(ROOT / "python" / "Lib" / "site-packages"),
        "model_cache_dir": os.environ.get("MODELSCOPE_CACHE")
        or os.environ.get("SENTENCE_TRANSFORMERS_HOME")
        or str(ROOT / ".pmm_data"),
    }
    try:
        AI_PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
        AI_PROGRESS_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass


def requirement_name(requirement: str) -> str:
    for separator in ("==", ">=", "<=", "~=", "!=", ">", "<"):
        if separator in requirement:
            return requirement.split(separator, 1)[0].strip()
    return requirement.strip()


def requirement_sort_key(requirement: str) -> tuple[int, str]:
    name = requirement_name(requirement).lower()
    if name in PACKAGE_INSTALL_ORDER:
        return (PACKAGE_INSTALL_ORDER.index(name), name)
    return (len(PACKAGE_INSTALL_ORDER), name)


def requirement_lines() -> list[str]:
    if not AI_REQUIREMENTS.exists():
        return []
    requirements: list[str] = []
    for line in AI_REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value and not value.startswith("#"):
            requirements.append(value)
    return sorted(requirements, key=requirement_sort_key)


def remove_user_site_from_sys_path() -> list[str]:
    try:
        import site
    except Exception:
        return []
    try:
        raw_paths = site.getusersitepackages()
    except Exception:
        return []
    paths = raw_paths if isinstance(raw_paths, list) else [raw_paths]
    normalized = {str(Path(path)).lower() for path in paths if path}
    removed = [path for path in sys.path if str(Path(path)).lower() in normalized]
    sys.path[:] = [path for path in sys.path if str(Path(path)).lower() not in normalized]
    return removed


def configure_environment() -> None:
    os.chdir(ROOT)
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ.pop("PYTHONPATH", None)
    os.environ.setdefault("PIP_INDEX_URL", PIP_INDEX_URL)
    os.environ.setdefault("PIP_TRUSTED_HOST", "pypi.tuna.tsinghua.edu.cn")
    os.environ.setdefault("PIP_RETRIES", "5")
    os.environ.setdefault("PIP_TIMEOUT", "120")
    os.environ.setdefault("HF_ENDPOINT", HF_ENDPOINT)
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    data_dir = ROOT / ".pmm_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    set_env_path("PMM_DATA_DIR", data_dir)
    set_forced_env_path("MODELSCOPE_CACHE", data_dir / "ms")
    set_forced_env_path("HF_HOME", data_dir / "hf")
    set_forced_env_path("HF_HUB_CACHE", data_dir / "hf" / "hub")
    set_forced_env_path("TRANSFORMERS_CACHE", data_dir / "tf")
    set_forced_env_path("SENTENCE_TRANSFORMERS_HOME", data_dir / "st")
    set_forced_env_path("TORCH_HOME", data_dir / "torch")
    for name in (
        "MODELSCOPE_CACHE",
        "HF_HOME",
        "HF_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "SENTENCE_TRANSFORMERS_HOME",
        "TORCH_HOME",
    ):
        Path(os.environ[name]).mkdir(parents=True, exist_ok=True)

    ffmpeg_bin = ROOT / "tools" / "ffmpeg" / "bin"
    python_dir = ROOT / "python"
    os.environ["PATH"] = os.pathsep.join(
        [str(ffmpeg_bin), str(python_dir), str(python_dir / "Scripts"), os.environ.get("PATH", "")]
    )
    removed_user_paths = remove_user_site_from_sys_path()
    if removed_user_paths:
        log("Removed user site-packages from sys.path: " + "; ".join(removed_user_paths))


def path_safety_warning() -> str | None:
    if os.name != "nt":
        return None
    estimated_model_file = (
        ROOT
        / ".pmm_data"
        / "st"
        / "models--BAAI--bge-small-zh-v1.5"
        / "snapshots"
        / "7999e1d3359715c523056ef9478215996d62a620"
        / "sentence_transformers_config.json"
    )
    estimated_dependency_file = (
        ROOT
        / "python"
        / "Lib"
        / "site-packages"
        / "modelscope"
        / "msdatasets"
        / "dataset_cls"
        / "custom_datasets"
        / "image_quality_assessment_degradation"
        / "image_quality_assessment_degradation_dataset.py"
    )
    root_length = len(str(ROOT))
    estimated_length = len(str(estimated_model_file))
    dependency_length = len(str(estimated_dependency_file))
    if (
        root_length <= MAX_SAFE_ROOT_LENGTH
        and estimated_length <= MAX_SAFE_MODEL_PATH_LENGTH
        and dependency_length <= MAX_SAFE_MODEL_PATH_LENGTH
    ):
        return None
    return (
        "App path is too long for first-run dependency/model downloads. "
        f"Current root length: {root_length}; estimated model path length: {estimated_length}; "
        f"estimated dependency path length: {dependency_length}. "
        "Please move the whole app folder to a short path such as C:\\PMM or E:\\PMM, then run Start again. "
        f"Current path: {ROOT}"
    )


def optional_ai_dependencies_ready() -> bool:
    importlib.invalidate_caches()
    required_modules = ["funasr", "modelscope", "opencc", "sentence_transformers", "torch", "torchaudio"]
    return all(importlib.util.find_spec(module) is not None for module in required_modules)


def install_optional_ai_dependencies() -> bool:
    if optional_ai_dependencies_ready():
        write_status("checking")
        write_ai_progress(
            "checking",
            "dependencies-ready",
            "AI dependencies are installed. Checking local models.",
            1,
            1,
            percent=55,
        )
        log("AI dependencies are already installed.")
        return True

    if os.environ.get("PMM_SKIP_AI_DEP_INSTALL", "").strip().lower() in {"1", "true", "yes"}:
        write_status("skipped")
        write_ai_progress("skipped", "skipped", "AI dependency installation was skipped.")
        log("AI dependencies are not installed because PMM_SKIP_AI_DEP_INSTALL is enabled.")
        return False

    if not AI_REQUIREMENTS.exists():
        write_status("missing-requirements")
        write_ai_progress("missing-requirements", "failed", f"Missing AI requirements file: {AI_REQUIREMENTS}")
        log(f"Missing AI requirements file: {AI_REQUIREMENTS}")
        return False

    requirements = requirement_lines()
    if not requirements:
        write_status("missing-requirements")
        write_ai_progress("missing-requirements", "failed", f"AI requirements file is empty: {AI_REQUIREMENTS}")
        log(f"AI requirements file is empty: {AI_REQUIREMENTS}")
        return False

    write_status("installing")
    write_ai_progress("installing", "prepare", "Preparing AI dependency installation.", 0, len(requirements), percent=2)
    log("First run needs local speech recognition and semantic search dependencies.")
    log("The app will open first. Transcription and local semantic search stay locked until this preparation finishes.")
    log(f"pip indexes: {', '.join(PIP_INDEX_URLS)}")

    total = len(requirements)
    pip_python = ROOT / "python" / "python.exe"
    pip_env = os.environ.copy()
    pip_env["PYTHONNOUSERSITE"] = "1"
    pip_env.pop("PYTHONPATH", None)

    for index, requirement in enumerate(requirements, start=1):
        package_name = requirement_name(requirement)
        start_percent = ((index - 1) / total) * 53.0 + 2.0
        write_ai_progress(
            "installing",
            "installing",
            f"Installing {requirement}",
            index,
            total,
            package_name,
            start_percent,
        )
        log(f"Installing AI dependency {index}/{total}: {requirement}")
        latest_percent = start_percent
        return_code = 1
        for pip_index in PIP_INDEX_URLS:
            log(f"Using pip index: {pip_index}")
            write_ai_progress(
                "installing",
                "downloading",
                f"Using pip index: {pip_index}",
                index,
                total,
                package_name,
                latest_percent,
            )
            command = [
                str(pip_python),
                "-s",
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--progress-bar",
                "on",
                "--retries",
                "5",
                "--timeout",
                "120",
                "-i",
                pip_index,
                requirement,
            ]
            process = subprocess.Popen(
                command,
                cwd=str(ROOT),
                env=pip_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            assert process.stdout is not None
            for raw_line in process.stdout:
                line = raw_line.strip()
                if not line:
                    continue
                log(f"pip: {line}")
                latest_percent = min(start_percent + (53.0 / total) * 0.85, latest_percent + 0.8)
                write_ai_progress(
                    "installing",
                    "downloading",
                    line,
                    index,
                    total,
                    package_name,
                    latest_percent,
                )
            return_code = process.wait()
            if return_code == 0:
                break
            log(f"pip index failed for {requirement}: {pip_index}")
        if return_code != 0:
            write_status("failed")
            message = f"{requirement} install failed. Check network and run Start again."
            write_ai_progress("failed", "failed", message, index, total, package_name, latest_percent)
            log("AI dependency auto-install failed. The app is still running, but transcription and local semantic search are not available yet.")
            log(message)
            return False

        write_ai_progress(
            "installing",
            "installed",
            f"{requirement} installed.",
            index,
            total,
            package_name,
            (index / total) * 53.0 + 2.0,
        )
        log(f"AI dependency {index}/{total} installed: {requirement}")

    importlib.invalidate_caches()
    ready = optional_ai_dependencies_ready()
    write_status("checking" if ready else "failed")
    write_ai_progress(
        "checking" if ready else "failed",
        "dependencies-ready" if ready else "failed",
        "AI dependencies are ready. Checking local models." if ready else "AI dependency install finished, but verification failed.",
        total,
        total,
        percent=55 if ready else 98,
    )
    log("AI dependencies are ready." if ready else "AI dependency install finished, but verification failed.")
    return ready


def prepare_ai_runtime() -> bool:
    if not install_optional_ai_dependencies():
        return False
    return preflight_local_models()


def preflight_local_models() -> bool:
    steps = [
        (
            "speech-model",
            "Checking local speech recognition model.",
            """
from app.config import load_settings
from app.transcription import build_transcriber
settings = load_settings()
transcriber = build_transcriber(settings)
print(f"speech model ready: {settings.transcription_backend} / {settings.funasr_model}")
""",
        ),
        (
            "embedding-model",
            "Checking local semantic embedding model.",
            """
from app.config import load_settings
from app.embeddings import build_embedding_client, embedding_cache_folder
settings = load_settings()
client = build_embedding_client(settings)
vector = client.embed_texts(["环境自检"])[0]
print(f"embedding model ready: {settings.local_embedding_model} / dim={len(vector)} / cache={embedding_cache_folder()}")
""",
        ),
        (
            "reranker-model",
            "Checking local reranker model if enabled.",
            """
from app.config import load_settings
from app.reranker import build_reranker
settings = load_settings()
if settings.enable_local_reranker:
    reranker = build_reranker(settings)
    scores = reranker.score_pairs("环境自检", ["环境自检"])
    print(f"reranker model ready: {settings.local_reranker_model} / scores={len(scores)}")
else:
    print("reranker model skipped: disabled")
""",
        ),
    ]
    total = len(steps)
    for index, (stage, message, code) in enumerate(steps, start=1):
        start_percent = 55.0 + ((index - 1) / total) * 42.0
        end_percent = 55.0 + (index / total) * 42.0
        write_status("checking")
        write_ai_progress("checking", stage, message, index, total, stage, start_percent)
        log(message)
        model_log(message)
        ok = run_preflight_python(code, stage, index, total, start_percent, end_percent)
        if not ok:
            return False
    write_status("ready")
    write_ai_progress("ready", "done", "Environment is ready.", total, total, percent=100)
    log("Environment is ready.")
    model_log("Environment is ready.")
    return True


def run_preflight_python(
    code: str,
    stage: str,
    current: int,
    total: int,
    start_percent: float,
    end_percent: float,
) -> bool:
    command = [str(ROOT / "python" / "python.exe"), "-s", "-u", "-c", code]
    process = subprocess.Popen(
        command,
        cwd=str(ROOT),
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    latest_percent = start_percent
    for raw_line in process.stdout:
        line = raw_line.strip()
        if not line:
            continue
        log(f"model: {line}")
        model_log(line)
        latest_percent = min(end_percent - 1.0, latest_percent + 0.7)
        write_ai_progress("checking", stage, line, current, total, stage, latest_percent)
    return_code = process.wait()
    if return_code != 0:
        message = f"{stage} failed. See {MODEL_LOG_FILE} for details."
        write_status("failed")
        write_ai_progress("failed", stage, message, current, total, stage, latest_percent)
        log(message)
        model_log(message)
        return False
    write_ai_progress("checking", stage, f"{stage} passed.", current, total, stage, end_percent)
    model_log(f"{stage} passed.")
    return True


def install_optional_ai_dependencies_in_background() -> threading.Thread:
    thread = threading.Thread(target=prepare_ai_runtime, daemon=True)
    thread.start()
    return thread


def mark_ai_runtime_ready(message: str = "Environment is ready.") -> None:
    write_status("ready")
    write_ai_progress("ready", "done", message, 1, 1, percent=100)


def port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) != 0


def choose_port(start: int = 8000, stop: int = 8020) -> int:
    for port in range(start, stop + 1):
        if port_is_free(port):
            return port
    raise RuntimeError("No free local port found from 8000 to 8020.")


def write_pid_file(port: int) -> None:
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(
        f"pid={os.getpid()}\nparent_pid={os.getppid()}\nport={port}\nroot={ROOT}\n",
        encoding="utf-8",
    )


def remove_pid_file() -> None:
    try:
        if PID_FILE.exists():
            PID_FILE.unlink()
    except OSError:
        pass


def open_browser_when_ready(url: str) -> None:
    health_url = f"{url}/api/health"
    log("Waiting for local service to become ready.")
    for _ in range(90):
        try:
            with urllib.request.urlopen(health_url, timeout=1.0):
                log(f"Startup succeeded. Opening browser: {url}")
                webbrowser.open(url)
                return
        except Exception:
            time.sleep(1)
    log(f"Service response is slow. Please open this URL manually: {url}")
    webbrowser.open(url)


def main() -> int:
    configure_environment()
    log("=" * 60)
    log("Photography Material Manager starting.")
    log(f"App directory: {ROOT}")
    log(f"Log file: {LOG_FILE}")
    log("First run downloads AI dependencies and model files. Downloads are saved under .pmm_data.")
    log("=" * 60)

    warning = path_safety_warning()
    path_too_long = bool(warning)
    if path_too_long:
        write_status("path-too-long")
        write_ai_progress("path-too-long", "blocked", warning, percent=0)
        log("Path is too long. Starting the web UI to show migration guidance.")
        log(warning)
        ai_runtime_ready = False
    else:
        previous_status = read_status()
        ai_runtime_ready = previous_status == "ready" and optional_ai_dependencies_ready()
        if ai_runtime_ready:
            mark_ai_runtime_ready("Environment is ready. Models will load silently when needed.")
            log("AI runtime is already prepared. Skipping visible first-run checks.")
        else:
            write_status("checking")
            write_ai_progress(
                "checking",
                "checking",
                "Preparing first-run environment check.",
                percent=0,
            )

    port = choose_port()
    url = f"http://127.0.0.1:{port}"
    write_pid_file(port)
    log(f"Local URL: {url}")
    log("Close this window or run stop.bat to stop the service.")
    if path_too_long:
        log("AI preparation is blocked until the app folder is moved to a shorter path.")
    elif ai_runtime_ready:
        log("AI dependencies and first-run model checks are already complete.")
    else:
        log("AI dependencies will be checked or installed in the background.")
        install_optional_ai_dependencies_in_background()

    if os.environ.get("PMM_NO_AUTO_BROWSER", "").strip().lower() not in {"1", "true", "yes"}:
        threading.Thread(target=open_browser_when_ready, args=(url,), daemon=True).start()

    try:
        import uvicorn
    except Exception as exc:
        log(f"Could not import runtime dependency: {exc}")
        return 1

    try:
        log("Starting local service.")
        uvicorn.run("app.main:app", host="127.0.0.1", port=port, log_level="info")
        log("Service stopped.")
        return 0
    finally:
        remove_pid_file()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        log("Startup failed. Error details:")
        log(traceback.format_exc())
        raise
