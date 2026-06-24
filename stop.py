from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
PID_FILE = ROOT / ".pmm_data" / "server.pid"


def normalize(value: str) -> str:
    return value.lower().replace("/", "\\")


def read_pid_values() -> dict[str, str]:
    if not PID_FILE.exists():
        return {}
    values: dict[str, str] = {}
    for line in PID_FILE.read_text(encoding="utf-8", errors="ignore").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def read_int(values: dict[str, str], key: str) -> int | None:
    raw_value = values.get(key, "")
    if not raw_value.isdigit():
        return None
    return int(raw_value)


def run_powershell(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["powershell", "-NoProfile", "-Command", script],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def query_process(pid: int) -> dict[str, Any] | None:
    script = f"""
$process = Get-CimInstance Win32_Process -Filter "ProcessId={pid}"
if ($process) {{
    $process |
        Select-Object ProcessId, ParentProcessId, Name, CommandLine, ExecutablePath |
        ConvertTo-Json -Compress
}}
"""
    completed = run_powershell(script)
    output = completed.stdout.strip()
    if not output:
        return None
    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def process_text(process: dict[str, Any] | None, key: str) -> str:
    if not process:
        return ""
    value = process.get(key)
    return value if isinstance(value, str) else ""


def process_name(process: dict[str, Any] | None) -> str:
    return process_text(process, "Name").lower()


def process_command(process: dict[str, Any] | None) -> str:
    return normalize(process_text(process, "CommandLine"))


def process_executable(process: dict[str, Any] | None) -> str:
    return normalize(process_text(process, "ExecutablePath"))


def is_under_root(process: dict[str, Any] | None) -> bool:
    normalized_root = normalize(str(ROOT))
    command = process_command(process)
    executable = process_executable(process)
    return command.find(normalized_root) >= 0 or executable.startswith(normalized_root + "\\")


def is_launcher_process(process: dict[str, Any] | None) -> bool:
    if not process:
        return False
    command = process_command(process)
    executable = process_executable(process)
    packaged_python = normalize(str(ROOT / "python" / "python.exe"))
    is_relevant_command = "launcher.py" in command or "uvicorn" in command or "app.main:app" in command
    return is_under_root(process) and (
        is_relevant_command
        or (executable == packaged_python and "stop.py" not in command)
    )


def is_start_window(process: dict[str, Any] | None) -> bool:
    if not process or process_name(process) != "cmd.exe":
        return False
    command = process_command(process)
    normalized_root = normalize(str(ROOT))
    return normalized_root in command and ("启动.bat" in command or "start.bat" in command)


def remove_pid_file() -> None:
    try:
        if PID_FILE.exists():
            PID_FILE.unlink()
    except OSError:
        pass


def find_package_processes() -> list[int]:
    root = normalize(str(ROOT))
    packaged_python = normalize(str(ROOT / "python" / "python.exe"))
    script = f"""
$root = @'
{root}
'@
$python = @'
{packaged_python}
'@
$current = {os.getpid()}
Get-CimInstance Win32_Process | ForEach-Object {{
    if ($_.ProcessId -eq $current) {{
        return
    }}
    $cmd = if ($_.CommandLine) {{ $_.CommandLine.ToLower().Replace('/', '\\') }} else {{ '' }}
    $exe = if ($_.ExecutablePath) {{ $_.ExecutablePath.ToLower().Replace('/', '\\') }} else {{ '' }}
    $name = if ($_.Name) {{ $_.Name.ToLower() }} else {{ '' }}
    $underRoot = $cmd.Contains($root) -or $exe.StartsWith($root + '\\')
    $isLauncher = $cmd.Contains('launcher.py') -or $cmd.Contains('uvicorn') -or $cmd.Contains('app.main:app')
    $isPackagedPython = $exe -eq $python -and -not $cmd.Contains('stop.py')
    $isStartWindow = $name -eq 'cmd.exe' -and $cmd.Contains($root) -and ($cmd.Contains('启动.bat') -or $cmd.Contains('start.bat'))
    if (($underRoot -and ($isLauncher -or $isPackagedPython)) -or $isStartWindow) {{
        [Console]::WriteLine($_.ProcessId)
    }}
}}
"""
    completed = run_powershell(script)
    pids: list[int] = []
    for line in completed.stdout.splitlines():
        value = line.strip()
        if value.isdigit():
            pid = int(value)
            if pid not in pids:
                pids.append(pid)
    return pids


def append_pid(pids: list[int], pid: int | None) -> None:
    if pid is not None and pid != os.getpid() and pid not in pids:
        pids.append(pid)


def collect_processes_to_stop() -> list[int]:
    values = read_pid_values()
    launcher_pid = read_int(values, "pid")
    parent_pid = read_int(values, "parent_pid")
    pids: list[int] = []

    parent_process = query_process(parent_pid) if parent_pid is not None else None
    if is_start_window(parent_process):
        append_pid(pids, parent_pid)

    launcher_process = query_process(launcher_pid) if launcher_pid is not None else None
    if is_launcher_process(launcher_process):
        append_pid(pids, launcher_pid)
    elif launcher_pid is not None:
        print("The PID file is stale. Checking running package processes.")

    for fallback_pid in find_package_processes():
        append_pid(pids, fallback_pid)
    return pids


def process_already_gone(pid: int, output: str) -> bool:
    if query_process(pid) is None:
        return True
    normalized_output = output.lower()
    return "not found" in normalized_output or "not running" in normalized_output


def stop_processes(pids: list[int]) -> int:
    if not pids:
        print("No running service was found.")
        remove_pid_file()
        return 0

    exit_code = 0
    stopped = 0
    already_stopped = 0
    for pid in pids:
        completed = subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        output = (completed.stdout or "") + (completed.stderr or "")
        if completed.returncode == 0:
            print(f"Stopped process {pid}.")
            stopped += 1
            continue
        if process_already_gone(pid, output):
            print(f"Process {pid} was already stopped.")
            already_stopped += 1
            continue
        exit_code = completed.returncode
        print(f"Stop failed for process {pid}:")
        print(output)

    remove_pid_file()
    if exit_code == 0 and stopped == 0 and already_stopped > 0:
        print("Service is already stopped.")
    return exit_code


def main() -> int:
    return stop_processes(collect_processes_to_stop())


if __name__ == "__main__":
    raise SystemExit(main())
