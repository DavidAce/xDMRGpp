from __future__ import annotations

import json
import os
import shlex
import shutil
import socket
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from slurm2.defaults import (
    DEFAULT_LOCAL_EXPERIMENT_ROOT,
    DEFAULT_REMOTE_ROOT,
    PRIMARY_EXPERIMENT_METADATA_FILENAME,
    REPO_ROOT,
)


# Return a stable UTC timestamp string for metadata and event logs.
def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# Report the current hostname for submission and runtime records.
def host_name() -> str:
    return socket.gethostname()


# Create a directory tree if needed and hand the path back.
def mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# Write JSON atomically so scans and workers never see partial files.
def atomic_write_json(path: Path, payload: Any) -> None:
    mkdir(path.parent)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temp_path = Path(handle.name)
    temp_path.replace(path)


# Load JSON when present, otherwise return the supplied default.
def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# Append one JSON object per line for event-style logs.
def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    mkdir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True))
        handle.write("\n")


# Read the newest JSONL event without loading the whole file structure.
def read_last_jsonl(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]
    if not lines:
        return None
    return json.loads(lines[-1])


# Resolve an experiment directory using the default local experiment root.
def resolve_experiment_dir(experiment_dir: str | None, fallback_name: str) -> Path:
    if experiment_dir:
        return Path(experiment_dir).expanduser().resolve()
    if DEFAULT_LOCAL_EXPERIMENT_ROOT.exists():
        return (DEFAULT_LOCAL_EXPERIMENT_ROOT / fallback_name).resolve()
    return (Path.cwd() / fallback_name).resolve()


# Resolve by explicit path or by name under a configurable base directory.
def resolve_named_experiment_dir(*, experiment_dir: str | None = None, experiment_name: str | None = None, base_dir: str | None = None, fallback_name: str | None = None) -> Path:
    if experiment_dir:
        resolved = Path(experiment_dir).expanduser().resolve()
        expected_name = experiment_name or fallback_name
        if expected_name and resolved.name != expected_name:
            raise ValueError(f"Explicit experiment directory {resolved} does not match --name {expected_name}")
        return resolved
    name = experiment_name or fallback_name
    if not name:
        raise ValueError("Could not resolve experiment directory without --experiment-dir or --name")
    if base_dir:
        return (Path(base_dir).expanduser().resolve() / name).resolve()
    if DEFAULT_LOCAL_EXPERIMENT_ROOT.exists():
        return (DEFAULT_LOCAL_EXPERIMENT_ROOT / name).resolve()
    return (Path.cwd() / name).resolve()


# Centralize the on-disk experiment layout used by every command.
def experiment_layout(experiment_dir: Path) -> dict[str, Path]:
    return {
        "root": experiment_dir,
        "configs": experiment_dir / "configs",
        "points": experiment_dir / "points",
        "status": experiment_dir / "status",
        "events": experiment_dir / "events",
        "logs": experiment_dir / "logs",
        "output": experiment_dir / "output",
        "submissions": experiment_dir / "submissions",
        "slurm_logs": experiment_dir / "logs" / "slurm",
        "experiment_file": experiment_dir / PRIMARY_EXPERIMENT_METADATA_FILENAME,
    }


# Ensure every managed experiment subdirectory exists before writing.
def ensure_experiment_dirs(experiment_dir: Path) -> dict[str, Path]:
    layout = experiment_layout(experiment_dir)
    for key, path in layout.items():
        if key.endswith("_file") or key == "root":
            continue
        mkdir(path)
    mkdir(experiment_dir)
    return layout


# Split one seed span into submission-sized chunks for Slurm arrays.
def split_seed_range(extent: int, offset: int, chunk_size: int) -> tuple[list[int], list[int]]:
    extents = [min(extent, chunk_size)]
    offsets = [offset]
    while sum(extents) < extent:
        offsets.append(offsets[-1] + extents[-1])
        extents.append(min(chunk_size, extent - sum(extents)))
    return extents, offsets


# Quote a command for readable logs and dry-run output.
def render_command(cmd: Iterable[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


# Run a command, with dry-run support for planning and debugging.
def run_command(
    cmd: list[str],
    *,
    env: dict[str, str] | None = None,
    dryrun: bool = False,
    capture_output: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    if dryrun:
        print(render_command(cmd))
        return subprocess.CompletedProcess(cmd, 0, "", "")
    return subprocess.run(
        cmd,
        env=env,
        text=True,
        capture_output=capture_output,
        check=check,
    )


# Run a best-effort command and always return the subprocess result.
def run_optional_command(cmd: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


# Join a remote experiment root with a relative path in rclone syntax.
def rclone_path(remote_experiment: str, relative_path: str | Path) -> str:
    relative = str(relative_path).replace(os.sep, "/")
    if not relative:
        return remote_experiment
    return f"{remote_experiment.rstrip('/')}/{relative.lstrip('/')}"


# Discover whether rclone is available in the current environment.
def rclone_binary() -> str | None:
    return shutil.which("rclone")


# Copy a subtree with rclone, mainly for metadata and runtime directory sync.
def rclone_copy(
    source: str,
    target: str,
    *,
    includes: list[str] | None = None,
    dryrun: bool = False,
) -> bool:
    binary = rclone_binary()
    if binary is None:
        return False
    cmd = [binary, "copy", source, target, "--update", "--multi-thread-streams", "1"]
    if includes:
        cmd.append("--no-traverse")
        for pattern in includes:
            cmd.extend(["--include", pattern])
    result = run_optional_command(cmd) if not dryrun else subprocess.CompletedProcess(cmd, 0, "", "")
    if dryrun:
        print(render_command(cmd))
        return True
    return result.returncode == 0


# Copy one file with rclone; workers use this for per-seed transfers.
def rclone_copyto(
    source: str,
    target: str,
    *,
    dryrun: bool = False,
) -> bool:
    binary = rclone_binary()
    if binary is None:
        return False
    cmd = [binary, "copyto", source, target, "--update", "--multi-thread-streams", "1", "--no-traverse"]
    result = run_optional_command(cmd) if not dryrun else subprocess.CompletedProcess(cmd, 0, "", "")
    if dryrun:
        print(render_command(cmd))
        return True
    return result.returncode == 0


# Derive the remote experiment path from an explicit remote or a prefix.
def resolve_remote_experiment(remote_experiment: str | None, rclone_prefix: str | None, remote_root: str = DEFAULT_REMOTE_ROOT) -> str | None:
    if remote_experiment:
        return remote_experiment
    if rclone_prefix:
        return f"{remote_root.rstrip('/')}/{rclone_prefix.strip('/')}"
    return None


# Match a point against a simple substring filter used by scan and submit.
def point_match(pattern: str | None, *texts: str) -> bool:
    if pattern is None:
        return True
    return any(pattern in text for text in texts)


# Locate xDMRG++ from an explicit path or from build/<type>/.
def resolve_executable(build_type: str | None, execname: str, explicit_exec: str | None = None) -> Path:
    if explicit_exec:
        return Path(explicit_exec).expanduser().resolve()
    if not build_type or build_type == "None":
        located = shutil.which(execname)
        return Path(located).resolve() if located else Path(execname)
    return (REPO_ROOT / "build" / build_type / execname).resolve()


# Stamp outputs with the current git revision when available.
def read_git_commit() -> str | None:
    result = run_optional_command(["git", "rev-parse", "HEAD"], env=os.environ.copy())
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


# Pull experiment metadata from a remote, typically from neumann to a cluster.
def sync_experiment_metadata(experiment_dir: Path, remote_experiment: str, *, dryrun: bool = False) -> bool:
    includes = [
        PRIMARY_EXPERIMENT_METADATA_FILENAME,
        "configs/**",
        "points/**",
        "status/**",
        "events/**",
    ]
    mkdir(experiment_dir)
    return rclone_copy(remote_experiment, str(experiment_dir), includes=includes, dryrun=dryrun)


# Push runtime artifacts back to the remote, typically from a cluster to neumann.
def sync_experiment_runtime(experiment_dir: Path, remote_experiment: str, *, dryrun: bool = False) -> bool:
    includes = [
        "output/**",
        "logs/**",
        "events/**",
        "submissions/**",
    ]
    return rclone_copy(str(experiment_dir), remote_experiment, includes=includes, dryrun=dryrun)
