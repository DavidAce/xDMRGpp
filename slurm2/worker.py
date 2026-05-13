from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import os
import signal
import socket
import subprocess
from dataclasses import dataclass
from pathlib import Path

from slurm2.common import (
    append_jsonl,
    load_json,
    mkdir,
    now_utc,
    rclone_copyto,
    rclone_path,
    read_last_jsonl,
    resolve_named_experiment_dir,
)
from slurm2.models import PointSpec, PointStatus


def gpu_policy_value(value: str) -> str:
    policy = value.upper()
    if policy not in {"ON", "OFF", "TRY"}:
        raise argparse.ArgumentTypeError("expected one of ON, OFF, or TRY")
    return policy


def gpu_id_value(value: str) -> str:
    if value.lower() == "auto":
        return "auto"
    try:
        gpu_id = int(value)
    except ValueError as err:
        raise argparse.ArgumentTypeError("expected 'auto', -1, or a non-negative integer") from err
    if gpu_id < -1:
        raise argparse.ArgumentTypeError("expected 'auto', -1, or a non-negative integer")
    return str(gpu_id)


def fraction_0_to_1(value: str) -> float:
    try:
        fraction = float(value)
    except ValueError as err:
        raise argparse.ArgumentTypeError("expected a floating-point value in [0, 1]") from err
    if fraction < 0.0 or fraction > 1.0:
        raise argparse.ArgumentTypeError("expected a floating-point value in [0, 1]")
    return fraction


def non_negative_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as err:
        raise argparse.ArgumentTypeError("expected a non-negative integer") from err
    if number < 0:
        raise argparse.ArgumentTypeError("expected a non-negative integer")
    return number


@dataclass
class RuntimeContext:
    args: argparse.Namespace
    point: PointSpec
    status: PointStatus | None
    experiment_dir: Path
    remote_experiment: str | None


# Define the CLI for one array-task worker process.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one slurm2 chunk inside a Slurm array task. The worker checks the latest scanned status, "
            "refreshes remote per-seed runtime files when rclone is enabled, runs the requested seeds, "
            "and records JSONL events plus log and HDF5 output paths."
        )
    )
    parser.add_argument("--name", dest="name", type=str, default=None, help="Experiment directory name under --base-dir. This lets you target 'xdmrg6-gdplusk' without typing the full path.")
    parser.add_argument("--base-dir", "--basedir", dest="base_dir", type=str, default=None, help="Base directory prepended to --name. Defaults to /mnt/WDB-AN1500/mbl_transition when that path exists, otherwise the current working directory.")
    parser.add_argument("--experiment-dir", dest="experiment_dir", type=str, default=None, help="Explicit experiment directory path. This overrides --name and --base-dir.")
    parser.add_argument("--point-id", type=str, required=True, help="Exact point id to run, for example 'L20_g0.500_d+5.00'.")
    parser.add_argument("--config", type=str, required=True, help="Absolute or experiment-relative cfg file passed through to xDMRG++.")
    parser.add_argument("--exec", dest="executable", type=str, required=True, help="Executable path for xDMRG++.")
    parser.add_argument("--remote-experiment", dest="remote_experiment", type=str, default=None, help="Optional rclone remote for the experiment. When set, the worker refreshes remote runtime files before each seed and uploads outputs after each state change.")
    parser.add_argument("--seed-offset", type=int, required=True, help="First absolute seed in the chunk before Slurm array indexing is applied.")
    parser.add_argument("--seed-count", type=int, required=True, help="Total number of seeds owned by this chunk.")
    parser.add_argument("--force-run", action="store_true", help="MISSING and TIMEOUT seeds are already runnable. This flag additionally reopens FAILED seeds and ignores remote RUNNING locks for the scheduled seeds. Seeds already marked FINISHED or SKIP in the scanned status are still skipped.")
    parser.add_argument("--replace", action="store_true", help="Pass --replace to xDMRG++ instead of relying only on the cfg file's collision policy.")
    parser.add_argument("--rclone-remove", action="store_true", help="After a successful upload, delete the local .h5 and .txt runtime files. Event JSONL files are always kept locally.")
    parser.add_argument("--gpu-policy", type=gpu_policy_value, default=None, help="Pass --gpu-policy to xDMRG++. Leave unset to use the generated cfg value.")
    parser.add_argument("--gpu-id", type=gpu_id_value, default=None, help="Pass --gpu-id to xDMRG++. Accepts auto, -1, or a non-negative device id.")
    parser.add_argument("--gpu-switchsize", type=non_negative_int, default=None, help="Pass --gpu-switchsize to xDMRG++.")
    parser.add_argument("--gpu-max-alloc-fraction", type=fraction_0_to_1, default=None, help="Pass --gpu-max-alloc-fraction to xDMRG++.")
    return parser


# Load one point plus its last scanned status before running any seeds.
def load_runtime(args: argparse.Namespace) -> RuntimeContext:
    experiment_dir = resolve_named_experiment_dir(experiment_dir=args.experiment_dir, experiment_name=args.name, base_dir=args.base_dir)
    point_payload = None
    for point_path in (experiment_dir / "points").glob("*.json"):
        payload = load_json(point_path)
        if payload["point_id"] == args.point_id:
            point_payload = payload
            break
    if point_payload is None:
        raise FileNotFoundError(f"Could not find point {args.point_id} in {experiment_dir / 'points'}")
    point = PointSpec.from_dict(point_payload)
    status_payload = load_json(experiment_dir / point.status_relpath)
    status = PointStatus.from_dict(status_payload) if status_payload else None
    return RuntimeContext(args=args, point=point, status=status, experiment_dir=experiment_dir, remote_experiment=args.remote_experiment)


# Build the per-seed event log path inside the experiment directory.
def event_path(ctx: RuntimeContext, seed: int) -> Path:
    return ctx.experiment_dir / ctx.point.event_dir_relpath / f"{seed}.jsonl"


# Build the per-seed text log path inside the experiment directory.
def log_path(ctx: RuntimeContext, seed: int) -> Path:
    return ctx.experiment_dir / ctx.point.log_dir_relpath / f"{seed}.txt"


# Build the per-seed HDF5 output path inside the experiment directory.
def output_path(ctx: RuntimeContext, seed: int) -> Path:
    return ctx.experiment_dir / ctx.point.output_dir_relpath / f"{ctx.point.output_stem}_{seed}.h5"


# Read the scanned state for one seed from the latest status JSON.
def seed_state_from_status(ctx: RuntimeContext, seed: int) -> str:
    if ctx.status is None:
        return "MISSING"
    for seed_status in ctx.status.seeds:
        if seed_status.seed == seed:
            return seed_status.state
    return "MISSING"


# Append one runtime state transition to the seed's JSONL event log.
def append_event(ctx: RuntimeContext, seed: int, state: str, *, detail: str | None = None) -> None:
    payload = {
        "timestamp": now_utc(),
        "state": state,
        "cluster": os.environ.get("SLURM_CLUSTER_NAME"),
        "hostname": socket.gethostname(),
        "seed": seed,
        "slurm_job_id": os.environ.get("SLURM_JOBID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }
    if detail:
        payload["detail"] = detail
    append_jsonl(event_path(ctx, seed), payload)


# Push one local runtime file to the remote experiment, typically to neumann.
def sync_file_to_remote(ctx: RuntimeContext, local_path: Path) -> bool:
    if ctx.remote_experiment is None or not local_path.exists():
        return False
    relative = local_path.relative_to(ctx.experiment_dir)
    target = rclone_path(ctx.remote_experiment, relative)
    success = rclone_copyto(str(local_path), target)
    is_runtime_blob = local_path.suffix in {".h5", ".txt"}
    if success and ctx.args.rclone_remove and is_runtime_blob and local_path.is_file():
        local_path.unlink(missing_ok=True)
    return success


# Pull one runtime file from the remote experiment when remote sync is enabled.
def sync_file_from_remote(ctx: RuntimeContext, relative_path: Path) -> bool:
    if ctx.remote_experiment is None:
        return False
    local_path = ctx.experiment_dir / relative_path
    mkdir(local_path.parent)
    source = rclone_path(ctx.remote_experiment, relative_path)
    return rclone_copyto(source, str(local_path))


# Read the latest event after refreshing it from the remote if needed.
def latest_event(ctx: RuntimeContext, seed: int) -> dict | None:
    path = event_path(ctx, seed)
    if ctx.remote_experiment is not None:
        sync_file_from_remote(ctx, path.relative_to(ctx.experiment_dir))
    return read_last_jsonl(path)


# Refresh this seed's runtime inputs before deciding whether to run it.
def sync_runtime_inputs(ctx: RuntimeContext, seed: int) -> None:
    if ctx.remote_experiment is None:
        return
    for path in [
        event_path(ctx, seed).relative_to(ctx.experiment_dir),
        log_path(ctx, seed).relative_to(ctx.experiment_dir),
        output_path(ctx, seed).relative_to(ctx.experiment_dir),
    ]:
        sync_file_from_remote(ctx, path)


# Push the event, log, and HDF5 files for one seed after state changes.
def sync_runtime_outputs(ctx: RuntimeContext, seed: int) -> None:
    for path in [event_path(ctx, seed), log_path(ctx, seed), output_path(ctx, seed)]:
        sync_file_to_remote(ctx, path)


# Read Slurm's array-task step so one task can cover several seeds.
def current_task_step() -> int:
    raw = os.environ.get("SLURM_ARRAY_TASK_STEP")
    return int(raw) if raw else 1


# Read the current Slurm array-task index for seed mapping.
def current_task_id() -> int:
    raw = os.environ.get("SLURM_ARRAY_TASK_ID")
    return int(raw) if raw else 0


# Map the current array task onto its owned seed range.
def seed_sequence(args: argparse.Namespace) -> range:
    start = args.seed_offset + current_task_id()
    stop = min(args.seed_offset + args.seed_count, start + current_task_step())
    return range(start, stop)


# Decide whether this seed can be skipped because work already exists.
def should_skip_seed(ctx: RuntimeContext, seed: int) -> bool:
    status_state = seed_state_from_status(ctx, seed)
    if status_state in {"FINISHED", "SKIP"}:
        sync_runtime_outputs(ctx, seed)  # Opportunistically flush local files upstream.
        return True
    if status_state == "FAILED" and not ctx.args.force_run:
        sync_runtime_outputs(ctx, seed)
        return True

    latest = latest_event(ctx, seed)  # Refresh remote state before making a collision decision.
    latest_state = latest.get("state") if latest else None
    if latest_state in {"FINISHED", "SKIP"} and not ctx.args.force_run:
        sync_runtime_outputs(ctx, seed)
        return True
    if latest_state == "RUNNING" and not ctx.args.force_run:
        cluster = latest.get("cluster")
        if cluster == os.environ.get("SLURM_CLUSTER_NAME"):
            # Check sacct only for same-cluster collisions.
            array_job_id = latest.get("slurm_array_job_id")
            array_task_id = latest.get("slurm_array_task_id")
            if array_job_id and array_task_id:
                result = subprocess.run(
                    ["sacct", "-X", "--jobs", f"{array_job_id}_{array_task_id}", "--format=state", "--parsable2", "--noheader"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                state_text = result.stdout.strip()
                if result.returncode == 0 and "RUNNING" in state_text:
                    return True
        elif cluster and status_state not in {"FAILED", "TIMEOUT"}:  # Trust other clusters unless the seed needs retrying.
            return True
    return False


# Build the SIGINT/SIGTERM handler so interrupted seeds get TIMEOUT events.
def make_signal_handler(ctx: RuntimeContext, active_seed: dict[str, int | None]):
    # Close over the active seed so signals can mark the right one.
    def _handler(_signum, _frame):
        seed = active_seed.get("seed")
        if seed is not None:
            append_event(ctx, seed, "TIMEOUT")
            sync_runtime_outputs(ctx, seed)
        raise SystemExit(1)

    return _handler


# Run one seed end-to-end: sync, execute xDMRG++, log, record, and upload.
def run_seed(ctx: RuntimeContext, seed: int, active_seed: dict[str, int | None]) -> int:
    if should_skip_seed(ctx, seed):
        return 0

    sync_runtime_inputs(ctx, seed)  # Pull remote files just before launch.
    log_file = log_path(ctx, seed)
    out_file = output_path(ctx, seed)
    mkdir(log_file.parent)
    mkdir(out_file.parent)
    append_event(ctx, seed, "RUNNING")
    sync_file_to_remote(ctx, event_path(ctx, seed))  # Publish the RUNNING lease early.

    cmd = [
        ctx.args.executable,
        f"--config={ctx.args.config}",
        f"--outfile={out_file}",
        f"--seed={seed}",
        f"--threads={os.environ.get('SLURM_CPUS_PER_TASK', '1')}",
    ]
    if ctx.args.replace:
        cmd.append("--replace")
    if ctx.args.gpu_policy is not None:
        cmd.append(f"--gpu-policy={ctx.args.gpu_policy}")
    if ctx.args.gpu_id is not None:
        cmd.append(f"--gpu-id={ctx.args.gpu_id}")
    if ctx.args.gpu_switchsize is not None:
        cmd.append(f"--gpu-switchsize={ctx.args.gpu_switchsize}")
    if ctx.args.gpu_max_alloc_fraction is not None:
        cmd.append(f"--gpu-max-alloc-fraction={ctx.args.gpu_max_alloc_fraction}")

    active_seed["seed"] = seed
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write(f"{now_utc()} EXEC {' '.join(cmd)}\n")
        handle.flush()
        process = subprocess.run(cmd, stdout=handle, stderr=handle, text=True, check=False)
    active_seed["seed"] = None

    if process.returncode == 0:
        append_event(ctx, seed, "FINISHED")
        sync_runtime_outputs(ctx, seed)
        return 0

    append_event(ctx, seed, "FAILED", detail=f"exit_code={process.returncode}")
    sync_runtime_outputs(ctx, seed)
    return process.returncode


# Entry point for one Slurm array task running its owned seed subset.
def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.experiment_dir and not args.name:
        parser.error("one of --name or --experiment-dir is required")
    ctx = load_runtime(args)
    seeds = list(seed_sequence(args))
    active_seed: dict[str, int | None] = {"seed": None}
    handler = make_signal_handler(ctx, active_seed)
    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)

    exit_code = 0
    for seed in seeds:
        exit_code = max(exit_code, run_seed(ctx, seed, active_seed))  # Keep the highest failure code.
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
