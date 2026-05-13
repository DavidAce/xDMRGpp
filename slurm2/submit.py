from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import errno
import json
import os
import socket
import sys
from datetime import datetime
from pathlib import Path

from slurm2.common import (
    REPO_ROOT,
    atomic_write_json,
    ensure_experiment_dirs,
    host_name,
    load_json,
    now_utc,
    point_match,
    read_git_commit,
    render_command,
    resolve_executable,
    resolve_remote_experiment,
    resolve_named_experiment_dir,
    run_command,
    run_optional_command,
    split_seed_range,
    sync_experiment_metadata,
    sync_experiment_runtime,
)
from slurm2.defaults import DEFAULT_REMOTE_ROOT
from slurm2.models import ChunkPlan, PointSpec, PointStatus


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


SUBMIT_EXAMPLES = """examples:
  Full CPU submission with GPU TRY policy from the cfg/CLI layer:
    python -m slurm2 submit \\
      --name xdmrg6-gdplusk \\
      --rclone-prefix xdmrg6-gdplusk \\
      --pattern L12 \\
      --job-name xdmrg6-L12 \\
      --mem-per-cpu 2500M \\
      --time 4:00:00 \\
      --sims-per-array 1000 \\
      --cpus-per-task 2 \\
      --omp-num-threads 2 \\
      --threads-per-core 1 \\
      --ntasks 1 \\
      --ntasks-per-core 1 \\
      --requeue \\
      --sims-per-task 50 \\
      --build-type Release \\
      --rclone-remove \\
      --gpu-policy TRY

  Same shape, but requesting one GPU per worker task from Slurm:
    python -m slurm2 submit \\
      --name xdmrg6-gdplusk \\
      --rclone-prefix xdmrg6-gdplusk \\
      --pattern L12 \\
      --job-name xdmrg6-L12 \\
      --mem-per-cpu 2500M \\
      --time 4:00:00 \\
      --sims-per-array 1000 \\
      --cpus-per-task 2 \\
      --omp-num-threads 2 \\
      --threads-per-core 1 \\
      --ntasks 1 \\
      --ntasks-per-core 1 \\
      --requeue \\
      --sims-per-task 50 \\
      --build-type Release \\
      --rclone-remove \\
      --gpus-per-task 1 \\
      --gpu-policy TRY \\
      --gpu-id auto

  Some clusters allocate GPUs through generic resources instead:
    replace --gpus-per-task 1 with --gres gpu:1

notes:
  --gpu-policy TRY lets xDMRG++ use a usable GPU when one is visible, but it
  does not request a GPU allocation from Slurm by itself. Use --gpus-per-task,
  --gpus, --gpus-per-node, or --gres when the cluster requires an explicit GPU
  request.
"""


# Load the last scanned status for one point before planning chunks.
def load_point_status(experiment_dir: Path, point: PointSpec) -> PointStatus | None:
    payload = load_json(experiment_dir / point.status_relpath)
    return PointStatus.from_dict(payload) if payload else None


# Read the latest per-seed event state, if a worker has written one.
def load_latest_event_state(event_path: Path) -> str | None:
    if not event_path.exists():
        return None
    with event_path.open("r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]
    if not lines:
        return None
    return json.loads(lines[-1]).get("state")


# Combine scanned status with per-seed event logs into one effective state.
def get_seed_state(point: PointSpec, status: PointStatus | None, experiment_dir: Path, seed: int) -> str:
    if status is not None:
        for seed_status in status.seeds:
            if seed_status.seed == seed:
                state = seed_status.state
                break
        else:
            state = "MISSING"
    else:
        state = "MISSING"

    event_state = load_latest_event_state(experiment_dir / point.event_dir_relpath / f"{seed}.jsonl")
    if event_state == "FINISHED":
        return "FINISHED"
    return state


# Define the CLI for chunk planning and Slurm submission.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plan and submit Slurm job arrays for a generated slurm2 experiment. "
            "submit first refreshes local metadata from the remote experiment when rclone is enabled, "
            "then computes which seed chunks are still worth running, and finally emits one sbatch array per chunk."
        ),
        epilog=SUBMIT_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--name", dest="name", type=str, default=None, help="Experiment directory name under --base-dir. This lets you target 'xdmrg6-gdplusk' without typing the full path.")
    parser.add_argument("--base-dir", "--basedir", dest="base_dir", type=str, default=None, help="Base directory prepended to --name. Defaults to /mnt/WDB-AN1500/mbl_transition when that path exists, otherwise the current working directory.")
    parser.add_argument("--experiment-dir", dest="experiment_dir", type=str, default=None, help="Explicit experiment directory path. This overrides --name and --base-dir.")
    parser.add_argument("--remote-experiment", dest="remote_experiment", type=str, default=None, help="Explicit rclone remote path for the experiment, for example 'neumann:/mnt/WDB-AN1500/mbl_transition/xdmrg6-gdplusk'. This overrides --remote-root and --rclone-prefix.")
    parser.add_argument("--remote-root", type=str, default=DEFAULT_REMOTE_ROOT, help="Base rclone remote used together with --rclone-prefix when --remote-experiment is not given.")
    parser.add_argument("--rclone-prefix", type=str, default=None, help="Experiment path appended under --remote-root, usually the experiment name such as 'xdmrg6-gdplusk'.")
    parser.add_argument("--build-type", "-b", type=str, default="Release", help="Build directory under build/ used to locate xDMRG++ when --exec is not given.")
    parser.add_argument("--execname", type=str, default="xDMRG++", help="Executable name looked up inside build/<build-type>/ when --exec is not given.")
    parser.add_argument("--exec", dest="explicit_exec", type=str, default=None, help="Explicit executable path. Use this to bypass the build/<type>/ lookup logic entirely.")
    parser.add_argument("--clusters", "-M", type=str, default=None, help="Value passed through to sbatch --clusters, for example 'draken' or 'kraken'.")
    parser.add_argument("--nodelist", "-w", type=str, default=None, help="Restrict submission to a specific Slurm nodelist by passing --nodelist to sbatch.")
    parser.add_argument("--account", type=str, default=None, help="Slurm account name passed through unchanged to sbatch.")
    parser.add_argument("--reservation", type=str, default=None, help="Slurm reservation name passed through unchanged to sbatch.")
    parser.add_argument("--pattern", type=str, default=None, help="Only consider points whose point id, config stem, or tags contain this substring, for example 'L20'.")
    parser.add_argument("--omp-num-threads", type=int, default=None, help="Value exported as OMP_NUM_THREADS for the submitted jobs.")
    parser.add_argument("--omp-dynamic", action="store_true", default=None, help="Export OMP_DYNAMIC=true for the submitted jobs. Leave unset to keep the environment untouched.")
    parser.add_argument("--omp-max-active-levels", type=int, default=None, help="Value exported as OMP_MAX_ACTIVE_LEVELS for the submitted jobs.")
    parser.add_argument("--omp-places", type=str, choices=["threads", "cores", "sockets"], default=None, help="Value exported as OMP_PLACES for the submitted jobs.")
    parser.add_argument("--omp-proc-bind", type=str, choices=["true", "false", "close", "spread", "master"], default=None, help="Value exported as OMP_PROC_BIND for the submitted jobs.")
    parser.add_argument("--cpus-per-task", type=int, default=1, help="Number of CPU cores requested for each Slurm task. This usually matches the thread count used by xDMRG++.")
    parser.add_argument("--gpus", type=str, default=None, help="Value passed through to sbatch --gpus, for example '1' or 'v100:1'.")
    parser.add_argument("--gpus-per-node", type=str, default=None, help="Value passed through to sbatch --gpus-per-node.")
    parser.add_argument("--gpus-per-task", type=str, default=None, help="Value passed through to sbatch --gpus-per-task. This is usually the right allocation flag for one GPU per xDMRG++ worker task.")
    parser.add_argument("--gres", type=str, default=None, help="Value passed through to sbatch --gres, for clusters that allocate GPUs through generic resources, for example 'gpu:1'.")
    parser.add_argument("--nodes", type=int, default=1, help="Number of nodes requested for each sbatch submission. The normal xDMRG workflow uses one node per array job.")
    parser.add_argument("--ntasks", type=int, default=1, help="Number of Slurm tasks per submitted job.")
    parser.add_argument("--ntasks-per-core", type=int, default=1, help="Value passed through to sbatch --ntasks-per-core.")
    parser.add_argument("--ntasks-per-node", type=int, default=None, help="Optional value passed through to sbatch --ntasks-per-node.")
    parser.add_argument("--threads-per-core", type=int, default=1, help="Value passed through to sbatch --threads-per-core.")
    parser.add_argument("--openblas-coretype", type=str, default=None, help="Value exported as OPENBLAS_CORETYPE for the submitted jobs.")
    parser.add_argument("--dryrun", action="store_true", help="Print the generated sbatch and rclone commands without executing them.")
    parser.add_argument("--debug", action="store_true", help="Keep extra debug output enabled while planning submissions.")
    parser.add_argument("--exclusive", action="store_true", help="Request exclusive node allocation from Slurm by passing --exclusive to sbatch.")
    parser.add_argument("--hint", type=str, default=None, choices=["multithread", "nomultithread", "compute_bound", "memory_bound"], help="Slurm scheduling hint passed through to sbatch --hint.")
    parser.add_argument("--job-name", "-J", type=str, default="DMRG", help="Job name used for all sbatch submissions.")
    parser.add_argument("--mem-per-cpu", "-m", type=str, default="1G", help="Memory requested per CPU, for example 3000M or 4G.")
    parser.add_argument("--sims-per-array", "-n", type=int, default=1000, help="Maximum number of seeds grouped into one Slurm array submission for a given point or subrange.")
    parser.add_argument("--sims-per-task", type=int, default=10, help="Number of seeds handled sequentially by one array task. This becomes the Slurm array step size.")
    parser.add_argument("--other", "-o", type=str, default=None, help="Extra raw sbatch options split on spaces and appended verbatim to each generated sbatch command.")
    parser.add_argument("--open-mode", type=str, default="append", choices=["append", "truncate"], help="Value passed through to sbatch --open-mode for the Slurm log file.")
    parser.add_argument("--partition", "-p", type=str, default=None, help="Slurm partition passed through to sbatch.")
    parser.add_argument("--qos", "-q", type=str, default=None, help="Slurm QoS passed through to sbatch.")
    parser.add_argument("--requeue", action="store_true", help="Pass --requeue to sbatch so Slurm may requeue interrupted jobs according to cluster policy.")
    parser.add_argument("--time", "-t", type=str, default="0-01:00:00", help="Wall-clock limit passed through to sbatch --time, for example '4-00:00:00'.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Pass -v to sbatch for verbose submission output.")
    parser.add_argument("--default-kraken", action="store_true", help="Shortcut that fills in the current kraken defaults for partition and QoS unless you override them explicitly.")
    parser.add_argument("--default-tetralith", action="store_true", help="Shortcut that fills in the current tetralith partition unless you override it explicitly.")
    parser.add_argument("--rclone-remove", action="store_true", default=None, help="After a successful remote copy, delete the local .h5 and .txt runtime files. Event logs are kept. This gives safe copy-then-remove semantics rather than deleting on failure.")
    parser.add_argument("--minseed", type=int, default=None, help="Do not submit any seed below this absolute seed value.")
    parser.add_argument("--maxseed", type=int, default=None, help="Do not submit any seed greater than or equal to this absolute seed value.")
    parser.add_argument("--force-run", action="store_true", help="MISSING and TIMEOUT seeds are already runnable. This flag additionally reopens FAILED seeds and tells workers to ignore remote RUNNING locks for the explicitly scheduled seeds. Seeds already marked FINISHED or SKIP remain done.")
    parser.add_argument("--replace", action="store_true", help="Pass --replace to xDMRG++ for each scheduled seed instead of relying on the file-collision behavior encoded in the cfg.")
    parser.add_argument("--gpu-policy", type=gpu_policy_value, default=None, help="Pass --gpu-policy to xDMRG++ for each scheduled seed. Leave unset to use the generated cfg value.")
    parser.add_argument("--gpu-id", type=gpu_id_value, default=None, help="Pass --gpu-id to xDMRG++ for each scheduled seed. Accepts auto, -1, or a non-negative device id.")
    parser.add_argument("--gpu-switchsize", type=non_negative_int, default=None, help="Pass --gpu-switchsize to xDMRG++ for each scheduled seed.")
    parser.add_argument("--gpu-max-alloc-fraction", type=fraction_0_to_1, default=None, help="Pass --gpu-max-alloc-fraction to xDMRG++ for each scheduled seed.")
    parser.add_argument("--ignore-seed-order", action="store_true", help="Accepted for compatibility with the legacy CLI. slurm2 status lookups are keyed by seed, so seed order is not required.")
    return parser


# Fill in cluster-specific defaults when the shortcut flags are used.
def apply_cluster_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if args.default_kraken:
        args.partition = args.partition or "dedicated"
        args.qos = args.qos or "lowprio"
    if args.default_tetralith:
        args.partition = args.partition or "tetralith"
    return args


# Build the environment block exported to sbatch jobs.
def build_sbatch_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    if args.omp_num_threads:
        env["OMP_NUM_THREADS"] = str(args.omp_num_threads)
    if args.omp_dynamic is not None:
        env["OMP_DYNAMIC"] = str(args.omp_dynamic).lower()
    if args.omp_max_active_levels:
        env["OMP_MAX_ACTIVE_LEVELS"] = str(args.omp_max_active_levels)
    if args.omp_places:
        env["OMP_PLACES"] = args.omp_places
    if args.omp_proc_bind:
        env["OMP_PROC_BIND"] = args.omp_proc_bind
    if args.openblas_coretype:
        env["OPENBLAS_CORETYPE"] = args.openblas_coretype
    return env


# Assemble the sbatch arguments shared by every chunk submission.
def base_sbatch_args(args: argparse.Namespace, slurm_log_dir: Path) -> list[str]:
    sbatch_args = ["sbatch"]
    if args.clusters:
        sbatch_args.append(f"--clusters={args.clusters}")
    if args.nodelist:
        sbatch_args.append(f"--nodelist={args.nodelist}")
    if args.account:
        sbatch_args.append(f"--account={args.account}")
    if args.reservation:
        sbatch_args.append(f"--reservation={args.reservation}")
    if args.cpus_per_task:
        sbatch_args.append(f"--cpus-per-task={args.cpus_per_task}")
    if args.gpus:
        sbatch_args.append(f"--gpus={args.gpus}")
    if args.gpus_per_node:
        sbatch_args.append(f"--gpus-per-node={args.gpus_per_node}")
    if args.gpus_per_task:
        sbatch_args.append(f"--gpus-per-task={args.gpus_per_task}")
    if args.gres:
        sbatch_args.append(f"--gres={args.gres}")
    if args.nodes:
        sbatch_args.append(f"--nodes={args.nodes}")
    if args.ntasks_per_core:
        sbatch_args.append(f"--ntasks-per-core={args.ntasks_per_core}")
    if args.threads_per_core:
        sbatch_args.append(f"--threads-per-core={args.threads_per_core}")
    if args.ntasks_per_node:
        sbatch_args.append(f"--ntasks-per-node={args.ntasks_per_node}")
    if args.job_name:
        sbatch_args.append(f"--job-name={args.job_name}")
    if args.mem_per_cpu:
        sbatch_args.append(f"--mem-per-cpu={args.mem_per_cpu}")
    if args.ntasks:
        sbatch_args.append(f"--ntasks={args.ntasks}")
    if args.open_mode:
        sbatch_args.append(f"--open-mode={args.open_mode}")
    if args.partition:
        sbatch_args.append(f"--partition={args.partition}")
    if args.qos:
        sbatch_args.append(f"--qos={args.qos}")
    if args.requeue:
        sbatch_args.append("--requeue")
    if args.exclusive:
        sbatch_args.append("--exclusive")
    if args.hint:
        sbatch_args.append(f"--hint={args.hint}")
    if args.time:
        sbatch_args.append(f"--time={args.time}")
    if args.verbose:
        sbatch_args.append("-v")
    if args.other:
        sbatch_args.extend(args.other.split())
    sbatch_args.append(f"--output={slurm_log_dir / '%x-%A_%a.txt'}")
    sbatch_args.append(f"--chdir={REPO_ROOT}")
    return sbatch_args


# Load all points for the chosen experiment, then filter by substring.
def load_points(experiment_dir: Path, pattern: str | None) -> list[PointSpec]:
    points = []
    for point_file in sorted((experiment_dir / "points").glob("*.json")):
        point = PointSpec.from_dict(load_json(point_file))
        if point_match(pattern, point.point_id, point.config_stem, *point.tags):
            points.append(point)
    return points


# Decide whether a whole chunk can be skipped without submission.
def chunk_is_done(states: list[str], force_run: bool) -> bool:
    done_states = {"FINISHED", "SKIP"}
    if not force_run:
        done_states.add("FAILED")
    return all(state in done_states for state in states)


# Convert point seed ranges into the concrete chunks that need sbatch jobs.
def plan_chunks(experiment_dir: Path, points: list[PointSpec], args: argparse.Namespace) -> list[ChunkPlan]:
    plans: list[ChunkPlan] = []
    for point in points:
        status = load_point_status(experiment_dir, point)
        for seed_range in point.seed_ranges:
            extents, offsets = split_seed_range(seed_range.extent, seed_range.offset, args.sims_per_array)  # Break long ranges into manageable arrays.
            for extent, offset in zip(extents, offsets):
                step = min(extent, args.sims_per_task)
                states: list[str] = []
                for seed in range(offset, offset + extent):
                    if args.minseed is not None and seed < args.minseed:
                        continue
                    if args.maxseed is not None and seed >= args.maxseed:
                        continue
                    states.append(get_seed_state(point, status, experiment_dir, seed))
                if not states:
                    continue
                if chunk_is_done(states, args.force_run):  # Skip chunks already satisfied locally.
                    continue
                off_final = offset
                ext_final = extent
                if args.minseed is not None and off_final < args.minseed <= off_final + ext_final:
                    ext_final -= args.minseed - off_final
                    off_final = args.minseed
                if args.maxseed is not None and off_final < args.maxseed < off_final + ext_final:
                    ext_final = args.maxseed - off_final
                if ext_final <= 0:
                    continue
                plans.append(ChunkPlan(point=point, offset=off_final, extent=ext_final, step=min(step, ext_final),
                                       states=states))
    return plans


# Build the `python -m slurm2 worker ...` command passed into sbatch --wrap.
def build_worker_wrap(
        experiment_dir: Path,
        point: PointSpec,
        executable: Path,
        remote_experiment: str | None,
        args: argparse.Namespace,
        chunk: ChunkPlan,
) -> str:
    worker_cmd = [
        os.environ.get("PYTHON", "python"),
        "-m",
        "slurm2",
        "worker",
        "--experiment-dir",
        str(experiment_dir),
        "--point-id",
        point.point_id,
        "--config",
        str(experiment_dir / point.config_relpath),
        "--exec",
        str(executable),
        "--seed-offset",
        str(chunk.offset),
        "--seed-count",
        str(chunk.extent),
    ]
    if remote_experiment:
        worker_cmd.extend(["--remote-experiment", remote_experiment])
    if args.force_run:
        worker_cmd.append("--force-run")
    if args.replace:
        worker_cmd.append("--replace")
    if args.rclone_remove:
        worker_cmd.append("--rclone-remove")
    if args.gpu_policy is not None:
        worker_cmd.extend(["--gpu-policy", args.gpu_policy])
    if args.gpu_id is not None:
        worker_cmd.extend(["--gpu-id", args.gpu_id])
    if args.gpu_switchsize is not None:
        worker_cmd.extend(["--gpu-switchsize", str(args.gpu_switchsize)])
    if args.gpu_max_alloc_fraction is not None:
        worker_cmd.extend(["--gpu-max-alloc-fraction", str(args.gpu_max_alloc_fraction)])
    return render_command(worker_cmd)


# Submit every planned chunk and record the generated sbatch commands.
def submit_chunks(experiment_dir: Path, plans: list[ChunkPlan], args: argparse.Namespace) -> tuple[
    list[dict], dict[str, str]]:
    executable = resolve_executable(args.build_type, args.execname, args.explicit_exec)
    if not executable.exists():
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(executable))
    ldd_result = run_optional_command(["ldd", str(executable)])
    if ldd_result.returncode != 0 or "not found" in ldd_result.stdout:
        raise FileNotFoundError(errno.ENOENT, "Some dynamic libraries were not found", str(executable))

    remote_experiment = resolve_remote_experiment(args.remote_experiment, args.rclone_prefix, args.remote_root)  # Usually points at neumann from the cluster.
    layout = ensure_experiment_dirs(experiment_dir)
    env = build_sbatch_env(args)
    base_args = base_sbatch_args(args, layout["slurm_logs"])
    records: list[dict] = []

    for chunk in plans:
        wrap = build_worker_wrap(experiment_dir, chunk.point, executable, remote_experiment, args, chunk)  # One wrapped worker per chunk.
        sbatch_cmd = list(base_args)
        sbatch_cmd.append(f"--array=0-{chunk.extent - 1}:{chunk.step}")
        sbatch_cmd.extend(["--wrap", wrap])
        result = run_command(sbatch_cmd, env=env, dryrun=args.dryrun, capture_output=True)
        job_output = (result.stdout or "").strip() if result.stdout is not None else ""
        records.append(
            {
                "submitted_at": now_utc(),
                "point_id": chunk.point.point_id,
                "offset": chunk.offset,
                "extent": chunk.extent,
                "step": chunk.step,
                "command": sbatch_cmd,
                "command_text": render_command(sbatch_cmd),
                "stdout": job_output,
            }
        )
    return records, env


# Entry point for planning, syncing, and submitting experiment chunks.
def main(argv: list[str] | None = None) -> int:
    args = apply_cluster_defaults(build_parser().parse_args(argv))
    remote_experiment = resolve_remote_experiment(args.remote_experiment, args.rclone_prefix, args.remote_root)
    experiment_name = args.name or args.rclone_prefix or None
    experiment_dir = resolve_named_experiment_dir(experiment_dir=args.experiment_dir, experiment_name=experiment_name, base_dir=args.base_dir, fallback_name="experiment")
    layout = ensure_experiment_dirs(experiment_dir)

    if remote_experiment:
        sync_experiment_metadata(experiment_dir, remote_experiment, dryrun=args.dryrun)  # Pull latest metadata first.
        sync_experiment_runtime(experiment_dir, remote_experiment, dryrun=args.dryrun)  # Then push any local backlog.

    points = load_points(experiment_dir, args.pattern)
    if not points:
        raise FileNotFoundError(errno.ENOENT, f"{os.strerror(errno.ENOENT)}: no points found in {layout['points']}")

    plans = plan_chunks(experiment_dir, points, args)  # Convert point status into runnable array chunks.
    if not plans:
        print("No pending chunks to submit")
        return 0

    records, env = submit_chunks(experiment_dir, plans, args)
    report_stem = f"sbatch-{host_name()}-{datetime.now().strftime('%Y-%m-%dT%H.%M.%S')}"
    invocation = {
        "argv": list(sys.argv),
        "command": render_command([sys.executable, *sys.argv]),
        "cwd": str(Path.cwd()),
        "python": sys.executable,
    }
    report_path = layout["submissions"] / f"{report_stem}.json"
    atomic_write_json(
        report_path,
        {
            "created_at": now_utc(),
            "host": host_name(),
            "git_commit": read_git_commit(),
            "invocation": invocation,
            "arguments": vars(args),
            "experiment_dir": str(experiment_dir),
            "remote_experiment": remote_experiment,
            "environment": {key: env[key] for key in sorted(env) if
                            key.startswith("OMP_") or key.startswith("OPENBLAS_")},
            "records": records,
        },
    )
    text_report_path = layout["submissions"] / f"{report_stem}.txt"
    with text_report_path.open("w", encoding="utf-8") as handle:
        handle.write(f"# created_at: {now_utc()}\n")
        handle.write(f"# host: {host_name()}\n")
        handle.write(f"# cwd: {Path.cwd()}\n")
        handle.write(f"# python: {sys.executable}\n")
        handle.write(f"# invocation: {invocation['command']}\n")
        handle.write(f"# experiment_dir: {experiment_dir}\n")
        if remote_experiment:
            handle.write(f"# remote_experiment: {remote_experiment}\n")
        handle.write("\n")
        for record in records:
            handle.write(f"{record['command_text']}\n")
            if record["stdout"]:
                handle.write(f"# stdout: {record['stdout']}\n")
    print(f"Submitted {len(records)} job arrays")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
