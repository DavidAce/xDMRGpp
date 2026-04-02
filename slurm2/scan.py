from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import os
from multiprocessing import Pool
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from slurm2.common import atomic_write_json, experiment_layout, load_json, now_utc, point_match, resolve_named_experiment_dir
from slurm2.models import PointSpec, PointStatus, SeedStatus
from slurm2.xdmrg import get_h5_status, split_state_detail


# Load point metadata files and apply the optional substring filter.
def _load_points(experiment_dir: Path, pattern: str | None) -> list[tuple[Path, PointSpec]]:
    points: list[tuple[Path, PointSpec]] = []
    for point_path in sorted((experiment_dir / "points").glob("*.json")):
        point = PointSpec.from_dict(load_json(point_path))
        if point_match(pattern, point.point_id, point.config_stem, *point.tags):
            points.append((point_path, point))
    return points


# Scan one point's HDF5 outputs and rewrite both point and status JSON.
def _scan_point(job: tuple[str, str, dict]) -> dict:
    experiment_dir, point_path_str, point_payload = job
    experiment_root = Path(experiment_dir)
    point_path = Path(point_path_str)
    point = PointSpec.from_dict(point_payload)
    seeds: list[SeedStatus] = []
    missing_indices: list[int] = []

    for seed_range in point.seed_ranges:
        max_extent = max(seed_range.extent, point.scan_extent or seed_range.extent)  # Scan beyond the requested range when needed.
        for seed in range(seed_range.offset, seed_range.offset + max_extent):
            output_relpath = f"{point.output_dir_relpath}/{point.output_stem}_{seed}.h5"
            output_path = experiment_root / output_relpath
            raw_state = get_h5_status(output_path, point.model_type)
            state, detail = split_state_detail(raw_state)
            included = seed < seed_range.offset + seed_range.extent
            if not included and state != "FINISHED":
                state, detail = "SKIP", None
            record = SeedStatus(
                seed=seed,
                state=state,
                detail=detail,
                included=included,
                output_relpath=output_relpath,
            )
            if record.state == "MISSING":
                missing_indices.append(len(seeds))
            seeds.append(record)

    # Reuse spare finished seeds by converting the newest missing slots to SKIP.
    finished = sum(1 for seed in seeds if seed.state == "FINISHED")
    missing = sum(1 for seed in seeds if seed.state == "MISSING")
    needed_replacements = max(0, finished + missing - point.requested_seed_count)
    for index in reversed(missing_indices[-needed_replacements:]):
        seeds[index].state = "SKIP"

    range_states: list[str] = []
    for seed_range in point.seed_ranges:
        requested_states = [
            seed.state
            for seed in seeds
            if seed_range.offset <= seed.seed < seed_range.offset + seed_range.extent
        ]
        if all(state in {"FINISHED", "SKIP"} for state in requested_states):
            range_states.append("FINISHED")
        else:
            range_states.append("PENDING")

    summary = {
        "missing": sum(1 for seed in seeds if seed.included and seed.state == "MISSING"),
        "failed": sum(1 for seed in seeds if seed.included and seed.state == "FAILED"),
        "finished": sum(1 for seed in seeds if seed.included and seed.state == "FINISHED"),
        "skipped": sum(1 for seed in seeds if seed.included and seed.state == "SKIP"),
        "maxiter": sum(1 for seed in seeds if seed.included and seed.detail and "MAX_ITERS" in seed.detail),
        "saturated": sum(1 for seed in seeds if seed.included and seed.detail and "SATURATED" in seed.detail),
        "requested": point.requested_seed_count,
        "scanned": len(seeds),
    }
    point_state = "FINISHED" if all(state == "FINISHED" for state in range_states) else "PENDING"
    status = PointStatus(
        point_id=point.point_id,
        config_stem=point.config_stem,
        updated_at=now_utc(),
        scan_extent=point.scan_extent,
        requested_seed_count=point.requested_seed_count,
        point_state=point_state,
        range_states=range_states,
        summary=summary,
        seeds=seeds,
    )
    point.last_scan_at = status.updated_at
    point.point_state = point_state
    point.range_states = range_states
    point.summary = summary
    atomic_write_json(experiment_root / point.status_relpath, status.to_dict())
    atomic_write_json(point_path, point.to_dict())
    return {
        "point_id": point.point_id,
        "config_stem": point.config_stem,
        "status_relpath": point.status_relpath,
        "summary": summary,
    }


# Define the CLI for the HDF5 ground-truth status refresh step.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Scan experiment outputs and rebuild per-point status from the HDF5 files on disk. "
            "This is the ground-truth refresh step that updates status/*.json and each point's summary."
        )
    )
    parser.add_argument("--name", dest="name", type=str, default=None, help="Experiment directory name under --base-dir. This lets you target 'xdmrg6-gdplusk' without typing the full path.")
    parser.add_argument("--base-dir", "--basedir", dest="base_dir", type=str, default=None, help="Base directory prepended to --name. Defaults to /mnt/WDB-AN1500/mbl_transition when that path exists, otherwise the current working directory.")
    parser.add_argument("--experiment-dir", dest="experiment_dir", type=str, default=None, help="Explicit experiment directory path. This overrides --name and --base-dir.")
    parser.add_argument("--pattern", type=str, default=None, help="Only scan points whose point id, config stem, or tags contain this substring, for example 'L20'.")
    parser.add_argument("--jobs", type=int, default=max(1, os.cpu_count() or 1), help="Number of parallel scan workers. Values greater than one use a process pool when possible and fall back to threads if the local environment forbids multiprocessing.")
    return parser


# Run scan jobs in parallel, but fall back cleanly if processes are blocked.
def _run_scan_jobs(jobs: list[tuple[str, str, dict]], workers: int) -> list[dict]:
    if workers <= 1:
        return [_scan_point(job) for job in jobs]

    try:
        with Pool(processes=workers) as pool:  # Best throughput for HDF5-heavy scans.
            return pool.map(_scan_point, jobs)
    except (PermissionError, OSError):
        with ThreadPoolExecutor(max_workers=workers) as pool:  # Sandbox-safe fallback.
            return list(pool.map(_scan_point, jobs))


# Entry point for rescanning one experiment's outputs into status JSON files.
def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.experiment_dir and not args.name:
        parser.error("one of --name or --experiment-dir is required")
    experiment_dir = resolve_named_experiment_dir(experiment_dir=args.experiment_dir, experiment_name=args.name, base_dir=args.base_dir)
    layout = experiment_layout(experiment_dir)
    selected_points = _load_points(experiment_dir, args.pattern)
    if not selected_points:
        raise FileNotFoundError(f"No points found in {layout['points']}")

    print(f"Scanning {len(selected_points)} points in {experiment_dir}")
    jobs = [(str(experiment_dir), str(point_path), point.to_dict()) for point_path, point in selected_points]
    for _point_path, point in selected_points:
        print(f"Updating status: {point.status_relpath}")  # Match the legacy scan progress style.

    summaries = _run_scan_jobs(jobs, max(1, args.jobs))

    print(layout["status"])
    for summary_payload in summaries:
        summary = summary_payload["summary"]
        print(
            f"{summary_payload['config_stem']}: "
            f"missing={summary['missing']:5d} failed={summary['failed']:5d} "
            f"[maxiter={summary['maxiter']:5d} saturated={summary['saturated']:5d}] "
            f"finished={summary['finished']:5d} total={summary['requested']:5d}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
