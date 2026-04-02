from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from slurm2.common import (
    atomic_write_json,
    ensure_experiment_dirs,
    load_json,
    now_utc,
    read_git_commit,
    resolve_named_experiment_dir,
)
from slurm2.models import ExperimentMetadata, PointSpec
from slurm2.specs import available_experiments, build_points_from_experiment


# Replace one tokenized cfg value while preserving the rest of the line.
def replace_value(line: str, pos: int, value: str) -> str:
    old_value = line.split()[pos]
    index_start = line.find(old_value)
    index_end = index_start + len(old_value)
    len_diff = len(old_value) - len(value)
    return line[:index_start] + value + " " * max(len_diff, 0) + line[index_end:]


# Render one cfg file from the shared template plus point-specific overrides.
def write_config_file(template_path: Path, output_path: Path, config_overrides: dict[str, str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with template_path.open("r", encoding="utf-8") as template, output_path.open("w", encoding="utf-8") as handle:
        for line in template:
            for key, value in config_overrides.items():
                if key in line:
                    line = replace_value(line, 2, value)
            handle.write(line)


# Write the cfg file and the point JSON for one experiment point.
def _write_point(experiment_dir: Path, template_path: Path, point: PointSpec) -> None:
    config_path = experiment_dir / point.config_relpath
    point_path = experiment_dir / "points" / f"{point.point_id}.json"
    existing = load_json(point_path)
    if existing is not None:
        existing_point = PointSpec.from_dict(existing)
        if existing_point.seed_ranges != point.seed_ranges:
            raise ValueError(
                f"Seed ranges changed for {point.point_id}: "
                f"expected {point.seed_ranges} found {existing_point.seed_ranges}"
            )
    write_config_file(template_path, config_path, point.config_overrides)
    atomic_write_json(point_path, point.to_dict())


# Define the CLI for experiment generation on neumann or elsewhere.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a slurm2 experiment from a declarative experiment spec. This writes experiment.json, "
            "one cfg file per point, and one point metadata JSON per point."
        )
    )
    parser.add_argument("--name", dest="name", required=True, choices=available_experiments(), help="Canonical experiment name. This selects slurm2/specs/<name>.toml and also the default subdirectory name under --base-dir.")
    parser.add_argument("--base-dir", "--basedir", dest="base_dir", type=str, default=None, help="Base directory prepended to --name. Defaults to /mnt/WDB-AN1500/mbl_transition when that path exists, otherwise the current working directory.")
    parser.add_argument("--experiment-dir", dest="experiment_dir", type=str, default=None, help="Explicit experiment directory path. This overrides --base-dir but must still end in the same final path component as --name.")
    parser.add_argument("--jobs", type=int, default=max(1, os.cpu_count() or 1), help="Number of parallel writer threads used while creating cfg files and point metadata.")
    return parser


# Generate the full experiment directory from the named TOML specification.
def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    experiment_spec, points = build_points_from_experiment(args.name)
    experiment_dir = resolve_named_experiment_dir(experiment_dir=args.experiment_dir, experiment_name=args.name, base_dir=args.base_dir, fallback_name=args.name)
    layout = ensure_experiment_dirs(experiment_dir)
    template_path = experiment_spec.template_path

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = [pool.submit(_write_point, experiment_dir, template_path, point) for point in points]  # One task per point.
        for future in futures:
            future.result()

    existing = load_json(layout["experiment_file"])  # Preserve the original generated_at when regenerating.
    now = now_utc()
    metadata = ExperimentMetadata(
        experiment_name=experiment_spec.name,
        project_name=experiment_spec.project_name,
        family=experiment_spec.family,
        output_stem=experiment_spec.output_stem,
        template_relpath=str(template_path.relative_to(experiment_dir.parent)) if template_path.is_relative_to(experiment_dir.parent) else str(template_path),
        spec_name=experiment_spec.name,
        spec_relpath=str(experiment_spec.spec_path.relative_to(experiment_dir.parent)) if experiment_spec.spec_path.is_relative_to(experiment_dir.parent) else str(experiment_spec.spec_path),
        point_count=len(list((experiment_dir / "points").glob("*.json"))),
        generated_at=(existing or {}).get("generated_at", now),
        updated_at=now,
        git_commit=read_git_commit(),
    )
    atomic_write_json(layout["experiment_file"], metadata.to_dict())
    print(f"Generated {len(points)} points in {experiment_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
