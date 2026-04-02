from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SeedRange:
    offset: int
    extent: int

    # Serialize one requested seed span into JSON-friendly form.
    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class PointToken:
    name: str
    raw_value: str
    full_token: str
    settings_path: str

    # Serialize one parsed point-key token for point metadata.
    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass
class PointSpec:
    point_id: str
    config_stem: str
    point_key: str
    source_experiment: str
    source_group: str
    project_name: str
    model_type: str
    output_stem: str
    config_relpath: str
    output_dir_relpath: str
    status_relpath: str
    log_dir_relpath: str
    event_dir_relpath: str
    config_overrides: dict[str, str]
    settings_values: dict[str, str]
    token_values: dict[str, str]
    tokens: list[PointToken]
    seed_ranges: list[SeedRange]
    scan_extent: int | None
    requested_seed_count: int
    tags: list[str] = field(default_factory=list)
    last_scan_at: str | None = None
    point_state: str = "PENDING"
    range_states: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    # Serialize a point exactly as generate/scan/write expect it on disk.
    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["seed_ranges"] = [seed_range.to_dict() for seed_range in self.seed_ranges]
        data["tokens"] = [token.to_dict() for token in self.tokens]
        return data

    @classmethod
    # Rebuild a point object from the stored JSON metadata.
    def from_dict(cls, payload: dict[str, Any]) -> "PointSpec":
        payload = dict(payload)
        payload["seed_ranges"] = [SeedRange(**item) for item in payload["seed_ranges"]]
        payload["tokens"] = [PointToken(**item) for item in payload.get("tokens", [])]
        return cls(**payload)


@dataclass
class ExperimentMetadata:
    experiment_name: str
    project_name: str
    family: str
    output_stem: str
    template_relpath: str
    spec_name: str
    spec_relpath: str
    point_count: int
    generated_at: str
    updated_at: str
    git_commit: str | None = None

    # Serialize top-level experiment metadata for experiment.json.
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    # Rebuild experiment metadata from experiment.json.
    def from_dict(cls, payload: dict[str, Any]) -> "ExperimentMetadata":
        return cls(**payload)


@dataclass
class ChunkPlan:
    point: PointSpec
    offset: int
    extent: int
    step: int
    states: list[str]


@dataclass
class SeedStatus:
    seed: int
    state: str
    included: bool
    output_relpath: str
    detail: str | None = None

    # Serialize one seed status entry inside a point status file.
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PointStatus:
    point_id: str
    config_stem: str
    updated_at: str
    scan_extent: int | None
    requested_seed_count: int
    point_state: str
    range_states: list[str]
    summary: dict[str, Any]
    seeds: list[SeedStatus]

    # Serialize the scanned status for one point.
    def to_dict(self) -> dict[str, Any]:
        return {
            "point_id": self.point_id,
            "config_stem": self.config_stem,
            "updated_at": self.updated_at,
            "scan_extent": self.scan_extent,
            "requested_seed_count": self.requested_seed_count,
            "point_state": self.point_state,
            "range_states": list(self.range_states),
            "summary": dict(self.summary),
            "seeds": [seed.to_dict() for seed in self.seeds],
        }

    @classmethod
    # Rebuild a scanned point status from status/<point>.json.
    def from_dict(cls, payload: dict[str, Any]) -> "PointStatus":
        return cls(
            point_id=payload["point_id"],
            config_stem=payload["config_stem"],
            updated_at=payload["updated_at"],
            scan_extent=payload.get("scan_extent"),
            requested_seed_count=payload["requested_seed_count"],
            point_state=payload["point_state"],
            range_states=list(payload.get("range_states", [])),
            summary=dict(payload.get("summary", {})),
            seeds=[SeedStatus(**seed) for seed in payload.get("seeds", [])],
        )


# Convenience helper used by callers that need the cfg path for a point.
def point_file(experiment_dir: Path, point: PointSpec) -> Path:
    return experiment_dir / point.config_relpath
