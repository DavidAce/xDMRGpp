from __future__ import annotations

import copy
from decimal import Decimal, InvalidOperation
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slurm2.defaults import REPO_ROOT
from slurm2.models import PointSpec, PointToken, SeedRange

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib


SPECS_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    project_name: str
    family: str
    template_path: Path
    output_stem: str
    point_id_template: str
    output_dir_template: str
    config_stem_template: str
    token_map: dict[str, str]
    defaults_settings: dict[str, Any]
    groups: dict[str, dict[str, Any]]
    spec_path: Path


@dataclass(frozen=True)
class TemplateValue:
    raw_value: str

    # Preserve raw token text unless the template requests explicit formatting.
    def __format__(self, format_spec: str) -> str:
        if not format_spec:
            return self.raw_value

        candidate = self.raw_value
        if candidate.lstrip("+-").isdigit():
            return format(int(candidate), format_spec)

        try:
            return format(Decimal(candidate), format_spec)
        except (InvalidOperation, ValueError):
            return format(candidate, format_spec)


# List the available experiment spec files in slurm2/specs/.
def available_experiments() -> list[str]:
    return sorted(path.stem for path in SPECS_DIR.glob("*.toml"))


# Resolve one experiment name to its TOML spec path.
def spec_path_for_experiment(experiment_name: str) -> Path:
    path = SPECS_DIR / f"{experiment_name}.toml"
    if not path.is_file():
        raise FileNotFoundError(f"Could not find experiment spec {path}")
    return path


# Recursively merge nested settings dictionaries from defaults and overrides.
def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


# Flatten nested TOML settings into dotted `settings.*` strings.
def _flatten_settings(settings: dict[str, Any], prefix: str = "settings") -> dict[str, str]:
    flattened: dict[str, str] = {}
    for key, value in settings.items():
        path = f"{prefix}.{key}"
        if isinstance(value, dict):
            flattened.update(_flatten_settings(value, prefix=path))
        else:
            flattened[path] = _stringify_value(value)
    return flattened


# Convert booleans and numerics into cfg-friendly strings.
def _stringify_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


# Map a dotted settings path onto the `section::key` cfg syntax.
def _cfg_key_from_settings_path(settings_path: str) -> str:
    if not settings_path.startswith("settings."):
        raise ValueError(f"Expected settings path to start with 'settings.': {settings_path}")
    return settings_path.removeprefix("settings.").replace(".", "::")


# Parse a point key like `d+5.00|g0.500` into structured tokens.
def _parse_point_key(point_key: str, token_map: dict[str, str]) -> list[PointToken]:
    aliases = sorted(token_map, key=len, reverse=True)
    tokens: list[PointToken] = []
    seen_names: set[str] = set()
    for segment in point_key.split("|"):
        match = next((alias for alias in aliases if segment.startswith(alias)), None)
        if match is None:
            raise ValueError(f"Could not parse token '{segment}' from point key '{point_key}'")
        raw_value = segment[len(match):]
        if not raw_value:
            raise ValueError(f"Missing value for token '{match}' in point key '{point_key}'")
        if match in seen_names:
            raise ValueError(f"Duplicate token '{match}' in point key '{point_key}'")
        seen_names.add(match)
        tokens.append(
            PointToken(
                name=match,
                raw_value=raw_value,
                full_token=segment,
                settings_path=token_map[match],
            )
        )
    return tokens


# Write one token value into a nested settings dictionary.
def _apply_settings_path(settings: dict[str, Any], path: str, value: str) -> None:
    if not path.startswith("settings."):
        raise ValueError(f"Token map path must begin with 'settings.': {path}")
    parts = path.split(".")[1:]
    current = settings
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


# Read one dotted settings value back out of a nested settings dictionary.
def _get_settings_value(settings: dict[str, Any], path: str) -> str | None:
    if not path.startswith("settings."):
        raise ValueError(f"Expected settings path to start with 'settings.': {path}")
    current: Any = settings
    for part in path.split(".")[1:]:
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return _stringify_value(current)


# Build the template context used for point ids, config stems, and paths.
def _template_context(
    experiment_spec: ExperimentSpec,
    token_values: dict[str, str],
    *,
    point_key: str,
    group_name: str,
) -> dict[str, Any]:
    context: dict[str, Any] = {
        "experiment_name": experiment_spec.name,
        "group_name": group_name,
        "output_stem": experiment_spec.output_stem,
        "point_key": point_key,
    }
    context.update({name: TemplateValue(raw_value=value) for name, value in token_values.items()})
    return context


# Render one template string after checking that every field is present.
def _render_template(template: str, context: dict[str, Any]) -> str:
    formatter = string.Formatter()
    fields = [field_name for _, field_name, _, _ in formatter.parse(template) if field_name]
    missing = [field for field in fields if field not in context]
    if missing:
        raise KeyError(f"Missing template values {missing} for template '{template}'")
    return template.format_map(context)


# Load and validate one experiment spec from its TOML file.
def load_experiment_spec(experiment_name: str) -> ExperimentSpec:
    spec_path = spec_path_for_experiment(experiment_name)
    with spec_path.open("rb") as handle:
        payload = tomllib.load(handle)

    experiment = payload["experiment"]
    if experiment["name"] != experiment_name:
        raise ValueError(f"Experiment spec file {spec_path} must declare name = {experiment_name!r}, found {experiment['name']!r}")
    if experiment.get("project_name", experiment["name"]) != experiment_name:
        raise ValueError(f"Experiment spec file {spec_path} must use project_name = {experiment_name!r} when project_name is present")
    defaults = payload.get("defaults", {})
    groups = payload.get("groups", {})
    template_path = (REPO_ROOT / experiment["template"]).resolve()
    return ExperimentSpec(
        name=experiment["name"],
        project_name=experiment.get("project_name", experiment["name"]),
        family=experiment.get("family", experiment["name"]),
        template_path=template_path,
        output_stem=experiment.get("output_stem", "mbl"),
        point_id_template=experiment["point_id_template"],
        output_dir_template=experiment["output_dir_template"],
        config_stem_template=experiment.get("config_stem_template", "{output_stem}_{point_id}"),
        token_map=dict(experiment["token_map"]),
        defaults_settings=defaults.get("settings", {}),
        groups={name: dict(group) for name, group in groups.items()},
        spec_path=spec_path,
    )


# Build the tag list used later for filtering points by substring.
def _point_tags(group_name: str, point_key: str, point_id: str, tokens: list[PointToken]) -> list[str]:
    tags = [group_name, point_key, point_id]
    tags.extend(token.full_token for token in tokens)
    return tags


# Normalize model sizes to strings so point construction stays uniform.
def _normalized_model_sizes(group: dict[str, Any]) -> list[str]:
    return [_stringify_value(value) for value in group.get("model_sizes", [])]


# Expand one TOML experiment spec into the full list of point definitions.
def build_points_from_experiment(experiment_name: str) -> tuple[ExperimentSpec, list[PointSpec]]:
    experiment_spec = load_experiment_spec(experiment_name)
    points: list[PointSpec] = []
    for group_name, group in experiment_spec.groups.items():
        group_settings = group.get("settings", {})
        batch = group.get("batch", {})
        group_model_sizes = _normalized_model_sizes(group)
        group_point_id_template = group.get("point_id_template", experiment_spec.point_id_template)
        group_output_dir_template = group.get("output_dir_template", experiment_spec.output_dir_template)
        group_config_stem_template = group.get("config_stem_template", experiment_spec.config_stem_template)
        scan_extent = group.get("scan_extent")

        for point_key, seed_data in batch.items():
            parsed_tokens = _parse_point_key(point_key, experiment_spec.token_map)  # Decode the compact point key first.
            token_values = {token.name: token.raw_value for token in parsed_tokens}
            point_model_size = token_values.get("L")
            if point_model_size is not None:
                if group_model_sizes and point_model_size not in group_model_sizes:
                    raise ValueError(
                        f"Point key '{point_key}' in group '{group_name}' has L={point_model_size} "
                        f"which is not in group.model_sizes={group_model_sizes}"
                    )
                model_sizes = [point_model_size]
            elif group_model_sizes:
                model_sizes = group_model_sizes
            else:
                inherited_model_size = _get_settings_value(_deep_merge(experiment_spec.defaults_settings, group_settings), "settings.model.model_size")
                if inherited_model_size is None:
                    raise ValueError(
                        f"Group '{group_name}' must provide model_sizes, settings.model.model_size, "
                        f"or include L in each point key"
                    )
                model_sizes = [inherited_model_size]

            for model_size in model_sizes:
                merged_settings = _deep_merge(experiment_spec.defaults_settings, group_settings)  # Group overrides default settings.
                point_token_values = dict(token_values)
                if "L" in experiment_spec.token_map and "L" not in point_token_values:
                    point_token_values["L"] = model_size
                    parsed_tokens_for_point = [
                        PointToken(name="L", raw_value=model_size, full_token=f"L{model_size}", settings_path=experiment_spec.token_map["L"]),
                        *parsed_tokens,
                    ]
                else:
                    parsed_tokens_for_point = list(parsed_tokens)

                for token in parsed_tokens_for_point:
                    _apply_settings_path(merged_settings, token.settings_path, token.raw_value)  # Tokens win over inherited settings.

                flattened_settings = _flatten_settings(merged_settings)
                model_type = flattened_settings.get("settings.model.model_type")
                if model_type is None:
                    raise ValueError(f"Missing settings.model.model_type for point '{point_key}' in group '{group_name}'")

                context = _template_context(experiment_spec, point_token_values, point_key=point_key, group_name=group_name)
                point_id = _render_template(group_point_id_template, context)
                output_dir_relpath = _render_template(group_output_dir_template, context)
                config_stem = _render_template(group_config_stem_template, {**context, "point_id": point_id})

                config_overrides = {
                    _cfg_key_from_settings_path(settings_path): value
                    for settings_path, value in flattened_settings.items()
                }
                config_overrides["storage::output_filepath"] = f"{output_dir_relpath}/{experiment_spec.output_stem}.h5"  # xDMRG++ writes into the point output directory.

                points.append(
                    PointSpec(
                        point_id=point_id,
                        config_stem=config_stem,
                        point_key=point_key,
                        source_experiment=experiment_spec.name,
                        source_group=group_name,
                        project_name=experiment_spec.project_name,
                        model_type=model_type,
                        output_stem=experiment_spec.output_stem,
                        config_relpath=f"configs/{config_stem}.cfg",
                        output_dir_relpath=output_dir_relpath,
                        status_relpath=f"status/{config_stem}.json",
                        log_dir_relpath=f"logs/{config_stem}",
                        event_dir_relpath=f"events/{config_stem}",
                        config_overrides=config_overrides,
                        settings_values=flattened_settings,
                        token_values=point_token_values,
                        tokens=parsed_tokens_for_point,
                        seed_ranges=[
                            SeedRange(offset=offset, extent=extent)
                            for offset, extent in zip(seed_data["seed_offset"], seed_data["seed_extent"])
                        ],
                        scan_extent=scan_extent,
                        requested_seed_count=sum(seed_data["seed_extent"]),
                        tags=_point_tags(group_name, point_key, point_id, parsed_tokens_for_point),
                    )
                )
    return experiment_spec, points
