from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    name: str
    revision: str | None = None
    query_prefix: str = ""
    document_prefix: str = ""
    pooling: str = "model_default"
    expected_dimension: int | None = None
    batch_size: int = 32
    max_length: int = 512
    normalize: bool = True
    dropout_scope: str = "all"
    dropout_probability: float | None = None


@dataclass(frozen=True)
class DatasetConfig:
    ir_dataset_id: str
    name: str
    query_limit: int | None = None
    document_limit: int | None = None


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    seed: int = 42
    query_samples: int = 8
    document_samples: int = 1
    retrieval_k: int = 1000
    metric_cutoffs: tuple[int, ...] = (10, 100, 1000)
    methods: tuple[str, ...] = (
        "deterministic",
        "mean_embedding",
        "mean_score",
        "medoid_embedding",
        "rrf",
        "majority_vote",
        "oracle_best_of_n",
    )
    bootstrap_replicates: int = 2000


@dataclass(frozen=True)
class ProjectConfig:
    model: ModelConfig
    dataset: DatasetConfig
    experiment: ExperimentConfig
    artifact_root: str = "artifacts"
    device: str = "auto"
    num_workers: int = 0
    tags: dict[str, str] = field(default_factory=dict)

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _require_mapping(value: Any, key: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Configuration key '{key}' must be a mapping")
    return value


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration file must contain a mapping: {path}")
    return value


def _resolve_section(raw: dict[str, Any], key: str, config_path: Path) -> dict[str, Any]:
    inline = raw.get(key)
    reference = raw.get(f"{key}_config")
    if inline is not None and reference is not None:
        raise ValueError(f"Use either '{key}' or '{key}_config', not both")
    if inline is not None:
        return _require_mapping(inline, key)
    if not isinstance(reference, str):
        raise ValueError(f"Missing configuration key '{key}' or '{key}_config'")
    referenced_path = (config_path.parent / reference).resolve()
    return _load_yaml_mapping(referenced_path)


def load_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path).resolve()
    raw = _load_yaml_mapping(config_path)

    model = ModelConfig(**_resolve_section(raw, "model", config_path))
    dataset = DatasetConfig(**_resolve_section(raw, "dataset", config_path))
    experiment_raw = _require_mapping(raw.get("experiment"), "experiment")
    for key in ("metric_cutoffs", "methods"):
        if key in experiment_raw:
            experiment_raw[key] = tuple(experiment_raw[key])
    experiment = ExperimentConfig(**experiment_raw)

    return ProjectConfig(
        model=model,
        dataset=dataset,
        experiment=experiment,
        artifact_root=raw.get("artifact_root", "artifacts"),
        device=raw.get("device", "auto"),
        num_workers=raw.get("num_workers", 0),
        tags=raw.get("tags", {}),
    )
