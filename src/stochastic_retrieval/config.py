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


def load_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("The configuration root must be a mapping")

    model = ModelConfig(**_require_mapping(raw.get("model"), "model"))
    dataset = DatasetConfig(**_require_mapping(raw.get("dataset"), "dataset"))
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
