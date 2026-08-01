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
    sample_counts: tuple[int, ...] = ()
    independent_sample_banks: bool = False
    document_samples: int = 1
    retrieval_k: int = 1000
    retrieval_backend: str = "auto"
    retrieval_query_batch_size: int = 128
    retrieval_corpus_batch_size: int = 50_000
    ranking_write_query_batch_size: int = 256
    trim_fraction: float = 0.20
    ranking_medoid_depth: int = 100
    majority_vote_depth: int = 100
    variance_penalty_lambda: float = 1.0
    metric_cutoffs: tuple[int, ...] = (10, 100, 1000)
    success_cutoffs: tuple[int, ...] = (1, 5, 10)
    methods: tuple[str, ...] = (
        "deterministic",
        "mean_embedding",
        "mean_score",
        "medoid_embedding",
        "trimmed_centroid",
        "rrf",
        "majority_vote",
        "ranking_medoid",
        "maximum_score",
        "variance_penalized_score",
        "oracle_best_of_n",
    )
    bootstrap_replicates: int = 2000

    @property
    def resolved_sample_counts(self) -> tuple[int, ...]:
        return self.sample_counts or (self.query_samples,)


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
    for key in ("sample_counts", "metric_cutoffs", "success_cutoffs", "methods"):
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


def load_sweep(path: str | Path) -> list[ProjectConfig]:
    config_path = Path(path).resolve()
    raw = _load_yaml_mapping(config_path)
    sweep_name = raw.get("name")
    model_references = raw.get("model_configs")
    dataset_references = raw.get("dataset_configs")
    if not isinstance(sweep_name, str) or not sweep_name:
        raise ValueError("Sweep configuration requires a non-empty 'name'")
    if not isinstance(model_references, list) or not model_references:
        raise ValueError("Sweep configuration requires non-empty 'model_configs'")
    if not isinstance(dataset_references, list) or not dataset_references:
        raise ValueError("Sweep configuration requires non-empty 'dataset_configs'")
    model_overrides = raw.get("model_overrides")
    if model_overrides is None:
        model_overrides = [{"label": None, "values": {}}]
    if not isinstance(model_overrides, list) or not model_overrides:
        raise ValueError("'model_overrides' must be a non-empty list when provided")

    experiment_template = _require_mapping(raw.get("experiment"), "experiment").copy()
    experiment_template.pop("name", None)
    for key in ("sample_counts", "metric_cutoffs", "success_cutoffs", "methods"):
        if key in experiment_template:
            experiment_template[key] = tuple(experiment_template[key])

    runs: list[ProjectConfig] = []
    for model_reference in model_references:
        model_path = _resolve_reference(model_reference, config_path, "model_configs")
        base_model = _load_yaml_mapping(model_path)
        for override in model_overrides:
            if not isinstance(override, dict):
                raise ValueError("Every 'model_overrides' entry must be a mapping")
            label = override.get("label")
            values = override.get("values", {})
            if label is not None and (not isinstance(label, str) or not label):
                raise ValueError("Model override labels must be non-empty strings")
            if not isinstance(values, dict):
                raise ValueError("Model override 'values' must be a mapping")
            model = ModelConfig(**{**base_model, **values})
            model_label = (
                f"{model_path.stem}-{label}" if label is not None else model_path.stem
            )
            for dataset_reference in dataset_references:
                dataset_path = _resolve_reference(
                    dataset_reference, config_path, "dataset_configs"
                )
                dataset = DatasetConfig(**_load_yaml_mapping(dataset_path))
                run_name = f"{sweep_name}-{dataset.name}-{model_label}"
                tags = {
                    **raw.get("tags", {}),
                    "sweep": sweep_name,
                    "model_config": model_path.stem,
                    "dataset_config": dataset_path.stem,
                    **({"model_override": label} if label is not None else {}),
                }
                runs.append(
                    ProjectConfig(
                        model=model,
                        dataset=dataset,
                        experiment=ExperimentConfig(
                            name=run_name,
                            **experiment_template,
                        ),
                        artifact_root=raw.get("artifact_root", "artifacts"),
                        device=raw.get("device", "auto"),
                        num_workers=raw.get("num_workers", 0),
                        tags=tags,
                    )
                )
    return runs


def _resolve_reference(value: Any, config_path: Path, key: str) -> Path:
    if not isinstance(value, str):
        raise ValueError(f"Every '{key}' entry must be a path string")
    return (config_path.parent / value).resolve()
