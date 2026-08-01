from dataclasses import replace

import pandas as pd

from stochastic_retrieval.artifacts import ArtifactStore, document_cache_key
from stochastic_retrieval.config import (
    DatasetConfig,
    ExperimentConfig,
    ModelConfig,
    ProjectConfig,
)


def test_dataframe_chunks_create_single_parquet_artifact(tmp_path) -> None:
    config = ProjectConfig(
        model=ModelConfig(name="test/model"),
        dataset=DatasetConfig(name="test", ir_dataset_id="test/dataset"),
        experiment=ExperimentConfig(name="test-run"),
        artifact_root=str(tmp_path / "artifacts"),
    )
    store = ArtifactStore(tmp_path, config)
    frames = [
        pd.DataFrame({"query_id": ["q1"], "score": [1.0]}),
        pd.DataFrame({"query_id": ["q2"], "score": [0.5]}),
    ]

    path = store.write_dataframe_chunks(frames, "rankings", "streamed")
    result = pd.read_parquet(path)

    assert result["query_id"].tolist() == ["q1", "q2"]


def test_document_cache_key_ignores_experiment_sampling_settings() -> None:
    base = ProjectConfig(
        model=ModelConfig(name="test/model", revision="abc", pooling="mean"),
        dataset=DatasetConfig(name="test", ir_dataset_id="test/dataset"),
        experiment=ExperimentConfig(name="first", query_samples=8),
    )
    different_sampling = replace(
        base,
        experiment=ExperimentConfig(
            name="second",
            query_samples=128,
            sample_counts=(1, 2, 4, 8, 16, 32, 64, 128),
            independent_sample_banks=True,
        ),
    )
    different_revision = replace(
        base,
        model=replace(base.model, revision="def"),
    )

    assert document_cache_key(base) == document_cache_key(different_sampling)
    assert document_cache_key(base) != document_cache_key(different_revision)
