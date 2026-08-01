import pandas as pd

from stochastic_retrieval.artifacts import ArtifactStore
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
