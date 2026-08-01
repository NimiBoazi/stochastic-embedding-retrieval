import json

import numpy as np
import pytest
from torch import nn

from stochastic_retrieval.config import (
    DatasetConfig,
    ExperimentConfig,
    ModelConfig,
    ProjectConfig,
)
from stochastic_retrieval.data import TextRecord
from stochastic_retrieval.pipeline import (
    _encode_if_missing,
    _sample_seed_schedule,
    run_experiment,
)


def test_independent_seed_schedule_has_255_unique_samples() -> None:
    counts = (1, 2, 4, 8, 16, 32, 64, 128)

    schedule = _sample_seed_schedule(42, counts, independent=True)
    seeds = [seed for count in counts for seed in schedule[count]]

    assert len(seeds) == 255
    assert len(set(seeds)) == 255
    assert schedule[1][0] not in schedule[2]


def test_embedding_cache_rejects_wrong_id_order(tmp_path) -> None:
    output_path = tmp_path / "sample.npy"
    ids_path = tmp_path / "ids.jsonl"
    np.save(output_path, np.eye(2, dtype=np.float32))
    ids_path.write_text('"q2"\n"q1"\n')

    with pytest.raises(RuntimeError, match="IDs or ordering"):
        _encode_if_missing(
            encoder=None,  # type: ignore[arg-type]
            output_path=output_path,
            ids_path=ids_path,
            records=[],
            count=2,
            prefix="",
            seed=42,
            stochastic=False,
            description="cached",
            expected_ids=["q1", "q2"],
        )


def test_multi_n_pipeline_encodes_documents_once(monkeypatch, tmp_path) -> None:
    class FakeDataset:
        def __init__(self, _config) -> None:
            self.query_count = 2
            self.document_count = 3

        def iter_queries(self):
            yield TextRecord("q1", "first query")
            yield TextRecord("q2", "second query")

        def iter_documents(self):
            yield TextRecord("d1", "first document")
            yield TextRecord("d2", "second document")
            yield TextRecord("d3", "third document")

        def qrels(self):
            return {"q1": {"d1": 1}, "q2": {"d2": 1}}

    class FakeEncoder:
        calls: list[tuple[str, bool, int]] = []

        def __init__(self, config, _device) -> None:
            self.config = config
            self.device = "cpu"
            self.dimension = 2
            self.pooling = "mean"
            self.attention_implementation = "eager"
            self.resolved_revision = config.revision
            self.model = nn.Sequential(nn.Dropout(0.1))

        def verify_dropout_effect(self):
            return {"attention": True, "hidden": True}

        def encode_to_file(
            self,
            records,
            count,
            output_path,
            ids_path,
            prefix,
            seed,
            stochastic,
            description,
        ):
            del prefix
            records = list(records)
            self.calls.append((description, stochastic, seed))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if description.startswith("Deterministic documents"):
                array = np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]], dtype=np.float32)
            elif stochastic:
                rng = np.random.default_rng(seed)
                array = rng.normal(size=(count, 2)).astype(np.float32)
            else:
                array = np.array([[1.0, 0.1], [0.1, 1.0]], dtype=np.float32)
            array /= np.linalg.norm(array, axis=1, keepdims=True)
            np.save(output_path, array)
            with ids_path.open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record.item_id) + "\n")
            return {
                "items": count,
                "dimension": 2,
                "minimum_norm": float(np.linalg.norm(array, axis=1).min()),
                "maximum_norm": float(np.linalg.norm(array, axis=1).max()),
                "mean_norm": float(np.linalg.norm(array, axis=1).mean()),
                "cached": False,
                "seed": seed,
                "stochastic": stochastic,
                "enabled_dropout_modules": int(stochastic),
                "dropout_probabilities": {"0.1": 1},
                "pooling": "mean",
            }

        def encode_records(self, records, prefix, seed, stochastic):
            del records, prefix, seed, stochastic
            return np.array([[1.0, 0.1], [0.1, 1.0]], dtype=np.float32) / np.sqrt(
                1.01
            )

    monkeypatch.setattr(
        "stochastic_retrieval.pipeline.IRDatasetAdapter", FakeDataset
    )
    monkeypatch.setattr(
        "stochastic_retrieval.pipeline.SentenceEmbeddingEncoder", FakeEncoder
    )
    config = ProjectConfig(
        model=ModelConfig(
            name="fake/model",
            revision="commit",
            pooling="mean",
            expected_dimension=2,
        ),
        dataset=DatasetConfig(name="fake", ir_dataset_id="fake/test"),
        experiment=ExperimentConfig(
            name="multi-n",
            query_samples=4,
            sample_counts=(1, 2, 4),
            independent_sample_banks=True,
            retrieval_k=3,
            metric_cutoffs=(1,),
            success_cutoffs=(1,),
            methods=("deterministic", "mean_embedding"),
            bootstrap_replicates=20,
        ),
        artifact_root=str(tmp_path / "artifacts"),
        device="cpu",
    )

    store = run_experiment(config, tmp_path)
    calls_after_first_run = len(FakeEncoder.calls)
    run_experiment(config, tmp_path)

    document_calls = [
        call for call in FakeEncoder.calls if call[0].startswith("Deterministic documents")
    ]
    deterministic_query_calls = [
        call for call in FakeEncoder.calls if call[0] == "Deterministic queries"
    ]
    stochastic_calls = [call for call in FakeEncoder.calls if call[1]]
    summary = __import__("pandas").read_parquet(
        store.run_dir / "metrics" / "summary.parquet"
    )

    assert len(document_calls) == 1
    assert len(deterministic_query_calls) == 1
    assert len(stochastic_calls) == 7
    assert len(FakeEncoder.calls) == calls_after_first_run
    assert set(summary["sample_count"]) == {1, 2, 4}
