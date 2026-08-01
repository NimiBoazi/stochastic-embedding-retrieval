import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from stochastic_retrieval.config import ModelConfig
from stochastic_retrieval.data import TextRecord
from stochastic_retrieval.encoding import SentenceEmbeddingEncoder

MODEL_CASES = [
    ("bge_base_en_v1_5.yaml", 768, "cls"),
    ("e5_base_v2.yaml", 768, "mean"),
    ("contriever.yaml", 768, "mean"),
    ("gtr_t5_base.yaml", 768, "mean"),
    ("bge_large_en_v1_5.yaml", 1024, "cls"),
]


@pytest.mark.skipif(
    os.getenv("RUN_MODEL_CONTRACT_TESTS") != "1",
    reason="Set RUN_MODEL_CONTRACT_TESTS=1 to download and validate all checkpoints",
)
@pytest.mark.parametrize(("config_name", "dimension", "pooling"), MODEL_CASES)
def test_real_model_contract(
    config_name: str,
    dimension: int,
    pooling: str,
    tmp_path: Path,
) -> None:
    project_root = Path(__file__).resolve().parents[1]
    raw = yaml.safe_load(
        (project_root / "configs" / "models" / config_name).read_text()
    )
    config = ModelConfig(**raw)
    encoder = SentenceEmbeddingEncoder(config, device="cpu")
    records = [
        TextRecord("q1", "What evidence supports this scientific claim?"),
        TextRecord("q2", "A second retrieval query."),
    ]

    assert encoder.dimension == dimension
    assert encoder.pooling == pooling

    deterministic_path = tmp_path / f"{config_name}-deterministic.npy"
    stochastic_path = tmp_path / f"{config_name}-stochastic.npy"
    encoder.encode_to_file(
        records,
        len(records),
        deterministic_path,
        tmp_path / f"{config_name}-deterministic.ids",
        config.query_prefix,
        seed=42,
        stochastic=False,
        description="contract deterministic",
    )
    report = encoder.encode_to_file(
        records,
        len(records),
        stochastic_path,
        tmp_path / f"{config_name}-stochastic.ids",
        config.query_prefix,
        seed=43,
        stochastic=True,
        description="contract stochastic",
    )
    deterministic = np.load(deterministic_path)
    stochastic = np.load(stochastic_path)

    np.testing.assert_allclose(np.linalg.norm(deterministic, axis=1), 1.0, atol=1e-5)
    assert not np.array_equal(deterministic, stochastic)
    assert report["enabled_dropout_modules"] > 0
