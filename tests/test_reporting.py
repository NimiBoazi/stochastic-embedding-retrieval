import json

import numpy as np
import pytest

from stochastic_retrieval.encoding import inspect_embedding_file
from stochastic_retrieval.pipeline import _validate_stochasticity
from stochastic_retrieval.reporting import RunReporter


def test_embedding_inspection_reports_norms(tmp_path) -> None:
    path = tmp_path / "embeddings.npy"
    np.save(path, np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32))

    report = inspect_embedding_file(path, expected_count=2)

    assert report["items"] == 2
    assert report["dimension"] == 2
    assert report["minimum_norm"] == 1.0
    assert report["maximum_norm"] == 2.0


def test_embedding_inspection_rejects_non_finite_values(tmp_path) -> None:
    path = tmp_path / "embeddings.npy"
    np.save(path, np.array([[np.nan, 0.0]], dtype=np.float32))

    with pytest.raises(RuntimeError, match="Non-finite"):
        inspect_embedding_file(path, expected_count=1)


def test_stochasticity_validation_rejects_identical_samples() -> None:
    deterministic = np.ones((2, 3), dtype=np.float32)
    stochastic = np.stack([deterministic, deterministic])

    with pytest.raises(RuntimeError, match="identical"):
        _validate_stochasticity(deterministic, stochastic)


def test_reporter_records_failed_stage(tmp_path) -> None:
    reporter = RunReporter("test-run", tmp_path)

    with pytest.raises(ValueError, match="broken"):
        with reporter.stage("test-stage"):
            raise ValueError("broken")

    records = [
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
    ]
    assert [record["event"] for record in records] == [
        "stage_started",
        "stage_failed",
    ]
    assert records[-1]["error_type"] == "ValueError"
