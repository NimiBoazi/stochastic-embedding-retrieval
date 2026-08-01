import numpy as np

from stochastic_retrieval.evaluation import (
    average_precision,
    ndcg,
    oracle_best_of_n,
    recall,
    reciprocal_rank,
)
from stochastic_retrieval.retrieval import Rankings


def test_metrics_on_simple_ranking() -> None:
    retrieved = ["d1", "d2", "d3"]
    qrels = {"d1": 2, "d3": 1}

    assert ndcg(retrieved, qrels, 3) < 1.0
    assert recall(retrieved, qrels, 2) == 0.5
    assert average_precision(retrieved, qrels, 3) == (1.0 + 2 / 3) / 2
    assert reciprocal_rank(retrieved, qrels, 3) == 1.0


def test_oracle_selects_best_sample_per_query() -> None:
    first = Rankings(
        indices=np.array([[0, 1], [0, 1]]),
        scores=np.ones((2, 2), dtype=np.float32),
    )
    second = Rankings(
        indices=np.array([[1, 0], [1, 0]]),
        scores=np.ones((2, 2), dtype=np.float32),
    )
    qrels = {"q1": {"d1": 1}, "q2": {"d2": 1}}

    oracle, selections = oracle_best_of_n(
        [first, second],
        ["q1", "q2"],
        ["d1", "d2"],
        qrels,
        selection_cutoff=1,
    )

    np.testing.assert_array_equal(oracle.indices[:, 0], [0, 1])
    np.testing.assert_array_equal(selections["selected_sample"], [0, 1])
