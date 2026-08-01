import numpy as np

from stochastic_retrieval.evaluation import (
    average_precision,
    dataset_query_diagnostics,
    evaluate_rankings,
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


def test_evaluation_records_relevance_structure_and_qrels_overlap() -> None:
    ranking = Rankings(
        indices=np.array([[0, 2, 1]]),
        scores=np.array([[1.0, 0.8, 0.5]], dtype=np.float32),
    )
    qrels = {"q1": {"d1": 2, "d2": 1}}

    result = evaluate_rankings(
        "test",
        ranking,
        ["q1"],
        ["d1", "d2", "unjudged"],
        qrels,
        (3,),
    )
    diagnostics = dataset_query_diagnostics(["q1"], qrels)

    assert result.loc[0, "relevance_group"] == "multiple"
    assert result.loc[0, "qrels_overlap@3"] == 2 / 3
    assert diagnostics.loc[0, "relevant_documents"] == 2
