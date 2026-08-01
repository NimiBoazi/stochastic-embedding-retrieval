import math

import numpy as np
import pandas as pd
import pytest

from stochastic_retrieval.evaluation import (
    dataset_query_diagnostics,
    evaluate_rankings,
    ndcg,
    oracle_best_of_n,
    paired_bootstrap_comparisons,
    ranking_frame,
    reciprocal_rank,
    summarize,
)
from stochastic_retrieval.retrieval import Rankings


def test_linear_gain_ndcg_matches_hand_computation() -> None:
    retrieved = ["d1", "d2", "d3"]
    qrels = {"d1": 2, "d3": 1}

    dcg = 2 / math.log2(2) + 0 / math.log2(3) + 1 / math.log2(4)
    ideal = 2 / math.log2(2) + 1 / math.log2(3)

    assert ndcg(retrieved, qrels, 3) == pytest.approx(dcg / ideal)
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
    assert set(selections["candidate_source"]) == {"stochastic_only"}


def test_oracle_cannot_fall_back_to_a_better_deterministic_ranking() -> None:
    stochastic = Rankings(
        indices=np.array([[1, 0]]),
        scores=np.array([[1.0, 0.5]], dtype=np.float32),
    )
    # A deterministic ranking would put relevant d1 first, but it is deliberately
    # not an oracle candidate.
    oracle, selections = oracle_best_of_n(
        [stochastic],
        ["q1"],
        ["d1", "d2"],
        {"q1": {"d1": 1}},
        selection_cutoff=1,
    )

    assert oracle.indices[0, 0] == 1
    assert selections.loc[0, "selection_ndcg@1"] == 0.0
    assert selections.loc[0, "candidate_source"] == "stochastic_only"


def test_evaluate_rankings_matches_trec_eval_semantics() -> None:
    # Retrieved order: d1 (grade 2), unjudged, d2 (grade 1).
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

    dcg = 2 / math.log2(2) + 1 / math.log2(4)
    ideal = 2 / math.log2(2) + 1 / math.log2(3)
    assert result.loc[0, "ndcg@3"] == pytest.approx(dcg / ideal)
    assert result.loc[0, "recall@3"] == 1.0
    assert result.loc[0, "map@3"] == pytest.approx((1.0 + 2 / 3) / 2)
    assert result.loc[0, "mrr@3"] == 1.0
    assert result.loc[0, "relevance_group"] == "multiple"
    assert result.loc[0, "qrels_overlap@3"] == 2 / 3


def test_evaluate_rankings_preserves_submitted_order_despite_tied_scores() -> None:
    # Both documents share one raw score; trec_eval must not re-sort them by
    # doc id because the run is submitted with rank-derived scores.
    ranking = Rankings(
        indices=np.array([[1, 0]]),
        scores=np.array([[0.5, 0.5]], dtype=np.float32),
    )
    qrels = {"q1": {"a": 1}}

    result = evaluate_rankings("test", ranking, ["q1"], ["a", "b"], qrels, (2,))

    # Relevant document "a" sits at rank 2 as submitted.
    assert result.loc[0, "mrr@2"] == 0.5
    assert result.loc[0, "ndcg@2"] == pytest.approx(1 / math.log2(3))


def test_success_cutoffs_report_any_relevant_hit() -> None:
    # Relevant document sits at rank 3: success@1 misses it, success@5 finds it.
    ranking = Rankings(
        indices=np.array([[2, 1, 0, 3, 4]]),
        scores=np.ones((1, 5), dtype=np.float32),
    )
    qrels = {"q1": {"d0": 1}}

    result = evaluate_rankings(
        "test",
        ranking,
        ["q1"],
        ["d0", "d1", "d2", "d3", "d4"],
        qrels,
        (5,),
        success_cutoffs=(1, 5),
    )

    assert result.loc[0, "success@1"] == 0.0
    assert result.loc[0, "success@5"] == 1.0


def test_evaluate_rankings_keeps_queries_without_judgments() -> None:
    ranking = Rankings(
        indices=np.array([[0], [0]]),
        scores=np.ones((2, 1), dtype=np.float32),
    )
    qrels = {"q1": {"d1": 1}}

    result = evaluate_rankings("test", ranking, ["q1", "q2"], ["d1"], qrels, (1,))
    diagnostics = dataset_query_diagnostics(["q1", "q2"], qrels)

    assert list(result["query_id"]) == ["q1", "q2"]
    assert result.loc[1, "ndcg@1"] == 0.0
    assert result.loc[1, "relevance_group"] == "none"
    assert diagnostics.loc[1, "relevant_documents"] == 0


def test_multi_n_summaries_and_bootstrap_never_mix_sample_counts() -> None:
    rows = []
    for sample_count, candidate_values in ((1, [1.0, 1.0]), (2, [0.0, 0.0])):
        for query_id, baseline, candidate in zip(
            ("q1", "q2"), (0.5, 0.5), candidate_values, strict=True
        ):
            rows.extend(
                [
                    {
                        "method": "deterministic",
                        "sample_count": sample_count,
                        "query_id": query_id,
                        "ndcg@10": baseline,
                    },
                    {
                        "method": "candidate",
                        "sample_count": sample_count,
                        "query_id": query_id,
                        "ndcg@10": candidate,
                    },
                ]
            )
    per_query = pd.DataFrame(rows)

    summary = summarize(per_query)
    comparisons = paired_bootstrap_comparisons(
        per_query,
        baseline="deterministic",
        metric="ndcg@10",
        replicates=100,
        seed=42,
    )

    assert set(summary["sample_count"]) == {1, 2}
    deltas = comparisons.set_index("sample_count")["mean_delta"]
    assert deltas.loc[1] == 0.5
    assert deltas.loc[2] == -0.5


def test_ranking_frame_records_sample_count() -> None:
    frame = ranking_frame(
        "method",
        Rankings(np.array([[0]]), np.array([[1.0]], dtype=np.float32)),
        ["q1"],
        ["d1"],
        sample_count=8,
    )

    assert frame.loc[0, "sample_count"] == 8
