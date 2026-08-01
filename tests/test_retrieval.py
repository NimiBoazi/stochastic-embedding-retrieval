import numpy as np

from stochastic_retrieval.retrieval import (
    DenseRetriever,
    Rankings,
    anchored_centroid,
    deterministic_score_margin,
    exact_search,
    gated_ranking,
    majority_vote,
    maximum_score_rerank,
    mean_embedding,
    quartile_gate_masks,
    ranking_disagreement,
    ranking_medoid,
    reciprocal_rank_fusion,
    trimmed_centroid,
    variance_penalized_rerank,
)


def test_exact_search_returns_sorted_inner_products() -> None:
    queries = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    corpus = np.array(
        [[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]],
        dtype=np.float32,
    )

    result = exact_search(
        queries,
        corpus,
        k=2,
        query_batch_size=1,
        corpus_batch_size=2,
    )

    np.testing.assert_array_equal(result.indices, [[0, 1], [2, 1]])
    np.testing.assert_allclose(result.scores, [[1.0, 0.6], [1.0, 0.8]])


def test_mean_score_and_normalized_mean_have_same_ranking() -> None:
    samples = np.array(
        [
            [[1.0, 0.0]],
            [[0.8, 0.2]],
            [[0.7, -0.1]],
        ],
        dtype=np.float32,
    )
    corpus = np.array(
        [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]],
        dtype=np.float32,
    )

    raw = exact_search(samples.mean(axis=0), corpus, k=3)
    normalized = exact_search(mean_embedding(samples), corpus, k=3)

    np.testing.assert_array_equal(raw.indices, normalized.indices)


def test_anchored_centroid_interpolates_and_validates_alpha() -> None:
    deterministic = np.array([[1.0, 0.0]], dtype=np.float32)
    samples = np.array([[[0.0, 1.0]], [[0.0, 1.0]]], dtype=np.float32)

    np.testing.assert_allclose(
        anchored_centroid(deterministic, samples, 0.0),
        deterministic,
    )
    np.testing.assert_allclose(
        anchored_centroid(deterministic, samples, 1.0),
        [[0.0, 1.0]],
    )
    middle = anchored_centroid(deterministic, samples, 0.5)
    np.testing.assert_allclose(middle, [[2**-0.5, 2**-0.5]], atol=1e-6)


def test_gated_ranking_and_label_free_diagnostics() -> None:
    deterministic = Rankings(
        np.array([[0, 1], [1, 0]]),
        np.array([[1.0, 0.9], [0.8, 0.5]], dtype=np.float32),
    )
    alternative = Rankings(
        np.array([[1, 0], [0, 1]]),
        np.array([[0.7, 0.6], [0.9, 0.4]], dtype=np.float32),
    )

    gated = gated_ranking(deterministic, alternative, np.array([False, True]))

    np.testing.assert_array_equal(gated.indices, [[0, 1], [0, 1]])
    np.testing.assert_allclose(
        deterministic_score_margin(deterministic, 1),
        [0.1, 0.3],
        atol=1e-7,
    )
    np.testing.assert_allclose(
        ranking_disagreement([deterministic], depth=2),
        [0.0, 0.0],
    )
    np.testing.assert_allclose(
        ranking_disagreement([deterministic, alternative], depth=1),
        [1.0, 1.0],
    )


def test_quartile_gate_masks_and_n1_disagreement_fallback() -> None:
    margins = np.array([1.0, 2.0, 3.0, 4.0])
    disagreement = np.array([0.1, 0.2, 0.3, 0.4])

    margin_mask, margin_threshold, disagreement_mask, threshold = (
        quartile_gate_masks(margins, disagreement, sample_count=4)
    )

    np.testing.assert_array_equal(margin_mask, [True, False, False, False])
    np.testing.assert_array_equal(disagreement_mask, [False, False, False, True])
    assert margin_threshold == 1.75
    assert threshold == 0.325

    _, _, n1_mask, _ = quartile_gate_masks(
        margins,
        disagreement,
        sample_count=1,
    )
    assert not n1_mask.any()


def test_rank_fusion_prefers_repeated_documents() -> None:
    first = Rankings(
        indices=np.array([[1, 2, 3]]),
        scores=np.array([[0.9, 0.8, 0.7]], dtype=np.float32),
    )
    second = Rankings(
        indices=np.array([[2, 4, 1]]),
        scores=np.array([[0.9, 0.8, 0.7]], dtype=np.float32),
    )

    rrf = reciprocal_rank_fusion([first, second], k=3)
    vote = majority_vote([first, second], k=3)

    assert rrf.indices[0, 0] == 2
    assert set(vote.indices[0, :2]) == {1, 2}


def test_majority_vote_counts_votes_at_depth_and_breaks_ties_by_rrf() -> None:
    scores = np.array([[0.9, 0.8, 0.7]], dtype=np.float32)
    rankings = [
        Rankings(np.array([[1, 9, 8]]), scores),
        Rankings(np.array([[2, 9, 8]]), scores),
        Rankings(np.array([[2, 9, 8]]), scores),
    ]

    result = majority_vote(rankings, k=4, depth=1)

    # Document 2 has two top-1 votes, document 1 has one; documents 9 and 8
    # have no votes at depth 1 and are ordered by their full-list RRF sums,
    # even though document 9 has the highest RRF sum overall.
    np.testing.assert_array_equal(result.indices, [[2, 1, 9, 8]])


def test_reusable_numpy_retriever_handles_multiple_query_sets() -> None:
    corpus = np.eye(2, dtype=np.float32)
    retriever = DenseRetriever(corpus, k=1, backend="numpy")

    first = retriever.search(np.array([[1.0, 0.0]], dtype=np.float32))
    second = retriever.search(np.array([[0.0, 1.0]], dtype=np.float32))

    assert first.indices[0, 0] == 0
    assert second.indices[0, 0] == 1


def test_trimmed_centroid_removes_embedding_outlier() -> None:
    samples = np.array(
        [
            [[1.0, 0.0]],
            [[0.99, 0.01]],
            [[0.98, -0.02]],
            [[1.0, 0.02]],
            [[-1.0, 0.0]],
        ],
        dtype=np.float32,
    )

    result = trimmed_centroid(samples, trim_fraction=0.20)

    assert result[0, 0] > 0.99
    assert abs(result[0, 1]) < 0.02


def test_ranking_medoid_chooses_consensus_ranking() -> None:
    rankings = [
        Rankings(np.array([[0, 1, 2]]), np.array([[1.0, 0.9, 0.8]])),
        Rankings(np.array([[0, 1, 3]]), np.array([[1.0, 0.9, 0.8]])),
        Rankings(np.array([[4, 5, 6]]), np.array([[1.0, 0.9, 0.8]])),
    ]

    result = ranking_medoid(rankings, depth=3)

    np.testing.assert_array_equal(result.indices, rankings[0].indices)


def test_score_aggregators_distinguish_extreme_and_consistent_documents() -> None:
    samples = np.array([[[1.0, 0.0]], [[0.0, 1.0]]], dtype=np.float32)
    corpus = np.array(
        [[1.0, 0.0], [0.0, 1.0], [0.6, 0.6]],
        dtype=np.float32,
    )
    sampled_rankings = [
        Rankings(np.array([[0, 2, 1]]), np.array([[1.0, 0.6, 0.0]])),
        Rankings(np.array([[1, 2, 0]]), np.array([[1.0, 0.6, 0.0]])),
    ]

    maximum = maximum_score_rerank(samples, corpus, sampled_rankings, k=3)
    consistent = variance_penalized_rerank(
        samples,
        corpus,
        sampled_rankings,
        k=3,
        penalty=1.0,
    )

    assert maximum.indices[0, 0] == 0
    assert consistent.indices[0, 0] == 2
