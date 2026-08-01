import numpy as np

from stochastic_retrieval.retrieval import (
    DenseRetriever,
    Rankings,
    exact_search,
    majority_vote,
    mean_embedding,
    reciprocal_rank_fusion,
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


def test_reusable_numpy_retriever_handles_multiple_query_sets() -> None:
    corpus = np.eye(2, dtype=np.float32)
    retriever = DenseRetriever(corpus, k=1, backend="numpy")

    first = retriever.search(np.array([[1.0, 0.0]], dtype=np.float32))
    second = retriever.search(np.array([[0.0, 1.0]], dtype=np.float32))

    assert first.indices[0, 0] == 0
    assert second.indices[0, 0] == 1
