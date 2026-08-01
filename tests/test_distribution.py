import numpy as np
import pandas as pd

from stochastic_retrieval.distribution import (
    centroid_convergence,
    cluster_diagnostics,
    embedding_distribution,
    retrieval_distribution,
    retrieval_document_distribution,
)
from stochastic_retrieval.retrieval import Rankings


def test_embedding_distribution_handles_symmetric_singular_cloud_and_n1() -> None:
    samples = np.array(
        [
            [[1.0, 0.1, 0.0, 0.0]],
            [[1.0, -0.1, 0.0, 0.0]],
            [[1.0, 0.2, 0.0, 0.0]],
            [[1.0, -0.2, 0.0, 0.0]],
        ],
        dtype=np.float32,
    )
    frame = embedding_distribution(
        samples,
        np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        ["q1"],
    )

    assert frame.loc[0, "deterministic_centroid_cosine"] > 0.999
    assert 0.0 <= frame.loc[0, "effective_rank"] <= 1.1

    n1 = embedding_distribution(samples[:1], samples[0], ["q1"])
    assert n1.loc[0, "covariance_trace"] == 0.0
    assert n1.loc[0, "pc1_variance_share"] == 0.0


def test_embedding_distribution_flags_heavy_tailed_outlier() -> None:
    angles = np.array([0.0, 0.01, -0.01, 0.02, -0.02, np.pi / 2])
    samples = np.stack(
        [np.cos(angles), np.sin(angles)],
        axis=1,
    )[:, None, :].astype(np.float32)

    frame = embedding_distribution(samples, np.array([[1.0, 0.0]]), ["q1"])

    assert frame.loc[0, "centroid_distance_robust_outlier_fraction"] > 0


def test_spherical_clustering_finds_two_stable_modes() -> None:
    rng = np.random.default_rng(7)
    first = np.column_stack([np.ones(20), rng.normal(0, 0.02, 20)])
    second = np.column_stack([-np.ones(20), rng.normal(0, 0.02, 20)])
    samples = np.concatenate([first, second])[:, None, :].astype(np.float32)

    frame = cluster_diagnostics(
        samples,
        ["q1"],
        cluster_counts=(2,),
        bootstrap_replicates=3,
        seed=4,
    )

    assert frame.loc[0, "silhouette_cosine"] > 0.9
    assert frame.loc[0, "bootstrap_rand_stability"] > 0.9


def test_centroid_convergence_is_reproducible() -> None:
    rng = np.random.default_rng(9)
    samples = rng.normal(size=(8, 2, 4)).astype(np.float32)

    left = centroid_convergence(
        samples,
        ["q1", "q2"],
        subset_counts=(1, 4, 8),
        bootstrap_replicates=3,
        seed=2,
    )
    right = centroid_convergence(
        samples,
        ["q1", "q2"],
        subset_counts=(1, 4, 8),
        bootstrap_replicates=3,
        seed=2,
    )

    assert left.equals(right)


def test_retrieval_distribution_handles_boundary_swaps_and_n1() -> None:
    deterministic = Rankings(
        np.array([[0, 1, 2]]),
        np.array([[1.0, 0.9, 0.8]], dtype=np.float32),
    )
    swapped = Rankings(
        np.array([[1, 0, 2]]),
        np.array([[1.1, 1.0, 0.8]], dtype=np.float32),
    )
    frame = retrieval_distribution(
        [deterministic, swapped],
        deterministic,
        ["q1"],
        ["d1", "d2", "d3"],
        {"q1": {"d1": 1}},
        depth=2,
    )

    assert frame.loc[0, "top1_entropy"] == 1.0
    assert frame.loc[0, "relevant_top1_probability"] == 0.5
    assert frame.loc[0, "mean_pairwise_top10_jaccard"] == 1.0

    single = retrieval_distribution(
        [deterministic],
        deterministic,
        ["q1"],
        ["d1", "d2", "d3"],
        {"q1": {"d1": 1}},
        depth=2,
    )
    assert single.loc[0, "mean_pairwise_top10_rbo"] == 1.0

    documents = retrieval_document_distribution(
        [deterministic, swapped],
        deterministic,
        ["q1"],
        ["d1", "d2", "d3"],
        {"q1": {"d1": 1}},
        depth=2,
    ).set_index("document_id")
    assert documents.loc["d1", "inclusion_probability@1"] == 0.5
    assert documents.loc["d1", "rank_mean"] == 1.5
    assert documents.loc["d1", "is_relevant"]


def test_retrieval_distribution_matches_query_chunking() -> None:
    deterministic = Rankings(
        np.array([[0, 1, 2], [2, 1, 0]]),
        np.array([[1.0, 0.8, 0.6], [1.0, 0.8, 0.6]], dtype=np.float32),
    )
    alternative = Rankings(
        np.array([[1, 0, 2], [1, 2, 0]]),
        np.array([[1.1, 0.9, 0.6], [1.1, 0.9, 0.6]], dtype=np.float32),
    )
    query_ids = ["q1", "q2"]
    document_ids = ["d1", "d2", "d3"]
    qrels = {"q1": {"d1": 1}, "q2": {"d3": 1}}
    full = retrieval_distribution(
        [deterministic, alternative],
        deterministic,
        query_ids,
        document_ids,
        qrels,
        depth=2,
    )
    chunks = []
    for query_index, query_id in enumerate(query_ids):
        chunks.append(
            retrieval_distribution(
                [
                    Rankings(
                        deterministic.indices[query_index : query_index + 1],
                        deterministic.scores[query_index : query_index + 1],
                    ),
                    Rankings(
                        alternative.indices[query_index : query_index + 1],
                        alternative.scores[query_index : query_index + 1],
                    ),
                ],
                Rankings(
                    deterministic.indices[query_index : query_index + 1],
                    deterministic.scores[query_index : query_index + 1],
                ),
                [query_id],
                document_ids,
                qrels,
                depth=2,
            )
        )

    assert full.equals(pd.concat(chunks, ignore_index=True))
