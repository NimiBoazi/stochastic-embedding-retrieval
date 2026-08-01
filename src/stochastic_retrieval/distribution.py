from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew, spearmanr

from stochastic_retrieval.retrieval import (
    DenseRetriever,
    Rankings,
    embedding_medoid_indices,
    l2_normalize,
)


def _geometric_median(
    vectors: np.ndarray,
    iterations: int = 25,
    tolerance: float = 1e-6,
) -> np.ndarray:
    estimate = vectors.mean(axis=0)
    for _ in range(iterations):
        distances = np.linalg.norm(vectors - estimate, axis=1)
        if np.any(distances <= tolerance):
            return vectors[int(np.argmin(distances))]
        weights = 1.0 / np.maximum(distances, tolerance)
        updated = np.average(vectors, axis=0, weights=weights)
        if np.linalg.norm(updated - estimate) <= tolerance:
            estimate = updated
            break
        estimate = updated
    return estimate


def _distance_summary(values: np.ndarray, prefix: str) -> dict[str, float]:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    threshold = median + 3.0 * 1.4826 * mad
    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_median": median,
        f"{prefix}_std": float(values.std()),
        f"{prefix}_p90": float(np.quantile(values, 0.90)),
        f"{prefix}_p95": float(np.quantile(values, 0.95)),
        f"{prefix}_skewness": (
            float(skew(values, bias=False)) if len(values) > 2 else 0.0
        ),
        f"{prefix}_robust_outlier_fraction": float((values > threshold).mean())
        if mad > 0
        else 0.0,
    }


def embedding_distribution(
    samples: np.ndarray,
    deterministic: np.ndarray,
    query_ids: list[str],
) -> pd.DataFrame:
    """Per-query bias, radial shape, and low-rank covariance diagnostics."""
    normalized = l2_normalize(samples)
    deterministic = l2_normalize(deterministic)
    centroid = l2_normalize(normalized.mean(axis=0))
    medoid_indices = embedding_medoid_indices(normalized)
    rows: list[dict[str, float | str]] = []
    for query_index, query_id in enumerate(query_ids):
        vectors = normalized[:, query_index, :]
        center = centroid[query_index]
        medoid = vectors[medoid_indices[query_index]]
        geometric = l2_normalize(_geometric_median(vectors))
        centroid_distances = 1.0 - vectors @ center
        medoid_distances = 1.0 - vectors @ medoid
        deterministic_distance = float(
            1.0 - deterministic[query_index] @ center
        )
        centered = vectors - vectors.mean(axis=0, keepdims=True)
        if len(vectors) > 1 and np.any(centered):
            _, singular_values, right = np.linalg.svd(
                centered, full_matrices=False
            )
            eigenvalues = singular_values**2 / (len(vectors) - 1)
        else:
            right = np.zeros((1, vectors.shape[1]), dtype=np.float32)
            eigenvalues = np.zeros(1, dtype=np.float32)
        total = float(eigenvalues.sum())
        shares = eigenvalues / total if total > 0 else eigenvalues
        effective_rank = (
            float(total**2 / np.square(eigenvalues).sum())
            if total > 0
            else 0.0
        )
        bias = deterministic[query_index] - center
        bias_norm = float(np.linalg.norm(bias))
        bias_pc1_alignment = (
            float(abs((bias / bias_norm) @ right[0]))
            if bias_norm > 1e-12 and total > 0
            else 0.0
        )
        row: dict[str, float | str] = {
            "query_id": query_id,
            "deterministic_centroid_cosine": float(
                deterministic[query_index] @ center
            ),
            "deterministic_centroid_euclidean": bias_norm,
            "deterministic_radial_percentile": float(
                (centroid_distances <= deterministic_distance).mean()
            ),
            "deterministic_geometric_median_cosine": float(
                deterministic[query_index] @ geometric
            ),
            "geometric_median_centroid_cosine": float(geometric @ center),
            "covariance_trace": total,
            "effective_rank": effective_rank,
            "pc1_variance_share": float(shares[0]) if len(shares) else 0.0,
            "top5_variance_share": float(shares[:5].sum()),
            "top10_variance_share": float(shares[:10].sum()),
            "bias_pc1_alignment": bias_pc1_alignment,
        }
        row.update(_distance_summary(centroid_distances, "centroid_distance"))
        row.update(_distance_summary(medoid_distances, "medoid_distance"))
        rows.append(row)
    return pd.DataFrame(rows)


def _spherical_kmeans(
    vectors: np.ndarray,
    clusters: int,
    seed: int,
    iterations: int = 25,
) -> tuple[np.ndarray, np.ndarray]:
    vectors = l2_normalize(vectors)
    rng = np.random.default_rng(seed)
    centers = vectors[rng.choice(len(vectors), size=clusters, replace=False)].copy()
    labels = np.zeros(len(vectors), dtype=np.int64)
    for _ in range(iterations):
        updated_labels = np.argmax(vectors @ centers.T, axis=1)
        if np.array_equal(updated_labels, labels) and _ > 0:
            break
        labels = updated_labels
        for cluster in range(clusters):
            members = vectors[labels == cluster]
            if len(members):
                centers[cluster] = l2_normalize(members.mean(axis=0))
            else:
                similarity = np.max(vectors @ centers.T, axis=1)
                centers[cluster] = vectors[int(np.argmin(similarity))]
    return labels, l2_normalize(centers)


def _silhouette_cosine(vectors: np.ndarray, labels: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return 0.0
    distances = 1.0 - l2_normalize(vectors) @ l2_normalize(vectors).T
    values = []
    for index, label in enumerate(labels):
        same = labels == label
        same[index] = False
        if not same.any():
            values.append(0.0)
            continue
        a = float(distances[index, same].mean())
        alternatives = [
            float(distances[index, labels == other].mean())
            for other in np.unique(labels)
            if other != label
        ]
        b = min(alternatives)
        denominator = max(a, b)
        values.append((b - a) / denominator if denominator > 0 else 0.0)
    return float(np.mean(values))


def _rand_agreement(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2:
        return 1.0
    upper = np.triu_indices(len(left), k=1)
    left_pairs = left[:, None] == left[None, :]
    right_pairs = right[:, None] == right[None, :]
    return float((left_pairs[upper] == right_pairs[upper]).mean())


def cluster_diagnostics(
    samples: np.ndarray,
    query_ids: list[str],
    cluster_counts: tuple[int, ...] = (2, 3, 4),
    bootstrap_replicates: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    normalized = l2_normalize(samples)
    rows: list[dict[str, float | int | str]] = []
    for query_index, query_id in enumerate(query_ids):
        vectors = normalized[:, query_index, :]
        for clusters in cluster_counts:
            if len(vectors) < clusters:
                continue
            labels, centers = _spherical_kmeans(
                vectors, clusters, seed + query_index * 101 + clusters
            )
            sizes = np.bincount(labels, minlength=clusters)
            center_similarity = centers @ centers.T
            upper = center_similarity[np.triu_indices(clusters, k=1)]
            stability = []
            rng = np.random.default_rng(seed + query_index * 1009 + clusters)
            for replicate in range(bootstrap_replicates):
                selected = rng.integers(0, len(vectors), size=len(vectors))
                _, bootstrap_centers = _spherical_kmeans(
                    vectors[selected],
                    clusters,
                    seed + replicate + query_index * 17 + clusters,
                )
                bootstrap_labels = np.argmax(
                    vectors @ bootstrap_centers.T, axis=1
                )
                stability.append(_rand_agreement(labels, bootstrap_labels))
            rows.append(
                {
                    "query_id": query_id,
                    "clusters": clusters,
                    "silhouette_cosine": _silhouette_cosine(vectors, labels),
                    "minimum_cluster_size": int(sizes.min()),
                    "maximum_cluster_fraction": float(sizes.max() / len(vectors)),
                    "mean_centroid_cosine_distance": float((1.0 - upper).mean()),
                    "bootstrap_rand_stability": float(np.mean(stability)),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "query_id",
            "clusters",
            "silhouette_cosine",
            "minimum_cluster_size",
            "maximum_cluster_fraction",
            "mean_centroid_cosine_distance",
            "bootstrap_rand_stability",
        ],
    )


def centroid_convergence(
    samples: np.ndarray,
    query_ids: list[str],
    subset_counts: tuple[int, ...] | None = None,
    bootstrap_replicates: int = 20,
    seed: int = 42,
    search: Callable[[np.ndarray], Rankings] | None = None,
    overlap_depth: int = 10,
) -> pd.DataFrame:
    normalized = l2_normalize(samples)
    full_centroid = l2_normalize(normalized.mean(axis=0))
    if subset_counts is None:
        subset_counts = tuple(
            count
            for count in (1, 2, 4, 8, 16, 32, 64, 128)
            if count <= len(normalized)
        )
    full_ranking = search(full_centroid) if search is not None else None
    rng = np.random.default_rng(seed)
    rows = []
    for count in subset_counts:
        distances = np.empty((bootstrap_replicates, len(query_ids)), dtype=np.float32)
        overlaps = np.full_like(distances, np.nan)
        replicate_centroids = []
        for replicate in range(bootstrap_replicates):
            selected = rng.choice(len(normalized), size=count, replace=False)
            centroid = l2_normalize(normalized[selected].mean(axis=0))
            replicate_centroids.append(centroid)
            distances[replicate] = 1.0 - np.sum(
                centroid * full_centroid, axis=1
            )
        if search is not None and full_ranking is not None:
            batched_ranking = search(np.concatenate(replicate_centroids, axis=0))
            for replicate in range(bootstrap_replicates):
                for query_index in range(len(query_ids)):
                    reference = set(
                        full_ranking.indices[query_index, :overlap_depth]
                    )
                    batched_index = replicate * len(query_ids) + query_index
                    candidate = set(
                        batched_ranking.indices[batched_index, :overlap_depth]
                    )
                    union = reference | candidate
                    overlaps[replicate, query_index] = (
                        len(reference & candidate) / len(union) if union else 1.0
                    )
        centroid_stack = np.stack(replicate_centroids)
        standard_error = np.sqrt(centroid_stack.var(axis=0).sum(axis=1))
        for query_index, query_id in enumerate(query_ids):
            rows.append(
                {
                    "query_id": query_id,
                    "subset_count": count,
                    "bootstrap_replicates": bootstrap_replicates,
                    "centroid_cosine_distance_mean": float(
                        distances[:, query_index].mean()
                    ),
                    "centroid_cosine_distance_p95": float(
                        np.quantile(distances[:, query_index], 0.95)
                    ),
                    "centroid_coordinate_standard_error": float(
                        standard_error[query_index]
                    ),
                    "topk_jaccard_to_full_mean": float(
                        np.nanmean(overlaps[:, query_index])
                    )
                    if search is not None
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _rank_biased_overlap(
    left: np.ndarray,
    right: np.ndarray,
    depth: int,
    persistence: float = 0.9,
) -> float:
    score = 0.0
    left_seen: set[int] = set()
    right_seen: set[int] = set()
    for rank in range(1, depth + 1):
        left_seen.add(int(left[rank - 1]))
        right_seen.add(int(right[rank - 1]))
        agreement = len(left_seen & right_seen) / rank
        score += (1.0 - persistence) * persistence ** (rank - 1) * agreement
    return score + persistence**depth * len(left_seen & right_seen) / depth


def retrieval_distribution(
    sample_rankings: list[Rankings],
    deterministic: Rankings,
    query_ids: list[str],
    document_ids: list[str],
    qrels: dict[str, dict[str, int]],
    depth: int = 10,
) -> pd.DataFrame:
    rows = []
    sample_count = len(sample_rankings)
    depth = min(depth, deterministic.indices.shape[1])
    for query_index, query_id in enumerate(query_ids):
        indices = np.stack(
            [ranking.indices[query_index] for ranking in sample_rankings]
        )
        scores = np.stack(
            [ranking.scores[query_index] for ranking in sample_rankings]
        )
        top1_counts = np.unique(indices[:, 0], return_counts=True)[1]
        top1_probabilities = top1_counts / sample_count
        top1_entropy = float(
            -(top1_probabilities * np.log2(top1_probabilities)).sum()
        )
        jaccard = []
        rbo = []
        for left in range(sample_count):
            for right in range(left + 1, sample_count):
                left_set = set(indices[left, :depth])
                right_set = set(indices[right, :depth])
                union = left_set | right_set
                jaccard.append(
                    len(left_set & right_set) / len(union) if union else 1.0
                )
                rbo.append(
                    _rank_biased_overlap(
                        indices[left], indices[right], depth=depth
                    )
                )
        relevant = {
            document_id
            for document_id, relevance in qrels.get(query_id, {}).items()
            if relevance > 0
        }
        relevant_indices = {
            index
            for index, document_id in enumerate(document_ids)
            if document_id in relevant
        }
        best_ranks = np.full(sample_count, indices.shape[1] + 1, dtype=np.float32)
        for sample in range(sample_count):
            positions = [
                rank + 1
                for rank, index in enumerate(indices[sample])
                if int(index) in relevant_indices
            ]
            if positions:
                best_ranks[sample] = min(positions)
        deterministic_positions = [
            rank + 1
            for rank, index in enumerate(deterministic.indices[query_index])
            if int(index) in relevant_indices
        ]
        deterministic_best = (
            min(deterministic_positions)
            if deterministic_positions
            else deterministic.indices.shape[1] + 1
        )
        union = np.unique(indices[:, :depth])
        rank_values = []
        score_values = []
        for document_index in union:
            document_ranks = np.full(sample_count, depth + 1, dtype=np.float32)
            document_scores = np.full(sample_count, np.nan, dtype=np.float32)
            for sample in range(sample_count):
                locations = np.flatnonzero(
                    indices[sample, :depth] == document_index
                )
                if len(locations):
                    location = int(locations[0])
                    document_ranks[sample] = location + 1
                    document_scores[sample] = scores[sample, location]
            rank_values.append(float(document_ranks.var()))
            score_values.append(float(np.nanvar(document_scores)))
        rows.append(
            {
                "query_id": query_id,
                "top1_entropy": top1_entropy,
                "unique_top1_documents": int(len(top1_counts)),
                "mean_pairwise_top10_jaccard": float(np.mean(jaccard))
                if jaccard
                else 1.0,
                "mean_pairwise_top10_rbo": float(np.mean(rbo)) if rbo else 1.0,
                "mean_top10_document_rank_variance": float(np.mean(rank_values)),
                "mean_top10_document_score_variance": float(
                    np.nanmean(score_values)
                ),
                "deterministic_best_relevant_rank": deterministic_best,
                "stochastic_best_relevant_rank_mean": float(best_ranks.mean()),
                "stochastic_best_relevant_rank_std": float(best_ranks.std()),
                "relevant_top1_probability": float((best_ranks <= 1).mean()),
                "relevant_top5_probability": float((best_ranks <= 5).mean()),
                "relevant_top10_probability": float((best_ranks <= 10).mean()),
                "relevant_enter_top1_probability": float((best_ranks <= 1).mean())
                if deterministic_best > 1
                else 0.0,
                "relevant_leave_top1_probability": float((best_ranks > 1).mean())
                if deterministic_best <= 1
                else 0.0,
                "relevant_enter_top5_probability": float((best_ranks <= 5).mean())
                if deterministic_best > 5
                else 0.0,
                "relevant_leave_top5_probability": float((best_ranks > 5).mean())
                if deterministic_best <= 5
                else 0.0,
                "relevant_enter_top10_probability": float(
                    (best_ranks <= 10).mean()
                )
                if deterministic_best > 10
                else 0.0,
                "relevant_leave_top10_probability": float(
                    (best_ranks > 10).mean()
                )
                if deterministic_best <= 10
                else 0.0,
                "mean_relevant_rank_movement": float(
                    best_ranks.mean() - deterministic_best
                ),
            }
        )
    return pd.DataFrame(rows)


def retrieval_document_distribution(
    sample_rankings: list[Rankings],
    deterministic: Rankings,
    query_ids: list[str],
    document_ids: list[str],
    qrels: dict[str, dict[str, int]],
    depth: int = 10,
) -> pd.DataFrame:
    """Per-document inclusion, score, and rank distributions at shallow depth."""
    depth = min(depth, deterministic.indices.shape[1])
    cutoffs = tuple(cutoff for cutoff in (1, 5, 10) if cutoff <= depth)
    rows = []
    for query_index, query_id in enumerate(query_ids):
        sample_indices = np.stack(
            [ranking.indices[query_index] for ranking in sample_rankings]
        )
        sample_scores = np.stack(
            [ranking.scores[query_index] for ranking in sample_rankings]
        )
        deterministic_top = set(deterministic.indices[query_index, :depth])
        stochastic_union = set(sample_indices[:, :depth].ravel())
        candidates = sorted(deterministic_top | stochastic_union)
        relevant = {
            document_id
            for document_id, relevance in qrels.get(query_id, {}).items()
            if relevance > 0
        }
        deterministic_lookup = {
            int(document_index): (rank + 1, float(deterministic.scores[query_index, rank]))
            for rank, document_index in enumerate(deterministic.indices[query_index])
        }
        sample_lookups = [
            {
                int(document_index): (rank + 1, float(sample_scores[sample, rank]))
                for rank, document_index in enumerate(sample_indices[sample])
            }
            for sample in range(len(sample_rankings))
        ]
        for document_index in candidates:
            ranks = np.array(
                [
                    lookup.get(int(document_index), (sample_indices.shape[1] + 1, np.nan))[
                        0
                    ]
                    for lookup in sample_lookups
                ],
                dtype=np.float32,
            )
            scores = np.array(
                [
                    lookup.get(int(document_index), (0, np.nan))[1]
                    for lookup in sample_lookups
                ],
                dtype=np.float32,
            )
            observed_scores = scores[~np.isnan(scores)]
            deterministic_rank, deterministic_score = deterministic_lookup.get(
                int(document_index),
                (deterministic.indices.shape[1] + 1, np.nan),
            )
            row: dict[str, object] = {
                "query_id": query_id,
                "document_id": document_ids[int(document_index)],
                "is_relevant": document_ids[int(document_index)] in relevant,
                "in_deterministic_top10": document_index in deterministic_top,
                "in_stochastic_top10_union": document_index in stochastic_union,
                "deterministic_rank": deterministic_rank,
                "deterministic_score": deterministic_score,
                "rank_mean": float(ranks.mean()),
                "rank_variance": float(ranks.var()),
                "rank_p10": float(np.quantile(ranks, 0.10)),
                "rank_median": float(np.median(ranks)),
                "rank_p90": float(np.quantile(ranks, 0.90)),
                "score_mean": float(observed_scores.mean())
                if len(observed_scores)
                else np.nan,
                "score_variance": float(observed_scores.var())
                if len(observed_scores)
                else np.nan,
                "score_p10": float(np.quantile(observed_scores, 0.10))
                if len(observed_scores)
                else np.nan,
                "score_median": float(np.median(observed_scores))
                if len(observed_scores)
                else np.nan,
                "score_p90": float(np.quantile(observed_scores, 0.90))
                if len(observed_scores)
                else np.nan,
            }
            for cutoff in cutoffs:
                row[f"inclusion_probability@{cutoff}"] = float(
                    (ranks <= cutoff).mean()
                )
            rows.append(row)
    return pd.DataFrame(rows)


def distribution_outcome_correlations(
    features: pd.DataFrame,
    per_query: pd.DataFrame,
    sample_count: int,
    baseline: str = "deterministic",
    metric: str = "ndcg@10",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    condition = per_query[per_query["sample_count"] == sample_count]
    baseline_values = condition[condition["method"] == baseline].set_index(
        "query_id"
    )[metric]
    numeric_features = [
        column
        for column in features.select_dtypes(include="number").columns
        if column != "sample_count"
    ]
    correlation_rows = []
    risk_rows = []
    oracle = condition[condition["method"] == "oracle_best_of_n"].set_index(
        "query_id"
    )[metric]
    opportunities = oracle > baseline_values if len(oracle) else pd.Series(False)
    for method in condition["method"].unique():
        if method == baseline:
            continue
        candidate = condition[condition["method"] == method].set_index(
            "query_id"
        )[metric]
        delta = candidate - baseline_values
        joined = features.set_index("query_id").join(
            delta.rename("delta"), how="inner"
        )
        for feature in numeric_features:
            valid = joined[[feature, "delta"]].dropna()
            correlation = (
                spearmanr(valid[feature], valid["delta"]).statistic
                if len(valid) > 2
                and valid[feature].nunique() > 1
                and valid["delta"].nunique() > 1
                else np.nan
            )
            correlation_rows.append(
                {
                    "sample_count": sample_count,
                    "method": method,
                    "feature": feature,
                    "queries": len(valid),
                    "spearman_delta": float(correlation),
                }
            )
        losses = delta[delta < 0]
        wins = delta[delta > 0]
        worst_count = max(1, math.ceil(len(delta) * 0.10))
        risk_rows.append(
            {
                "sample_count": sample_count,
                "method": method,
                "wins": int((delta > 0).sum()),
                "losses": int((delta < 0).sum()),
                "ties": int((delta == 0).sum()),
                "mean_winning_gain": float(wins.mean()) if len(wins) else 0.0,
                "mean_losing_magnitude": float(-losses.mean())
                if len(losses)
                else 0.0,
                "delta_p10": float(delta.quantile(0.10)),
                "delta_median": float(delta.median()),
                "worst_decile_mean_delta": float(
                    delta.nsmallest(worst_count).mean()
                ),
                "oracle_opportunity_queries": int(opportunities.sum()),
                "oracle_opportunity_coverage": float(
                    (delta[opportunities] > 0).mean()
                )
                if opportunities.any()
                else 0.0,
            }
        )
    return pd.DataFrame(correlation_rows), pd.DataFrame(risk_rows)


def analyze_saved_run(
    run_dir: Path,
    sample_counts: tuple[int, ...] | None = None,
    bootstrap_replicates: int = 5,
    ranking_chunk_size: int = 32,
) -> list[Path]:
    """Generate distribution diagnostics using only persisted run artifacts."""
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = manifest["config"]
    available_counts = tuple(config["experiment"]["sample_counts"])
    selected_counts = sample_counts or available_counts
    unknown = set(selected_counts) - set(available_counts)
    if unknown:
        raise ValueError(f"Sample counts not present in run: {sorted(unknown)}")

    query_ids = _read_jsonl(run_dir / "embeddings/queries_deterministic/ids.jsonl")
    document_ids = _read_jsonl(run_dir / "embeddings/documents/ids.jsonl")
    qrels = json.loads((run_dir / "qrels.json").read_text(encoding="utf-8"))
    deterministic_queries = np.load(
        run_dir / "embeddings/queries_deterministic/sample_000.npy",
        mmap_mode="r",
    )
    documents = np.load(
        run_dir / "embeddings/documents/sample_000.npy",
        mmap_mode="r",
    )
    retriever = DenseRetriever(
        documents,
        config["experiment"]["retrieval_k"],
        backend=config["experiment"]["retrieval_backend"],
        query_batch_size=config["experiment"]["retrieval_query_batch_size"],
        corpus_batch_size=config["experiment"]["retrieval_corpus_batch_size"],
    )
    deterministic_ranking = retriever.search(deterministic_queries)
    per_query = pd.read_parquet(run_dir / "metrics/per_query.parquet")
    dataset_name = config["dataset"]["name"]
    probability = config["model"]["dropout_probability"]
    probability_label = "native" if probability is None else f"p{probability:.2f}"
    condition = f"{config['model']['dropout_scope']}_{probability_label}"
    embedding_parts = []
    cluster_parts = []
    convergence_parts = []
    retrieval_parts = []
    retrieval_document_parts = []
    correlation_parts = []
    risk_parts = []
    for sample_count in selected_counts:
        sample_directory = (
            run_dir / f"embeddings/queries_stochastic/n_{sample_count:03d}"
        )
        samples = np.stack(
            [
                np.load(sample_directory / f"sample_{sample:03d}.npy")
                for sample in range(sample_count)
            ]
        )
        ranking_directory = run_dir / f"rankings/samples/n_{sample_count:03d}"
        retrieval_chunks = []
        retrieval_document_chunks = []
        for start in range(0, len(query_ids), ranking_chunk_size):
            stop = min(start + ranking_chunk_size, len(query_ids))
            sample_rankings = []
            for sample in range(sample_count):
                with np.load(
                    ranking_directory / f"sample_{sample:03d}.npz"
                ) as data:
                    sample_rankings.append(
                        Rankings(
                            data["indices"][start:stop].copy(),
                            data["scores"][start:stop].copy(),
                        )
                    )
            deterministic_chunk = Rankings(
                deterministic_ranking.indices[start:stop],
                deterministic_ranking.scores[start:stop],
            )
            retrieval_chunks.append(
                retrieval_distribution(
                    sample_rankings,
                    deterministic_chunk,
                    query_ids[start:stop],
                    document_ids,
                    qrels,
                )
            )
            retrieval_document_chunks.append(
                retrieval_document_distribution(
                    sample_rankings,
                    deterministic_chunk,
                    query_ids[start:stop],
                    document_ids,
                    qrels,
                )
            )
        retrieval_frame = pd.concat(retrieval_chunks, ignore_index=True)
        retrieval_document_frame = pd.concat(
            retrieval_document_chunks,
            ignore_index=True,
        )
        frames = (
            embedding_distribution(samples, deterministic_queries, query_ids),
            cluster_diagnostics(
                samples,
                query_ids,
                bootstrap_replicates=bootstrap_replicates,
                seed=config["experiment"]["seed"],
            ),
            centroid_convergence(
                samples,
                query_ids,
                bootstrap_replicates=bootstrap_replicates,
                seed=config["experiment"]["seed"],
                search=retriever.search,
            ),
            retrieval_frame,
            retrieval_document_frame,
        )
        for frame, destination in zip(
            frames,
            (
                embedding_parts,
                cluster_parts,
                convergence_parts,
                retrieval_parts,
                retrieval_document_parts,
            ),
            strict=True,
        ):
            frame.insert(1, "sample_count", sample_count)
            frame.insert(1, "dropout_condition", condition)
            frame.insert(1, "dataset", dataset_name)
            destination.append(frame)
        features = frames[0].merge(
            frames[3],
            on=["dataset", "dropout_condition", "sample_count", "query_id"],
            suffixes=("", "_retrieval"),
        )
        metric = "ndcg@10" if "ndcg@10" in per_query else next(
            column for column in per_query if column.startswith("ndcg@")
        )
        correlations, risks = distribution_outcome_correlations(
            features,
            per_query,
            sample_count,
            metric=metric,
        )
        for frame in (correlations, risks):
            frame.insert(0, "dropout_condition", condition)
            frame.insert(0, "dataset", dataset_name)
        correlation_parts.append(correlations)
        risk_parts.append(risks)

    analyses = run_dir / "analyses"
    analyses.mkdir(exist_ok=True)
    outputs = []
    for parts, name in (
        (embedding_parts, "embedding_distribution"),
        (cluster_parts, "cluster_diagnostics"),
        (convergence_parts, "centroid_convergence"),
        (retrieval_parts, "retrieval_distribution"),
        (retrieval_document_parts, "retrieval_document_distribution"),
        (correlation_parts, "distribution_outcome_correlations"),
        (risk_parts, "distribution_outcome_risk"),
    ):
        path = analyses / f"{name}.parquet"
        pd.concat(parts, ignore_index=True).to_parquet(path, index=False)
        outputs.append(path)
        relative = path.relative_to(run_dir).as_posix()
        manifest.setdefault("artifacts", {})[relative] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    manifest["distribution_analysis"] = {
        "sample_counts": list(selected_counts),
        "normalization": "per-query unit-normalized embeddings",
        "cluster_counts": [2, 3, 4],
        "bootstrap_replicates": bootstrap_replicates,
        "ranking_chunk_size": ranking_chunk_size,
        "seed": config["experiment"]["seed"],
        "inference_reused": True,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return outputs


def _read_jsonl(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
