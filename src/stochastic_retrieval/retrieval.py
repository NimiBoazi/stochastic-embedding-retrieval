from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm import tqdm


@dataclass(frozen=True)
class Rankings:
    indices: np.ndarray
    scores: np.ndarray

    def __post_init__(self) -> None:
        if self.indices.shape != self.scores.shape:
            raise ValueError("Ranking indices and scores must have identical shapes")


def l2_normalize(array: np.ndarray, axis: int = -1) -> np.ndarray:
    norms = np.linalg.norm(array, axis=axis, keepdims=True)
    return array / np.maximum(norms, np.finfo(np.float32).eps)


def exact_search(
    queries: np.ndarray,
    corpus: np.ndarray,
    k: int,
    query_batch_size: int = 128,
    corpus_batch_size: int = 50_000,
) -> Rankings:
    """Exact blockwise inner-product search with bounded score-matrix memory."""
    k = min(k, len(corpus))
    all_indices = np.empty((len(queries), k), dtype=np.int64)
    all_scores = np.empty((len(queries), k), dtype=np.float32)

    for query_start in tqdm(
        range(0, len(queries), query_batch_size),
        desc="Exact retrieval",
        unit="query-batches",
    ):
        query_stop = min(query_start + query_batch_size, len(queries))
        query_block = np.asarray(queries[query_start:query_stop], dtype=np.float32)
        best_scores = np.full((len(query_block), k), -np.inf, dtype=np.float32)
        best_indices = np.full((len(query_block), k), -1, dtype=np.int64)

        for corpus_start in range(0, len(corpus), corpus_batch_size):
            corpus_stop = min(corpus_start + corpus_batch_size, len(corpus))
            corpus_block = np.asarray(corpus[corpus_start:corpus_stop], dtype=np.float32)
            scores = query_block @ corpus_block.T
            indices = np.broadcast_to(
                np.arange(corpus_start, corpus_stop, dtype=np.int64),
                scores.shape,
            )
            candidate_scores = np.concatenate((best_scores, scores), axis=1)
            candidate_indices = np.concatenate((best_indices, indices), axis=1)
            selected = np.argpartition(candidate_scores, -k, axis=1)[:, -k:]
            best_scores = np.take_along_axis(candidate_scores, selected, axis=1)
            best_indices = np.take_along_axis(candidate_indices, selected, axis=1)

        order = np.argsort(-best_scores, axis=1, kind="stable")
        all_scores[query_start:query_stop] = np.take_along_axis(best_scores, order, axis=1)
        all_indices[query_start:query_stop] = np.take_along_axis(
            best_indices, order, axis=1
        )
    return Rankings(indices=all_indices, scores=all_scores)


def mean_embedding(samples: np.ndarray) -> np.ndarray:
    return l2_normalize(samples.mean(axis=0))


def medoid_embedding(samples: np.ndarray) -> np.ndarray:
    """Choose the sample with highest average cosine agreement for every query."""
    normalized = l2_normalize(samples)
    sample_count, query_count, dimension = normalized.shape
    result = np.empty((query_count, dimension), dtype=np.float32)
    for query_index in range(query_count):
        vectors = normalized[:, query_index, :]
        agreement = vectors @ vectors.T
        result[query_index] = vectors[np.argmax(agreement.mean(axis=1))]
    return result


def reciprocal_rank_fusion(rankings: list[Rankings], k: int, offset: int = 60) -> Rankings:
    return _rank_fusion(rankings, k, lambda rank: 1.0 / (offset + rank))


def majority_vote(rankings: list[Rankings], k: int) -> Rankings:
    depth = rankings[0].indices.shape[1]
    return _rank_fusion(rankings, k, lambda rank: 1.0 + (depth - rank) * 1e-9)


def _rank_fusion(
    rankings: list[Rankings],
    k: int,
    rank_score: object,
) -> Rankings:
    if not rankings:
        raise ValueError("At least one ranking is required")
    query_count = rankings[0].indices.shape[0]
    output_indices = np.full((query_count, k), -1, dtype=np.int64)
    output_scores = np.full((query_count, k), -np.inf, dtype=np.float32)
    for query_index in range(query_count):
        fused: dict[int, float] = {}
        for ranking in rankings:
            for rank, document_index in enumerate(
                ranking.indices[query_index], start=1
            ):
                document_index = int(document_index)
                fused[document_index] = fused.get(document_index, 0.0) + rank_score(rank)  # type: ignore[operator]
        ordered = sorted(fused.items(), key=lambda item: (-item[1], item[0]))[:k]
        output_indices[query_index, : len(ordered)] = [item[0] for item in ordered]
        output_scores[query_index, : len(ordered)] = [item[1] for item in ordered]
    return Rankings(output_indices, output_scores)


def embedding_diversity(samples: np.ndarray, query_ids: list[str]) -> pd.DataFrame:
    normalized = l2_normalize(samples)
    centroid = l2_normalize(normalized.mean(axis=0))
    rows: list[dict[str, float | str]] = []
    for query_index, query_id in enumerate(query_ids):
        vectors = normalized[:, query_index, :]
        similarity = vectors @ vectors.T
        upper = similarity[np.triu_indices(len(vectors), k=1)]
        centered = vectors - vectors.mean(axis=0, keepdims=True)
        singular_values = np.linalg.svd(centered, compute_uv=False)
        eigenvalues = singular_values**2 / max(len(vectors) - 1, 1)
        total = float(eigenvalues.sum())
        effective_rank = (
            float(total**2 / np.square(eigenvalues).sum()) if total > 0 else 0.0
        )
        rows.append(
            {
                "query_id": query_id,
                "mean_pairwise_cosine_distance": float((1.0 - upper).mean())
                if len(upper)
                else 0.0,
                "centroid_agreement": float((vectors @ centroid[query_index]).mean()),
                "covariance_trace": total,
                "effective_rank": effective_rank,
            }
        )
    return pd.DataFrame(rows)
