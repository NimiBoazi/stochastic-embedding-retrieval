from __future__ import annotations

import importlib.util
from collections.abc import Callable
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


class DenseRetriever:
    """Reusable exact-search backend; FAISS indexes the corpus only once."""

    def __init__(
        self,
        corpus: np.ndarray,
        k: int,
        backend: str = "auto",
        query_batch_size: int = 128,
        corpus_batch_size: int = 50_000,
    ) -> None:
        valid_backends = {"auto", "numpy", "faiss-cpu", "faiss-gpu"}
        if backend not in valid_backends:
            raise ValueError(
                f"retrieval_backend must be one of {sorted(valid_backends)}"
            )
        self.corpus = corpus
        self.k = min(k, len(corpus))
        self.query_batch_size = query_batch_size
        self.corpus_batch_size = corpus_batch_size
        self.backend = (
            "faiss-cpu"
            if backend == "auto" and importlib.util.find_spec("faiss") is not None
            else "numpy"
            if backend == "auto"
            else backend
        )
        self.index = self._build_faiss_index() if self.backend.startswith("faiss") else None

    def search(self, queries: np.ndarray) -> Rankings:
        if self.backend == "numpy":
            return exact_search(
                queries,
                self.corpus,
                self.k,
                query_batch_size=self.query_batch_size,
                corpus_batch_size=self.corpus_batch_size,
            )
        return self._search_faiss(queries)

    def _build_faiss_index(self) -> object:
        try:
            import faiss
        except ImportError as exc:
            raise ImportError(
                "FAISS retrieval requested but FAISS is not installed. "
                "Install the cpu-index or gpu project extra."
            ) from exc

        index = faiss.IndexFlatIP(int(self.corpus.shape[1]))
        if self.backend == "faiss-gpu":
            if not hasattr(faiss, "get_num_gpus") or faiss.get_num_gpus() < 1:
                raise RuntimeError(
                    "faiss-gpu was requested but no FAISS GPU is available"
                )
            index = faiss.index_cpu_to_all_gpus(index)
        for start in tqdm(
            range(0, len(self.corpus), self.corpus_batch_size),
            desc="Building FAISS index",
            unit="corpus-batches",
        ):
            block = np.ascontiguousarray(
                self.corpus[start : start + self.corpus_batch_size],
                dtype=np.float32,
            )
            index.add(block)
        return index

    def _search_faiss(self, queries: np.ndarray) -> Rankings:
        all_scores = np.empty((len(queries), self.k), dtype=np.float32)
        all_indices = np.empty((len(queries), self.k), dtype=np.int64)
        for start in tqdm(
            range(0, len(queries), self.query_batch_size),
            desc="FAISS retrieval",
            unit="query-batches",
        ):
            stop = min(start + self.query_batch_size, len(queries))
            block = np.ascontiguousarray(queries[start:stop], dtype=np.float32)
            scores, indices = self.index.search(block, self.k)
            all_scores[start:stop] = scores
            all_indices[start:stop] = indices
        return Rankings(indices=all_indices, scores=all_scores)


def search_embeddings(
    queries: np.ndarray,
    corpus: np.ndarray,
    k: int,
    backend: str = "auto",
    query_batch_size: int = 128,
    corpus_batch_size: int = 50_000,
) -> Rankings:
    return DenseRetriever(
        corpus,
        k,
        backend,
        query_batch_size,
        corpus_batch_size,
    ).search(queries)


def mean_embedding(samples: np.ndarray) -> np.ndarray:
    return l2_normalize(samples.mean(axis=0))


def medoid_embedding(samples: np.ndarray) -> np.ndarray:
    """Choose the sample with highest average cosine agreement for every query."""
    normalized = l2_normalize(samples)
    _, query_count, dimension = normalized.shape
    result = np.empty((query_count, dimension), dtype=np.float32)
    indices = embedding_medoid_indices(normalized)
    for query_index in range(query_count):
        result[query_index] = normalized[indices[query_index], query_index]
    return result


def embedding_medoid_indices(samples: np.ndarray) -> np.ndarray:
    normalized = l2_normalize(samples)
    indices = np.empty(normalized.shape[1], dtype=np.int64)
    for query_index in range(normalized.shape[1]):
        vectors = normalized[:, query_index, :]
        agreement = vectors @ vectors.T
        indices[query_index] = np.argmax(agreement.mean(axis=1))
    return indices


def trimmed_centroid(samples: np.ndarray, trim_fraction: float = 0.20) -> np.ndarray:
    centroid, _ = _trimmed_centroid_with_mask(samples, trim_fraction)
    return centroid


def trimmed_centroid_diagnostics(
    samples: np.ndarray,
    query_ids: list[str],
    trim_fraction: float = 0.20,
) -> pd.DataFrame:
    normalized = l2_normalize(samples)
    medoid_indices = embedding_medoid_indices(normalized)
    _, retained = _trimmed_centroid_with_mask(normalized, trim_fraction)
    rows: list[dict[str, object]] = []
    for query_index, query_id in enumerate(query_ids):
        medoid = normalized[medoid_indices[query_index], query_index]
        similarities = normalized[:, query_index, :] @ medoid
        for sample in range(len(normalized)):
            rows.append(
                {
                    "query_id": query_id,
                    "sample": sample,
                    "retained": bool(retained[sample, query_index]),
                    "is_medoid": sample == medoid_indices[query_index],
                    "cosine_distance_to_medoid": float(1.0 - similarities[sample]),
                }
            )
    return pd.DataFrame(rows)


def _trimmed_centroid_with_mask(
    samples: np.ndarray,
    trim_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0 <= trim_fraction < 0.5:
        raise ValueError("trim_fraction must be in [0, 0.5)")
    normalized = l2_normalize(samples)
    sample_count, query_count, dimension = normalized.shape
    keep_count = max(1, int(np.ceil(sample_count * (1.0 - trim_fraction))))
    medoid_indices = embedding_medoid_indices(normalized)
    retained = np.zeros((sample_count, query_count), dtype=bool)
    result = np.empty((query_count, dimension), dtype=np.float32)
    for query_index in range(query_count):
        vectors = normalized[:, query_index, :]
        medoid = vectors[medoid_indices[query_index]]
        similarities = vectors @ medoid
        selected = np.argpartition(-similarities, keep_count - 1)[:keep_count]
        retained[selected, query_index] = True
        result[query_index] = l2_normalize(vectors[selected].mean(axis=0))
    return result, retained


def reciprocal_rank_fusion(rankings: list[Rankings], k: int, offset: int = 60) -> Rankings:
    return _rank_fusion(rankings, k, lambda rank: 1.0 / (offset + rank))


def majority_vote(rankings: list[Rankings], k: int) -> Rankings:
    depth = rankings[0].indices.shape[1]
    return _rank_fusion(rankings, k, lambda rank: 1.0 + (depth - rank) * 1e-9)


def ranking_medoid(rankings: list[Rankings], depth: int = 100) -> Rankings:
    if not rankings:
        raise ValueError("At least one ranking is required")
    if depth < 1:
        raise ValueError("ranking_medoid_depth must be positive")
    query_count, output_depth = rankings[0].indices.shape
    output_indices = np.empty((query_count, output_depth), dtype=np.int64)
    output_scores = np.empty((query_count, output_depth), dtype=np.float32)
    for query_index in range(query_count):
        sets = [
            set(int(index) for index in ranking.indices[query_index, :depth] if index >= 0)
            for ranking in rankings
        ]
        agreement = np.zeros(len(rankings), dtype=np.float64)
        for left in range(len(rankings)):
            for right in range(left + 1, len(rankings)):
                union = sets[left] | sets[right]
                similarity = len(sets[left] & sets[right]) / len(union) if union else 1.0
                agreement[left] += similarity
                agreement[right] += similarity
        selected = int(np.argmax(agreement))
        output_indices[query_index] = rankings[selected].indices[query_index]
        output_scores[query_index] = rankings[selected].scores[query_index]
    return Rankings(output_indices, output_scores)


def maximum_score_rerank(
    samples: np.ndarray,
    corpus: np.ndarray,
    sample_rankings: list[Rankings],
    k: int,
) -> Rankings:
    return _sample_score_rerank(
        samples,
        corpus,
        sample_rankings,
        k,
        score_function=lambda scores: scores.max(axis=0),
    )


def variance_penalized_rerank(
    samples: np.ndarray,
    corpus: np.ndarray,
    sample_rankings: list[Rankings],
    k: int,
    penalty: float = 1.0,
) -> Rankings:
    if penalty < 0:
        raise ValueError("variance_penalty_lambda must be non-negative")
    return _sample_score_rerank(
        samples,
        corpus,
        sample_rankings,
        k,
        score_function=lambda scores: scores.mean(axis=0)
        - penalty * scores.std(axis=0),
    )


def _sample_score_rerank(
    samples: np.ndarray,
    corpus: np.ndarray,
    sample_rankings: list[Rankings],
    k: int,
    score_function: Callable[[np.ndarray], np.ndarray],
) -> Rankings:
    if not sample_rankings:
        raise ValueError("At least one sampled ranking is required")
    query_count = samples.shape[1]
    output_indices = np.full((query_count, k), -1, dtype=np.int64)
    output_scores = np.full((query_count, k), -np.inf, dtype=np.float32)
    for query_index in range(query_count):
        candidates = np.unique(
            np.concatenate(
                [
                    ranking.indices[query_index][ranking.indices[query_index] >= 0]
                    for ranking in sample_rankings
                ]
            )
        )
        documents = np.asarray(corpus[candidates], dtype=np.float32)
        scores = samples[:, query_index, :] @ documents.T
        aggregated = score_function(scores)
        order = np.lexsort((candidates, -aggregated))[:k]
        output_indices[query_index, : len(order)] = candidates[order]
        output_scores[query_index, : len(order)] = aggregated[order]
    return Rankings(output_indices, output_scores)


def _rank_fusion(
    rankings: list[Rankings],
    k: int,
    rank_score: Callable[[int], float],
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
                fused[document_index] = fused.get(document_index, 0.0) + rank_score(rank)
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
