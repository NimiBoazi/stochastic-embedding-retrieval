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


def anchored_centroid(
    deterministic: np.ndarray,
    samples: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Shrink the stochastic centroid toward the deterministic embedding."""
    if not 0 <= alpha <= 1:
        raise ValueError("anchor alpha must be in [0, 1]")
    deterministic_normalized = l2_normalize(deterministic)
    stochastic_centroid = mean_embedding(samples)
    return l2_normalize(
        (1.0 - alpha) * deterministic_normalized + alpha * stochastic_centroid
    )


def gated_ranking(
    deterministic: Rankings,
    alternative: Rankings,
    use_alternative: np.ndarray,
) -> Rankings:
    """Select one complete ranking per query using a label-free boolean gate."""
    mask = np.asarray(use_alternative, dtype=bool)
    if mask.shape != (len(deterministic.indices),):
        raise ValueError("Ranking gate must contain one boolean per query")
    indices = deterministic.indices.copy()
    scores = deterministic.scores.copy()
    indices[mask] = alternative.indices[mask]
    scores[mask] = alternative.scores[mask]
    return Rankings(indices, scores)


def deterministic_score_margin(rankings: Rankings, rank: int) -> np.ndarray:
    """Score difference between documents at `rank` and `rank + 1`."""
    if rank < 1 or rankings.scores.shape[1] <= rank:
        raise ValueError("Score margin rank must have a following document")
    return rankings.scores[:, rank - 1] - rankings.scores[:, rank]


def ranking_disagreement(
    rankings: list[Rankings],
    depth: int = 10,
) -> np.ndarray:
    """Mean pairwise Jaccard distance among sampled top-k sets per query."""
    if not rankings:
        raise ValueError("At least one ranking is required")
    if depth < 1:
        raise ValueError("disagreement depth must be positive")
    query_count = rankings[0].indices.shape[0]
    if len(rankings) == 1:
        return np.zeros(query_count, dtype=np.float32)
    result = np.zeros(query_count, dtype=np.float32)
    pairs = 0
    for left in range(len(rankings)):
        for right in range(left + 1, len(rankings)):
            for query_index in range(query_count):
                left_set = set(
                    int(index)
                    for index in rankings[left].indices[query_index, :depth]
                    if index >= 0
                )
                right_set = set(
                    int(index)
                    for index in rankings[right].indices[query_index, :depth]
                    if index >= 0
                )
                union = left_set | right_set
                similarity = (
                    len(left_set & right_set) / len(union) if union else 1.0
                )
                result[query_index] += 1.0 - similarity
            pairs += 1
    return result / pairs


def quartile_gate_masks(
    deterministic_margins: np.ndarray,
    disagreement: np.ndarray,
    sample_count: int,
) -> tuple[np.ndarray, float, np.ndarray, float]:
    """Return pre-specified bottom-margin and top-disagreement quartile gates."""
    margin_threshold = float(np.nanquantile(deterministic_margins, 0.25))
    margin_mask = deterministic_margins <= margin_threshold
    disagreement_threshold = float(np.quantile(disagreement, 0.75))
    disagreement_mask = (
        disagreement >= disagreement_threshold
        if sample_count > 1
        else np.zeros(len(disagreement), dtype=bool)
    )
    return (
        margin_mask,
        margin_threshold,
        disagreement_mask,
        disagreement_threshold,
    )


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


def majority_vote(
    rankings: list[Rankings],
    k: int,
    depth: int = 100,
    offset: int = 60,
) -> Rankings:
    """Top-`depth` voting with reciprocal-rank tie breaking over the full lists.

    The primary signal is the integer number of sampled rankings that place a
    document within `depth`. Ties (including documents with zero votes) are
    broken by the reciprocal-rank-fusion sum over the full-depth lists, so the
    fused ranking stays full length and deep cutoffs remain comparable with the
    other methods. Remaining exact ties fall back to the document index.
    """
    if not rankings:
        raise ValueError("At least one ranking is required")
    if depth < 1:
        raise ValueError("majority_vote_depth must be positive")
    query_count = rankings[0].indices.shape[0]
    output_indices = np.full((query_count, k), -1, dtype=np.int64)
    output_scores = np.full((query_count, k), -np.inf, dtype=np.float32)
    for query_index in range(query_count):
        votes: dict[int, int] = {}
        fused: dict[int, float] = {}
        for ranking in rankings:
            for rank, document_index in enumerate(
                ranking.indices[query_index], start=1
            ):
                document_index = int(document_index)
                if document_index < 0:
                    continue
                if rank <= depth:
                    votes[document_index] = votes.get(document_index, 0) + 1
                fused[document_index] = fused.get(document_index, 0.0) + 1.0 / (
                    offset + rank
                )
        ordered = sorted(
            fused.items(),
            key=lambda item: (-votes.get(item[0], 0), -item[1], item[0]),
        )[:k]
        # RRF sums are far below 1000, so the tiers cannot collide in the score.
        output_indices[query_index, : len(ordered)] = [item[0] for item in ordered]
        output_scores[query_index, : len(ordered)] = [
            votes.get(item[0], 0) * 1000.0 + item[1] for item in ordered
        ]
    return Rankings(output_indices, output_scores)


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

    def mean_minus_std(scores: np.ndarray) -> np.ndarray:
        # Bessel-corrected sample standard deviation; undefined for one sample.
        if len(scores) < 2:
            return scores.mean(axis=0)
        return scores.mean(axis=0) - penalty * scores.std(axis=0, ddof=1)

    return _sample_score_rerank(
        samples,
        corpus,
        sample_rankings,
        k,
        score_function=mean_minus_std,
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
