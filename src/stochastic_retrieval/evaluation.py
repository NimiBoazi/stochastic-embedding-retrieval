from __future__ import annotations

import math

import numpy as np
import pandas as pd

from stochastic_retrieval.retrieval import Rankings


def ranking_frame(
    method: str,
    rankings: Rankings,
    query_ids: list[str],
    document_ids: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for query_index, query_id in enumerate(query_ids):
        for rank, (document_index, score) in enumerate(
            zip(
                rankings.indices[query_index],
                rankings.scores[query_index],
                strict=True,
            ),
            start=1,
        ):
            if document_index < 0:
                continue
            rows.append(
                {
                    "method": method,
                    "query_id": query_id,
                    "doc_id": document_ids[int(document_index)],
                    "rank": rank,
                    "score": float(score),
                }
            )
    return pd.DataFrame(rows)


def evaluate_rankings(
    method: str,
    rankings: Rankings,
    query_ids: list[str],
    document_ids: list[str],
    qrels: dict[str, dict[str, int]],
    cutoffs: tuple[int, ...],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for query_index, query_id in enumerate(query_ids):
        retrieved = [
            document_ids[int(index)]
            for index in rankings.indices[query_index]
            if index >= 0
        ]
        relevance = qrels.get(query_id, {})
        row: dict[str, object] = {"method": method, "query_id": query_id}
        for cutoff in cutoffs:
            row[f"ndcg@{cutoff}"] = ndcg(retrieved, relevance, cutoff)
            row[f"recall@{cutoff}"] = recall(retrieved, relevance, cutoff)
            row[f"map@{cutoff}"] = average_precision(retrieved, relevance, cutoff)
            row[f"mrr@{cutoff}"] = reciprocal_rank(retrieved, relevance, cutoff)
        rows.append(row)
    return pd.DataFrame(rows)


def ndcg(retrieved: list[str], relevance: dict[str, int], k: int) -> float:
    gains = [relevance.get(document_id, 0) for document_id in retrieved[:k]]
    ideal = sorted((value for value in relevance.values() if value > 0), reverse=True)[:k]
    denominator = _dcg(ideal)
    return _dcg(gains) / denominator if denominator else 0.0


def _dcg(relevance: list[int]) -> float:
    return sum(
        (2**grade - 1) / math.log2(rank + 1)
        for rank, grade in enumerate(relevance, start=1)
    )


def recall(retrieved: list[str], relevance: dict[str, int], k: int) -> float:
    relevant = {document_id for document_id, value in relevance.items() if value > 0}
    if not relevant:
        return 0.0
    return len(relevant.intersection(retrieved[:k])) / len(relevant)


def average_precision(
    retrieved: list[str], relevance: dict[str, int], k: int
) -> float:
    relevant = {document_id for document_id, value in relevance.items() if value > 0}
    if not relevant:
        return 0.0
    hits = 0
    score = 0.0
    for rank, document_id in enumerate(retrieved[:k], start=1):
        if document_id in relevant:
            hits += 1
            score += hits / rank
    return score / min(len(relevant), k)


def reciprocal_rank(
    retrieved: list[str], relevance: dict[str, int], k: int
) -> float:
    for rank, document_id in enumerate(retrieved[:k], start=1):
        if relevance.get(document_id, 0) > 0:
            return 1.0 / rank
    return 0.0


def oracle_best_of_n(
    rankings: list[Rankings],
    query_ids: list[str],
    document_ids: list[str],
    qrels: dict[str, dict[str, int]],
    selection_cutoff: int = 10,
) -> tuple[Rankings, pd.DataFrame]:
    """Select the highest-nDCG sample per query; this intentionally uses labels."""
    if not rankings:
        raise ValueError("At least one sampled ranking is required")
    query_count, depth = rankings[0].indices.shape
    indices = np.empty((query_count, depth), dtype=np.int64)
    scores = np.empty((query_count, depth), dtype=np.float32)
    selections: list[dict[str, object]] = []

    for query_index, query_id in enumerate(query_ids):
        utilities = []
        for ranking in rankings:
            retrieved = [
                document_ids[int(index)]
                for index in ranking.indices[query_index]
                if index >= 0
            ]
            utilities.append(ndcg(retrieved, qrels.get(query_id, {}), selection_cutoff))
        selected = int(np.argmax(utilities))
        indices[query_index] = rankings[selected].indices[query_index]
        scores[query_index] = rankings[selected].scores[query_index]
        selections.append(
            {
                "query_id": query_id,
                "selected_sample": selected,
                f"selection_ndcg@{selection_cutoff}": utilities[selected],
            }
        )
    return Rankings(indices, scores), pd.DataFrame(selections)


def summarize(per_query: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        column
        for column in per_query.columns
        if column not in {"method", "query_id"}
    ]
    return (
        per_query.groupby("method", sort=False)[metric_columns]
        .mean()
        .reset_index()
    )


def paired_bootstrap_comparisons(
    per_query: pd.DataFrame,
    baseline: str,
    metric: str,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    baseline_values = (
        per_query.loc[per_query["method"] == baseline, ["query_id", metric]]
        .set_index("query_id")[metric]
    )
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(seed)
    for method in per_query["method"].unique():
        if method == baseline:
            continue
        candidate = (
            per_query.loc[per_query["method"] == method, ["query_id", metric]]
            .set_index("query_id")[metric]
        )
        aligned = pd.concat(
            [baseline_values.rename("baseline"), candidate.rename("candidate")],
            axis=1,
            join="inner",
        ).dropna()
        differences = (aligned["candidate"] - aligned["baseline"]).to_numpy()
        sample_indices = rng.integers(
            0, len(differences), size=(replicates, len(differences))
        )
        bootstrap = differences[sample_indices].mean(axis=1)
        rows.append(
            {
                "baseline": baseline,
                "method": method,
                "metric": metric,
                "queries": len(differences),
                "mean_delta": float(differences.mean()),
                "ci_low": float(np.quantile(bootstrap, 0.025)),
                "ci_high": float(np.quantile(bootstrap, 0.975)),
                "bootstrap_probability_delta_le_zero": float(
                    (bootstrap <= 0).mean()
                ),
            }
        )
    return pd.DataFrame(rows)
