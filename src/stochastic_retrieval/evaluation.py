from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytrec_eval

from stochastic_retrieval.retrieval import Rankings


def ranking_frame(
    method: str,
    rankings: Rankings,
    query_ids: list[str],
    document_ids: list[str],
    sample_count: int | None = None,
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
            row: dict[str, object] = {
                "method": method,
                "query_id": query_id,
                "doc_id": document_ids[int(document_index)],
                "rank": rank,
                "score": float(score),
            }
            if sample_count is not None:
                row["sample_count"] = sample_count
            rows.append(row)
    return pd.DataFrame(rows)


def iter_ranking_frames(
    method: str,
    rankings: Rankings,
    query_ids: list[str],
    document_ids: list[str],
    query_batch_size: int = 256,
    sample_count: int | None = None,
) -> Iterator[pd.DataFrame]:
    for start in range(0, len(query_ids), query_batch_size):
        stop = min(start + query_batch_size, len(query_ids))
        yield ranking_frame(
            method,
            Rankings(
                indices=rankings.indices[start:stop],
                scores=rankings.scores[start:stop],
            ),
            query_ids[start:stop],
            document_ids,
            sample_count=sample_count,
        )


def evaluate_rankings(
    method: str,
    rankings: Rankings,
    query_ids: list[str],
    document_ids: list[str],
    qrels: dict[str, dict[str, int]],
    cutoffs: tuple[int, ...],
    success_cutoffs: tuple[int, ...] = (),
) -> pd.DataFrame:
    """Evaluate one ranking with pytrec_eval (trec_eval semantics).

    nDCG uses linear gain and MAP@k normalizes by the total number of relevant
    documents, matching the official BEIR evaluation. success@k (any relevant
    document within the top k) is reported at `success_cutoffs` as a shallow
    secondary diagnostic. MRR@k and qrels_overlap@k have no trec_eval
    equivalent and are computed directly. trec_eval re-sorts runs by
    (score desc, doc_id desc), so the run is submitted with strictly decreasing
    rank-derived scores to preserve the aggregation's exact ordering.
    """
    retrieved_lists: dict[str, list[str]] = {}
    run: dict[str, dict[str, float]] = {}
    depth = rankings.indices.shape[1]
    for query_index, query_id in enumerate(query_ids):
        retrieved = [
            document_ids[int(index)]
            for index in rankings.indices[query_index]
            if index >= 0
        ]
        retrieved_lists[query_id] = retrieved
        run[query_id] = {
            document_id: float(depth - position)
            for position, document_id in enumerate(retrieved)
        }
    trec_scores = _trec_eval_scores(run, qrels, cutoffs, success_cutoffs)

    rows: list[dict[str, object]] = []
    for query_id in query_ids:
        retrieved = retrieved_lists[query_id]
        relevance = qrels.get(query_id, {})
        relevant_count = sum(value > 0 for value in relevance.values())
        query_scores = trec_scores.get(query_id, {})
        row: dict[str, object] = {
            "method": method,
            "query_id": query_id,
            "relevant_documents": relevant_count,
            "qrels_entries": len(relevance),
            "relevance_group": (
                "none"
                if relevant_count == 0
                else "single"
                if relevant_count == 1
                else "multiple"
            ),
        }
        for cutoff in cutoffs:
            row[f"ndcg@{cutoff}"] = query_scores.get(f"ndcg_cut_{cutoff}", 0.0)
            row[f"recall@{cutoff}"] = query_scores.get(f"recall_{cutoff}", 0.0)
            row[f"map@{cutoff}"] = query_scores.get(f"map_cut_{cutoff}", 0.0)
            row[f"mrr@{cutoff}"] = reciprocal_rank(retrieved, relevance, cutoff)
            considered = retrieved[:cutoff]
            row[f"qrels_overlap@{cutoff}"] = (
                sum(document_id in relevance for document_id in considered)
                / len(considered)
                if considered
                else 0.0
            )
        for cutoff in success_cutoffs:
            row[f"success@{cutoff}"] = query_scores.get(f"success_{cutoff}", 0.0)
        rows.append(row)
    return pd.DataFrame(rows)


def _trec_eval_scores(
    run: dict[str, dict[str, float]],
    qrels: dict[str, dict[str, int]],
    cutoffs: tuple[int, ...],
    success_cutoffs: tuple[int, ...] = (),
) -> dict[str, dict[str, float]]:
    # pytrec_eval silently drops queries without judgments; evaluate only judged
    # queries and let callers fill zeros so query alignment is preserved.
    judged = {query_id: relevance for query_id, relevance in qrels.items() if relevance}
    if not judged:
        return {}
    cutoff_spec = ",".join(str(cutoff) for cutoff in cutoffs)
    measures = {
        f"ndcg_cut.{cutoff_spec}",
        f"recall.{cutoff_spec}",
        f"map_cut.{cutoff_spec}",
    }
    if success_cutoffs:
        success_spec = ",".join(str(cutoff) for cutoff in success_cutoffs)
        measures.add(f"success.{success_spec}")
    evaluator = pytrec_eval.RelevanceEvaluator(judged, measures)
    return evaluator.evaluate(
        {query_id: documents for query_id, documents in run.items() if query_id in judged}
    )


def ndcg(retrieved: list[str], relevance: dict[str, int], k: int) -> float:
    """Linear-gain nDCG@k, matching trec_eval's ndcg_cut (used for oracle selection)."""
    gains = [relevance.get(document_id, 0) for document_id in retrieved[:k]]
    ideal = sorted((value for value in relevance.values() if value > 0), reverse=True)[:k]
    denominator = _dcg(ideal)
    return _dcg(gains) / denominator if denominator else 0.0


def _dcg(relevance: list[int]) -> float:
    return sum(
        grade / math.log2(rank + 1)
        for rank, grade in enumerate(relevance, start=1)
    )


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
    candidate_source: str = "stochastic_only",
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
                "candidate_source": candidate_source,
                f"selection_ndcg@{selection_cutoff}": utilities[selected],
            }
        )
    return Rankings(indices, scores), pd.DataFrame(selections)


def summarize(per_query: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["method"]
    if "sample_count" in per_query.columns:
        group_columns.append("sample_count")
    metric_columns = list(
        per_query.drop(
            columns=["method", "query_id", "sample_count"], errors="ignore"
        ).select_dtypes(include="number")
    )
    return per_query.groupby(group_columns, sort=False)[metric_columns].mean().reset_index()


def summarize_by_relevance_group(per_query: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [column for column in per_query.columns if "@" in column]
    group_columns = ["method"]
    if "sample_count" in per_query.columns:
        group_columns.append("sample_count")
    group_columns.append("relevance_group")
    grouped = per_query.groupby(group_columns, sort=False)
    result = grouped[metric_columns].mean().reset_index()
    result.insert(
        len(group_columns),
        "queries",
        grouped.size().to_numpy(),
    )
    return result


def dataset_query_diagnostics(
    query_ids: list[str],
    qrels: dict[str, dict[str, int]],
) -> pd.DataFrame:
    rows = []
    for query_id in query_ids:
        relevance = qrels.get(query_id, {})
        positive = [value for value in relevance.values() if value > 0]
        rows.append(
            {
                "query_id": query_id,
                "qrels_entries": len(relevance),
                "relevant_documents": len(positive),
                "maximum_relevance_grade": max(positive, default=0),
                "relevance_group": (
                    "none"
                    if not positive
                    else "single"
                    if len(positive) == 1
                    else "multiple"
                ),
            }
        )
    return pd.DataFrame(rows)


def paired_bootstrap_comparisons(
    per_query: pd.DataFrame,
    baseline: str,
    metric: str,
    replicates: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    groups = (
        per_query.groupby("sample_count", sort=False)
        if "sample_count" in per_query.columns
        else [(None, per_query)]
    )
    for sample_count, condition in groups:
        baseline_values = (
            condition.loc[condition["method"] == baseline, ["query_id", metric]]
            .set_index("query_id")[metric]
        )
        condition_seed = seed if sample_count is None else seed + int(sample_count)
        rng = np.random.default_rng(condition_seed)
        for method in condition["method"].unique():
            if method == baseline:
                continue
            candidate = (
                condition.loc[condition["method"] == method, ["query_id", metric]]
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
            row: dict[str, object] = {
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
            if sample_count is not None:
                row["sample_count"] = int(sample_count)
            rows.append(row)
    return pd.DataFrame(rows)
