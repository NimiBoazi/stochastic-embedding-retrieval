from __future__ import annotations

import json
import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from stochastic_retrieval.artifacts import ArtifactStore
from stochastic_retrieval.config import ProjectConfig
from stochastic_retrieval.data import IRDatasetAdapter, TextRecord
from stochastic_retrieval.encoding import (
    SentenceEmbeddingEncoder,
    configure_dropout,
    inspect_embedding_file,
)
from stochastic_retrieval.evaluation import (
    evaluate_rankings,
    oracle_best_of_n,
    paired_bootstrap_comparisons,
    ranking_frame,
    summarize,
)
from stochastic_retrieval.reporting import RunReporter
from stochastic_retrieval.retrieval import (
    Rankings,
    embedding_diversity,
    exact_search,
    majority_vote,
    mean_embedding,
    medoid_embedding,
    reciprocal_rank_fusion,
)


def run_experiment(config: ProjectConfig, project_root: Path) -> ArtifactStore:
    store = ArtifactStore(project_root, config)
    reporter = RunReporter(store.run_id, store.run_dir)
    started = time.perf_counter()
    reporter.emit(
        "run_started",
        config_fingerprint=config.fingerprint,
        model=config.model.name,
        dataset=config.dataset.ir_dataset_id,
        methods=list(config.experiment.methods),
    )
    store.write_manifest(config, project_root, extra={"status": "running"})
    try:
        result = _execute_experiment(config, project_root, store, reporter)
    except Exception as exc:
        reporter.emit(
            "run_failed",
            duration_seconds=round(time.perf_counter() - started, 3),
            error_type=type(exc).__name__,
            error=str(exc),
        )
        store.write_manifest(
            config,
            project_root,
            extra={
                "status": "failed",
                "failure": {"type": type(exc).__name__, "message": str(exc)},
            },
        )
        raise
    reporter.emit(
        "run_completed",
        duration_seconds=round(time.perf_counter() - started, 3),
    )
    return result


def _execute_experiment(
    config: ProjectConfig,
    project_root: Path,
    store: ArtifactStore,
    reporter: RunReporter,
) -> ArtifactStore:
    if "deterministic" not in config.experiment.methods:
        raise ValueError("methods must include 'deterministic' as the paired baseline")
    if config.experiment.document_samples != 1:
        raise NotImplementedError(
            "The first implementation supports deterministic documents only. "
            "Keep document_samples=1; stochastic document banks are the next stage."
        )
    if config.experiment.query_samples < 1:
        raise ValueError("query_samples must be at least 1")
    if config.experiment.retrieval_k < max(config.experiment.metric_cutoffs):
        raise ValueError("retrieval_k must cover the largest metric cutoff")

    with reporter.stage("load_dataset"):
        dataset = IRDatasetAdapter(config.dataset)
    reporter.emit(
        "dataset_loaded",
        queries=dataset.query_count,
        documents=dataset.document_count,
    )
    with reporter.stage("load_model"):
        encoder = SentenceEmbeddingEncoder(config.model, config.device)
    reporter.emit(
        "model_loaded",
        device=encoder.device,
        embedding_dimension=encoder.dimension,
    )

    with reporter.stage("encode_and_validate_embeddings"):
        document_report = _encode_if_missing(
            encoder=encoder,
            output_path=store.embedding_path("documents", 0),
            ids_path=store.ids_path("documents"),
            records=dataset.iter_documents(),
            count=dataset.document_count,
            prefix=config.model.document_prefix,
            seed=config.experiment.seed,
            stochastic=False,
            description="Deterministic documents",
        )
        reporter.emit("embedding_validated", kind="documents", **document_report)
        deterministic_report = _encode_if_missing(
            encoder=encoder,
            output_path=store.embedding_path("queries_deterministic", 0),
            ids_path=store.ids_path("queries_deterministic"),
            records=dataset.iter_queries(),
            count=dataset.query_count,
            prefix=config.model.query_prefix,
            seed=config.experiment.seed,
            stochastic=False,
            description="Deterministic queries",
        )
        reporter.emit(
            "embedding_validated",
            kind="queries_deterministic",
            **deterministic_report,
        )
        for sample in range(config.experiment.query_samples):
            sample_report = _encode_if_missing(
                encoder=encoder,
                output_path=store.embedding_path("queries_stochastic", sample),
                ids_path=store.ids_path("queries_stochastic"),
                records=dataset.iter_queries(),
                count=dataset.query_count,
                prefix=config.model.query_prefix,
                seed=config.experiment.seed + 10_000 + sample,
                stochastic=True,
                description=(
                    f"Stochastic queries {sample + 1}/"
                    f"{config.experiment.query_samples}"
                ),
            )
            reporter.emit(
                "embedding_validated",
                kind="queries_stochastic",
                sample=sample,
                **sample_report,
            )

    document_embeddings = store.load_embeddings("documents", 0)
    deterministic_queries = store.load_embeddings("queries_deterministic", 0)
    stochastic_queries = np.stack(
        [
            store.load_embeddings("queries_stochastic", sample)
            for sample in range(config.experiment.query_samples)
        ]
    )
    query_ids = store.load_ids("queries_deterministic")
    document_ids = store.load_ids("documents")
    qrels = dataset.qrels()
    stochasticity = _validate_stochasticity(
        deterministic_queries,
        stochastic_queries,
    )
    reporter.emit("stochasticity_validated", **stochasticity)

    requested = set(config.experiment.methods)
    unknown = requested.difference(
        {
            "deterministic",
            "mean_embedding",
            "mean_score",
            "medoid_embedding",
            "rrf",
            "majority_vote",
            "oracle_best_of_n",
        }
    )
    if unknown:
        raise ValueError(f"Unknown aggregation methods: {sorted(unknown)}")

    rankings: dict[str, Rankings] = {}
    with reporter.stage(
        "retrieve_base_rankings",
        deterministic_runs=1,
        stochastic_runs=config.experiment.query_samples,
    ):
        rankings["deterministic"] = exact_search(
            deterministic_queries, document_embeddings, config.experiment.retrieval_k
        )
        sample_rankings = [
            exact_search(sample, document_embeddings, config.experiment.retrieval_k)
            for sample in stochastic_queries
        ]
        _save_sample_rankings(store, sample_rankings)

    with reporter.stage("aggregate_rankings", methods=sorted(requested)):
        if "mean_embedding" in requested:
            rankings["mean_embedding"] = exact_search(
                mean_embedding(stochastic_queries),
                document_embeddings,
                config.experiment.retrieval_k,
            )
        if "mean_score" in requested:
            # Dot(mean(q_i), d) exactly equals mean_i dot(q_i, d).
            rankings["mean_score"] = exact_search(
                stochastic_queries.mean(axis=0),
                document_embeddings,
                config.experiment.retrieval_k,
            )
        if "medoid_embedding" in requested:
            rankings["medoid_embedding"] = exact_search(
                medoid_embedding(stochastic_queries),
                document_embeddings,
                config.experiment.retrieval_k,
            )
        if "rrf" in requested:
            rankings["rrf"] = reciprocal_rank_fusion(
                sample_rankings, config.experiment.retrieval_k
            )
        if "majority_vote" in requested:
            rankings["majority_vote"] = majority_vote(
                sample_rankings, config.experiment.retrieval_k
            )
        if "oracle_best_of_n" in requested:
            oracle, selections = oracle_best_of_n(
                sample_rankings,
                query_ids,
                document_ids,
                qrels,
                selection_cutoff=10,
            )
            rankings["oracle_best_of_n"] = oracle
            store.write_dataframe(selections, "analyses", "oracle_selections")

    with reporter.stage("evaluate"):
        selected_rankings = {
            method: ranking
            for method, ranking in rankings.items()
            if method in requested
        }
        per_query = pd.concat(
            [
                evaluate_rankings(
                    method,
                    ranking,
                    query_ids,
                    document_ids,
                    qrels,
                    config.experiment.metric_cutoffs,
                )
                for method, ranking in selected_rankings.items()
            ],
            ignore_index=True,
        )
        ranking_rows = pd.concat(
            [
                ranking_frame(method, ranking, query_ids, document_ids)
                for method, ranking in selected_rankings.items()
            ],
            ignore_index=True,
        )
        summary = summarize(per_query)
        primary_metric = (
            "ndcg@10"
            if 10 in config.experiment.metric_cutoffs
            else f"ndcg@{config.experiment.metric_cutoffs[0]}"
        )
        comparisons = paired_bootstrap_comparisons(
            per_query,
            baseline="deterministic",
            metric=primary_metric,
            replicates=config.experiment.bootstrap_replicates,
            seed=config.experiment.seed,
        )
        diversity = embedding_diversity(stochastic_queries, query_ids)
    reporter.emit("metrics_summary", records=summary.to_dict(orient="records"))

    with reporter.stage("persist_results"):
        store.write_dataframe(ranking_rows, "rankings", "aggregated_rankings")
        store.write_dataframe(per_query, "metrics", "per_query")
        store.write_dataframe(summary, "metrics", "summary")
        store.write_dataframe(comparisons, "metrics", "paired_bootstrap")
        store.write_dataframe(diversity, "analyses", "embedding_diversity")
        _write_qrels(store, qrels)
        store.write_manifest(
            config,
            project_root,
            extra={
                "status": "completed",
                "dataset": {
                    "queries": len(query_ids),
                    "documents": len(document_ids),
                    "qrels_queries": len(qrels),
                },
                "embedding_validation": {
                    "documents": document_report,
                    "queries_deterministic": deterministic_report,
                    "stochasticity": stochasticity,
                },
                "methodology_notes": {
                    "oracle_best_of_n": "Uses qrels and is an oracle upper bound.",
                    "mean_score": (
                        "For dot-product retrieval this is algebraically equivalent to "
                        "scoring with the unnormalized mean query embedding."
                    ),
                },
            },
        )
    return store


def _encode_if_missing(
    encoder: SentenceEmbeddingEncoder,
    output_path: Path,
    ids_path: Path,
    records: Iterable[TextRecord],
    count: int,
    prefix: str,
    seed: int,
    stochastic: bool,
    description: str,
) -> dict[str, float | int | bool]:
    if output_path.exists() and ids_path.exists():
        enabled = configure_dropout(
            encoder.model,
            stochastic=stochastic,
            scope=encoder.config.dropout_scope,
            probability=encoder.config.dropout_probability,
        )
        return {
            **inspect_embedding_file(output_path, expected_count=count),
            "cached": True,
            "seed": seed,
            "stochastic": stochastic,
            "enabled_dropout_modules": enabled,
        }
    return encoder.encode_to_file(
        records=records,
        count=count,
        output_path=output_path,
        ids_path=ids_path,
        prefix=prefix,
        seed=seed,
        stochastic=stochastic,
        description=description,
    )


def _validate_stochasticity(
    deterministic: np.ndarray,
    stochastic: np.ndarray,
) -> dict[str, float | int]:
    if stochastic.ndim != 3 or stochastic.shape[1:] != deterministic.shape:
        raise RuntimeError(
            "Stochastic and deterministic query embedding shapes are inconsistent"
        )
    if np.array_equal(stochastic[0], deterministic):
        raise RuntimeError(
            "Stochastic query embeddings are identical to deterministic embeddings"
        )
    if len(stochastic) > 1 and all(
        np.array_equal(stochastic[0], sample) for sample in stochastic[1:]
    ):
        raise RuntimeError("All stochastic query samples are identical")
    return {
        "samples": len(stochastic),
        "mean_absolute_difference_from_deterministic": float(
            np.abs(stochastic - deterministic[None, :, :]).mean()
        ),
        "mean_coordinate_std": float(stochastic.std(axis=0).mean()),
    }


def _save_sample_rankings(store: ArtifactStore, rankings: list[Rankings]) -> None:
    directory = store.ranking_dir / "samples"
    directory.mkdir(parents=True, exist_ok=True)
    for sample, ranking in enumerate(rankings):
        np.savez_compressed(
            directory / f"sample_{sample:03d}.npz",
            indices=ranking.indices,
            scores=ranking.scores,
        )


def _write_qrels(
    store: ArtifactStore, qrels: dict[str, dict[str, int]]
) -> None:
    path = store.run_dir / "qrels.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(qrels, handle, sort_keys=True)
