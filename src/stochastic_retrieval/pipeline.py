from __future__ import annotations

import json
import time
from collections.abc import Iterable
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stochastic_retrieval.artifacts import ArtifactStore
from stochastic_retrieval.config import ProjectConfig
from stochastic_retrieval.data import IRDatasetAdapter, TextRecord
from stochastic_retrieval.encoding import (
    SentenceEmbeddingEncoder,
    configure_dropout,
    dropout_probability_summary,
    inspect_embedding_file,
)
from stochastic_retrieval.evaluation import (
    dataset_query_diagnostics,
    evaluate_rankings,
    iter_ranking_frames,
    oracle_best_of_n,
    paired_bootstrap_comparisons,
    summarize,
    summarize_by_relevance_group,
)
from stochastic_retrieval.reporting import RunReporter
from stochastic_retrieval.retrieval import (
    DenseRetriever,
    Rankings,
    embedding_diversity,
    majority_vote,
    maximum_score_rerank,
    mean_embedding,
    medoid_embedding,
    ranking_medoid,
    reciprocal_rank_fusion,
    trimmed_centroid,
    trimmed_centroid_diagnostics,
    variance_penalized_rerank,
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
    if config.experiment.retrieval_k < max(
        (*config.experiment.metric_cutoffs, *config.experiment.success_cutoffs)
    ):
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
        pooling=encoder.pooling,
        attention_implementation=encoder.attention_implementation,
        resolved_revision=encoder.resolved_revision,
    )
    dropout_effect = encoder.verify_dropout_effect()
    reporter.emit("dropout_effect_validated", scopes=dropout_effect)

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
    determinism = _verify_deterministic_reencode(
        encoder,
        dataset,
        config,
        deterministic_queries,
    )
    reporter.emit("deterministic_reencode_validated", **determinism)

    requested = set(config.experiment.methods)
    unknown = requested.difference(
        {
            "deterministic",
            "mean_embedding",
            "mean_score",
            "medoid_embedding",
            "trimmed_centroid",
            "rrf",
            "majority_vote",
            "ranking_medoid",
            "maximum_score",
            "variance_penalized_score",
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
        backend=config.experiment.retrieval_backend,
    ):
        retriever = DenseRetriever(
            document_embeddings,
            config.experiment.retrieval_k,
            backend=config.experiment.retrieval_backend,
            query_batch_size=config.experiment.retrieval_query_batch_size,
            corpus_batch_size=config.experiment.retrieval_corpus_batch_size,
        )
        rankings["deterministic"] = retriever.search(deterministic_queries)
        sample_rankings = [
            retriever.search(sample) for sample in stochastic_queries
        ]
        _save_sample_rankings(store, sample_rankings)

    with reporter.stage("aggregate_rankings", methods=sorted(requested)):
        trimmed_diagnostics: pd.DataFrame | None = None
        if "mean_embedding" in requested:
            rankings["mean_embedding"] = retriever.search(
                mean_embedding(stochastic_queries)
            )
        if "mean_score" in requested:
            # Dot(mean(q_i), d) exactly equals mean_i dot(q_i, d).
            rankings["mean_score"] = retriever.search(
                stochastic_queries.mean(axis=0)
            )
        if "medoid_embedding" in requested:
            rankings["medoid_embedding"] = retriever.search(
                medoid_embedding(stochastic_queries)
            )
        if "trimmed_centroid" in requested:
            rankings["trimmed_centroid"] = retriever.search(
                trimmed_centroid(
                    stochastic_queries,
                    config.experiment.trim_fraction,
                )
            )
            trimmed_diagnostics = trimmed_centroid_diagnostics(
                stochastic_queries,
                query_ids,
                config.experiment.trim_fraction,
            )
        if "rrf" in requested:
            rankings["rrf"] = reciprocal_rank_fusion(
                sample_rankings, config.experiment.retrieval_k
            )
        if "majority_vote" in requested:
            rankings["majority_vote"] = majority_vote(
                sample_rankings,
                config.experiment.retrieval_k,
                depth=config.experiment.majority_vote_depth,
            )
        if "ranking_medoid" in requested:
            rankings["ranking_medoid"] = ranking_medoid(
                sample_rankings,
                depth=config.experiment.ranking_medoid_depth,
            )
        if "maximum_score" in requested:
            rankings["maximum_score"] = maximum_score_rerank(
                stochastic_queries,
                document_embeddings,
                sample_rankings,
                config.experiment.retrieval_k,
            )
        if "variance_penalized_score" in requested:
            rankings["variance_penalized_score"] = variance_penalized_rerank(
                stochastic_queries,
                document_embeddings,
                sample_rankings,
                config.experiment.retrieval_k,
                penalty=config.experiment.variance_penalty_lambda,
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
                    success_cutoffs=config.experiment.success_cutoffs,
                )
                for method, ranking in selected_rankings.items()
            ],
            ignore_index=True,
        )
        summary = summarize(per_query)
        stratified_summary = summarize_by_relevance_group(per_query)
        query_diagnostics = dataset_query_diagnostics(query_ids, qrels)
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
        store.write_dataframe_chunks(
            _iter_all_ranking_frames(
                selected_rankings,
                query_ids,
                document_ids,
                config.experiment.ranking_write_query_batch_size,
            ),
            "rankings",
            "aggregated_rankings",
        )
        store.write_dataframe(per_query, "metrics", "per_query")
        store.write_dataframe(summary, "metrics", "summary")
        store.write_dataframe(
            stratified_summary,
            "metrics",
            "summary_by_relevance_group",
        )
        store.write_dataframe(comparisons, "metrics", "paired_bootstrap")
        store.write_dataframe(diversity, "analyses", "embedding_diversity")
        if trimmed_diagnostics is not None:
            store.write_dataframe(
                trimmed_diagnostics,
                "analyses",
                "trimmed_centroid_samples",
            )
        store.write_dataframe(
            query_diagnostics,
            "analyses",
            "dataset_query_diagnostics",
        )
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
                "model_provenance": {
                    "requested_revision": config.model.revision,
                    "resolved_revision": encoder.resolved_revision,
                    "attention_implementation": encoder.attention_implementation,
                },
                "embedding_validation": {
                    "documents": document_report,
                    "queries_deterministic": deterministic_report,
                    "stochasticity": stochasticity,
                    "deterministic_reencode": determinism,
                    "dropout_effect_scopes": dropout_effect,
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


def _iter_all_ranking_frames(
    rankings: dict[str, Rankings],
    query_ids: list[str],
    document_ids: list[str],
    query_batch_size: int,
) -> Iterable[pd.DataFrame]:
    for method, ranking in rankings.items():
        yield from iter_ranking_frames(
            method,
            ranking,
            query_ids,
            document_ids,
            query_batch_size=query_batch_size,
        )


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
) -> dict[str, Any]:
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
            "dropout_probabilities": dropout_probability_summary(
                encoder.model, encoder.config.dropout_scope
            ),
            "pooling": encoder.pooling,
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
    for sample_index, sample in enumerate(stochastic):
        if np.array_equal(sample, deterministic):
            raise RuntimeError(
                f"Stochastic query sample {sample_index} is identical to the "
                "deterministic embeddings"
            )
    for left in range(len(stochastic)):
        for right in range(left + 1, len(stochastic)):
            if np.array_equal(stochastic[left], stochastic[right]):
                raise RuntimeError(
                    f"Stochastic query samples {left} and {right} are identical"
                )
    return {
        "samples": len(stochastic),
        "mean_absolute_difference_from_deterministic": float(
            np.abs(stochastic - deterministic[None, :, :]).mean()
        ),
        "mean_coordinate_std": float(stochastic.std(axis=0).mean()),
    }


def _verify_deterministic_reencode(
    encoder: SentenceEmbeddingEncoder,
    dataset: IRDatasetAdapter,
    config: ProjectConfig,
    stored: np.ndarray,
    tolerance: float = 1e-6,
) -> dict[str, float | int | bool]:
    """Re-encode a probe batch of queries and compare with the stored embeddings.

    This runs on every invocation, including cache hits, so deterministic
    reproducibility is verified even when the embedding files are reused. Any
    difference above `tolerance` indicates state leakage (for example dropout
    left active) rather than benign floating-point noise.
    """
    probe_count = min(config.model.batch_size, len(stored))
    records = list(islice(dataset.iter_queries(), probe_count))
    reencoded = encoder.encode_records(
        records,
        prefix=config.model.query_prefix,
        seed=config.experiment.seed,
        stochastic=False,
    )
    reference = np.asarray(stored[:probe_count])
    max_difference = float(np.abs(reencoded - reference).max())
    if max_difference > tolerance:
        raise RuntimeError(
            "Deterministic re-encoding diverged from stored embeddings "
            f"(max difference {max_difference:.3e}); the deterministic pathway "
            "is not reproducible"
        )
    return {
        "probe_queries": probe_count,
        "bitwise_identical": bool(np.array_equal(reencoded, reference)),
        "max_absolute_difference": max_difference,
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
