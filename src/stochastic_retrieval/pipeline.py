from __future__ import annotations

import json
import time
from collections.abc import Iterable
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stochastic_retrieval.artifacts import (
    ArtifactStore,
    document_cache_key,
    document_encoding_contract,
    file_sha256,
)
from stochastic_retrieval.config import ProjectConfig
from stochastic_retrieval.data import IRDatasetAdapter, TextRecord
from stochastic_retrieval.distribution import (
    centroid_convergence,
    cluster_diagnostics,
    distribution_outcome_correlations,
    embedding_distribution,
    retrieval_distribution,
    retrieval_document_distribution,
)
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
from stochastic_retrieval.noise_controls import evaluate_matched_noise_oracles
from stochastic_retrieval.reporting import RunReporter
from stochastic_retrieval.retrieval import (
    DenseRetriever,
    Rankings,
    anchored_centroid,
    deterministic_score_margin,
    embedding_diversity,
    gated_ranking,
    majority_vote,
    maximum_score_rerank,
    mean_embedding,
    medoid_embedding,
    quartile_gate_masks,
    ranking_disagreement,
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
    sample_counts = config.experiment.resolved_sample_counts
    if "deterministic" not in config.experiment.methods:
        raise ValueError("methods must include 'deterministic' as the paired baseline")
    if config.experiment.document_samples != 1:
        raise NotImplementedError(
            "The first implementation supports deterministic documents only. "
            "Keep document_samples=1; stochastic document banks are the next stage."
        )
    if not sample_counts or any(count < 1 for count in sample_counts):
        raise ValueError("All sample_counts must be positive")
    if tuple(sorted(set(sample_counts))) != sample_counts:
        raise ValueError("sample_counts must be unique and sorted")
    if config.experiment.query_samples < max(sample_counts):
        raise ValueError("query_samples must be at least the largest sample_count")
    if config.experiment.retrieval_k < max(
        (*config.experiment.metric_cutoffs, *config.experiment.success_cutoffs)
    ):
        raise ValueError("retrieval_k must cover the largest metric cutoff")

    with reporter.stage("load_dataset"):
        dataset = IRDatasetAdapter(config.dataset)
        expected_query_ids = [record.item_id for record in dataset.iter_queries()]
        expected_document_ids = [record.item_id for record in dataset.iter_documents()]
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
        document_report = _prepare_document_embeddings(
            encoder=encoder,
            dataset=dataset,
            config=config,
            store=store,
            expected_ids=expected_document_ids,
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
            expected_ids=expected_query_ids,
        )
        reporter.emit(
            "embedding_validated",
            kind="queries_deterministic",
            **deterministic_report,
        )
        seed_schedule = _sample_seed_schedule(
            config.experiment.seed,
            sample_counts,
            config.experiment.independent_sample_banks,
        )
        storage_counts = (
            sample_counts
            if config.experiment.independent_sample_banks
            else (max(sample_counts),)
        )
        for storage_count in storage_counts:
            condition_seeds = (
                seed_schedule[storage_count]
                if config.experiment.independent_sample_banks
                else seed_schedule[max(sample_counts)]
            )
            for sample, seed in enumerate(condition_seeds):
                sample_report = _encode_if_missing(
                    encoder=encoder,
                    output_path=store.embedding_path(
                        "queries_stochastic", sample, storage_count
                    ),
                    ids_path=store.ids_path("queries_stochastic", storage_count),
                    records=dataset.iter_queries(),
                    count=dataset.query_count,
                    prefix=config.model.query_prefix,
                    seed=seed,
                    stochastic=True,
                    description=(
                        f"Stochastic queries N={storage_count}: "
                        f"{sample + 1}/{len(condition_seeds)}"
                    ),
                    expected_ids=expected_query_ids,
                )
                reporter.emit(
                    "embedding_validated",
                    kind="queries_stochastic",
                    sample_count=storage_count,
                    sample=sample,
                    **sample_report,
                )

    document_embeddings = store.load_embeddings("documents", 0)
    deterministic_queries = store.load_embeddings("queries_deterministic", 0)
    query_ids = store.load_ids("queries_deterministic")
    document_ids = store.load_ids("documents")
    qrels = dataset.qrels()
    _validate_alignment(
        query_ids,
        document_ids,
        qrels,
        expected_query_ids,
        expected_document_ids,
    )
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
            "anchored_centroid_a010",
            "anchored_centroid_a025",
            "anchored_centroid_a050",
            "anchored_centroid_a075",
            "margin_gated_anchor",
            "disagreement_gated_anchor",
        }
    )
    if unknown:
        raise ValueError(f"Unknown aggregation methods: {sorted(unknown)}")

    rankings_by_count: dict[int, dict[str, Rankings]] = {}
    per_query_parts: list[pd.DataFrame] = []
    diversity_parts: list[pd.DataFrame] = []
    trimmed_parts: list[pd.DataFrame] = []
    oracle_parts: list[pd.DataFrame] = []
    gate_parts: list[pd.DataFrame] = []
    embedding_distribution_parts: list[pd.DataFrame] = []
    cluster_parts: list[pd.DataFrame] = []
    convergence_parts: list[pd.DataFrame] = []
    retrieval_distribution_parts: list[pd.DataFrame] = []
    retrieval_document_parts: list[pd.DataFrame] = []
    noise_oracle_per_query_parts: list[pd.DataFrame] = []
    noise_oracle_selection_parts: list[pd.DataFrame] = []
    noise_control_diagnostic_parts: list[pd.DataFrame] = []
    stochasticity_by_count: dict[str, dict[str, float | int]] = {}
    with reporter.stage(
        "retrieve_base_rankings",
        deterministic_runs=1,
        stochastic_runs=sum(sample_counts)
        if config.experiment.independent_sample_banks
        else max(sample_counts),
        backend=config.experiment.retrieval_backend,
    ):
        retriever = DenseRetriever(
            document_embeddings,
            config.experiment.retrieval_k,
            backend=config.experiment.retrieval_backend,
            query_batch_size=config.experiment.retrieval_query_batch_size,
            corpus_batch_size=config.experiment.retrieval_corpus_batch_size,
        )
        deterministic_ranking = retriever.search(deterministic_queries)
        for sample_count in sample_counts:
            storage_count = (
                sample_count
                if config.experiment.independent_sample_banks
                else max(sample_counts)
            )
            stochastic_queries = np.stack(
                [
                    store.load_embeddings(
                        "queries_stochastic", sample, storage_count
                    )
                    for sample in range(sample_count)
                ]
            )
            stochasticity = _validate_stochasticity(
                deterministic_queries, stochastic_queries
            )
            stochasticity_by_count[str(sample_count)] = stochasticity
            reporter.emit(
                "stochasticity_validated",
                sample_count=sample_count,
                **stochasticity,
            )
            sample_rankings = [
                retriever.search(sample) for sample in stochastic_queries
            ]
            _save_sample_rankings(store, sample_rankings, sample_count)
            (
                noise_per_query,
                noise_selections,
                noise_diagnostics,
            ) = evaluate_matched_noise_oracles(
                deterministic_queries,
                stochastic_queries,
                deterministic_ranking,
                sample_rankings,
                retriever,
                query_ids,
                document_ids,
                qrels,
                metric_cutoffs=config.experiment.metric_cutoffs,
                success_cutoffs=config.experiment.success_cutoffs,
                sample_count=sample_count,
                seed=config.experiment.seed + 900_000 + sample_count,
                selection_cutoff=10,
            )
            noise_oracle_per_query_parts.append(noise_per_query)
            noise_oracle_selection_parts.append(noise_selections)
            noise_control_diagnostic_parts.append(noise_diagnostics)
            with reporter.stage(
                "aggregate_and_evaluate",
                sample_count=sample_count,
                methods=sorted(requested),
            ):
                (
                    rankings,
                    trimmed_diagnostics,
                    selections,
                    gate_diagnostics,
                ) = _aggregate_condition(
                    requested=requested,
                    deterministic_ranking=deterministic_ranking,
                    deterministic_queries=deterministic_queries,
                    stochastic_queries=stochastic_queries,
                    sample_rankings=sample_rankings,
                    retriever=retriever,
                    document_embeddings=document_embeddings,
                    query_ids=query_ids,
                    document_ids=document_ids,
                    qrels=qrels,
                    config=config,
                )
                rankings_by_count[sample_count] = rankings
                for method, ranking in rankings.items():
                    frame = evaluate_rankings(
                        method,
                        ranking,
                        query_ids,
                        document_ids,
                        qrels,
                        config.experiment.metric_cutoffs,
                        success_cutoffs=config.experiment.success_cutoffs,
                    )
                    frame.insert(1, "sample_count", sample_count)
                    per_query_parts.append(frame)
                diversity = embedding_diversity(stochastic_queries, query_ids)
                diversity.insert(1, "sample_count", sample_count)
                diversity_parts.append(diversity)
                if trimmed_diagnostics is not None:
                    trimmed_diagnostics.insert(1, "sample_count", sample_count)
                    trimmed_parts.append(trimmed_diagnostics)
                if selections is not None:
                    selections.insert(1, "sample_count", sample_count)
                    selected_seeds = seed_schedule[sample_count]
                    selections["selected_seed"] = [
                        selected_seeds[int(sample)]
                        for sample in selections["selected_sample"]
                    ]
                    oracle_parts.append(selections)
                gate_diagnostics.insert(1, "sample_count", sample_count)
                gate_parts.append(gate_diagnostics)
                embedding_shape = embedding_distribution(
                    stochastic_queries,
                    deterministic_queries,
                    query_ids,
                )
                clusters = cluster_diagnostics(
                    stochastic_queries,
                    query_ids,
                    seed=config.experiment.seed,
                )
                convergence = centroid_convergence(
                    stochastic_queries,
                    query_ids,
                    bootstrap_replicates=5,
                    seed=config.experiment.seed,
                    search=retriever.search,
                )
                retrieval_shape = retrieval_distribution(
                    sample_rankings,
                    deterministic_ranking,
                    query_ids,
                    document_ids,
                    qrels,
                )
                retrieval_documents = retrieval_document_distribution(
                    sample_rankings,
                    deterministic_ranking,
                    query_ids,
                    document_ids,
                    qrels,
                )
                for frame, destination in (
                    (embedding_shape, embedding_distribution_parts),
                    (clusters, cluster_parts),
                    (convergence, convergence_parts),
                    (retrieval_shape, retrieval_distribution_parts),
                    (retrieval_documents, retrieval_document_parts),
                ):
                    frame.insert(1, "sample_count", sample_count)
                    frame.insert(1, "dropout_condition", _dropout_condition(config))
                    frame.insert(1, "dataset", config.dataset.name)
                    destination.append(frame)

    with reporter.stage("evaluate_summaries"):
        per_query = pd.concat(per_query_parts, ignore_index=True)
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
        noise_oracle_per_query = pd.concat(
            noise_oracle_per_query_parts,
            ignore_index=True,
        )
        noise_oracle_summary = summarize(noise_oracle_per_query)
        noise_bootstrap_parts = []
        for baseline in ("deterministic", "dropout_oracle"):
            comparison = paired_bootstrap_comparisons(
                noise_oracle_per_query,
                baseline=baseline,
                metric=primary_metric,
                replicates=config.experiment.bootstrap_replicates,
                seed=config.experiment.seed,
            )
            comparison.insert(0, "comparison", f"versus_{baseline}")
            noise_bootstrap_parts.append(comparison)
        noise_oracle_bootstrap = pd.concat(
            noise_bootstrap_parts,
            ignore_index=True,
        )
        outcome_correlation_parts: list[pd.DataFrame] = []
        outcome_risk_parts: list[pd.DataFrame] = []
        for sample_count in sample_counts:
            embedding_features = next(
                frame
                for frame in embedding_distribution_parts
                if int(frame["sample_count"].iloc[0]) == sample_count
            )
            retrieval_features = next(
                frame
                for frame in retrieval_distribution_parts
                if int(frame["sample_count"].iloc[0]) == sample_count
            )
            gate_features = next(
                frame
                for frame in gate_parts
                if int(frame["sample_count"].iloc[0]) == sample_count
            )
            features = embedding_features.merge(
                retrieval_features,
                on=["dataset", "dropout_condition", "sample_count", "query_id"],
                suffixes=("", "_retrieval"),
            ).merge(
                gate_features,
                on=["sample_count", "query_id"],
                suffixes=("", "_gate"),
            )
            correlations, risks = distribution_outcome_correlations(
                features,
                per_query,
                sample_count,
                metric=primary_metric,
            )
            for frame in (correlations, risks):
                frame.insert(0, "dropout_condition", _dropout_condition(config))
                frame.insert(0, "dataset", config.dataset.name)
            outcome_correlation_parts.append(correlations)
            outcome_risk_parts.append(risks)
    reporter.emit("metrics_summary", records=summary.to_dict(orient="records"))

    with reporter.stage("persist_results"):
        store.write_dataframe_chunks(
            _iter_all_ranking_frames(
                rankings_by_count,
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
        store.write_dataframe(
            noise_oracle_per_query,
            "metrics",
            "noise_oracle_per_query",
        )
        store.write_dataframe(
            noise_oracle_summary,
            "metrics",
            "noise_oracle_summary",
        )
        store.write_dataframe(
            noise_oracle_bootstrap,
            "metrics",
            "noise_oracle_bootstrap",
        )
        store.write_dataframe(
            pd.concat(noise_oracle_selection_parts, ignore_index=True),
            "analyses",
            "noise_oracle_selections",
        )
        store.write_dataframe(
            pd.concat(noise_control_diagnostic_parts, ignore_index=True),
            "analyses",
            "noise_control_diagnostics",
        )
        store.write_dataframe(
            pd.concat(diversity_parts, ignore_index=True),
            "analyses",
            "embedding_diversity",
        )
        if trimmed_parts:
            store.write_dataframe(
                pd.concat(trimmed_parts, ignore_index=True),
                "analyses",
                "trimmed_centroid_samples",
            )
        if oracle_parts:
            store.write_dataframe(
                pd.concat(oracle_parts, ignore_index=True),
                "analyses",
                "oracle_selections",
            )
        store.write_dataframe(
            pd.concat(gate_parts, ignore_index=True),
            "analyses",
            "gate_diagnostics",
        )
        for frames, name in (
            (embedding_distribution_parts, "embedding_distribution"),
            (cluster_parts, "cluster_diagnostics"),
            (convergence_parts, "centroid_convergence"),
            (retrieval_distribution_parts, "retrieval_distribution"),
            (retrieval_document_parts, "retrieval_document_distribution"),
            (outcome_correlation_parts, "distribution_outcome_correlations"),
            (outcome_risk_parts, "distribution_outcome_risk"),
        ):
            store.write_dataframe(
                pd.concat(frames, ignore_index=True),
                "analyses",
                name,
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
                    "stochasticity_by_sample_count": stochasticity_by_count,
                    "deterministic_reencode": determinism,
                    "dropout_effect_scopes": dropout_effect,
                },
                "sample_count_protocol": {
                    "sample_counts": list(sample_counts),
                    "independent_sample_banks": (
                        config.experiment.independent_sample_banks
                    ),
                    "total_stochastic_query_passes": (
                        sum(sample_counts)
                        if config.experiment.independent_sample_banks
                        else max(sample_counts)
                    ),
                    "seeds": {
                        str(count): seeds
                        for count, seeds in seed_schedule.items()
                    },
                },
                "document_cache": {
                    "cache_key": document_cache_key(config),
                    **document_report["cache"],
                },
                "methodology_notes": {
                    "oracle_best_of_n": (
                        "Uses qrels and selects only from stochastic-query "
                        "rankings; deterministic is never a candidate."
                    ),
                    "mean_score": (
                        "For dot-product retrieval this is algebraically equivalent to "
                        "scoring with the unnormalized mean query embedding."
                    ),
                },
                "oracle": {
                    "candidate_source": "stochastic_only",
                    "selection_metric": "linear_gain_ndcg@10",
                    "uses_qrels": True,
                },
                "matched_noise_oracle": {
                    "seed_offset": 900_000,
                    "selection_metric": "linear_gain_ndcg@10",
                    "full_covariance_gaussian": (
                        "Empirical tangent mean/covariance directions with exact "
                        "per-sample dropout angular displacement"
                    ),
                    "isotropic": (
                        "Uniform tangent directions with exact per-sample "
                        "dropout angular displacement"
                    ),
                    "magnitude_contract": (
                        "Each artificial sample exactly matches its paired "
                        "dropout angle and unit-sphere chord distance"
                    ),
                },
                "distribution_analysis": {
                    "normalization": "per-query unit-normalized embeddings",
                    "cluster_counts": [2, 3, 4],
                    "cluster_bootstrap_replicates": 5,
                    "centroid_bootstrap_replicates": 5,
                    "seed": config.experiment.seed,
                    "diagnostic_correlations": "post_hoc_label_using",
                },
            },
        )
    return store


def _iter_all_ranking_frames(
    rankings_by_count: dict[int, dict[str, Rankings]],
    query_ids: list[str],
    document_ids: list[str],
    query_batch_size: int,
) -> Iterable[pd.DataFrame]:
    for sample_count, rankings in rankings_by_count.items():
        for method, ranking in rankings.items():
            yield from iter_ranking_frames(
                method,
                ranking,
                query_ids,
                document_ids,
                query_batch_size=query_batch_size,
                sample_count=sample_count,
            )


def _sample_seed_schedule(
    seed: int,
    sample_counts: tuple[int, ...],
    independent: bool,
) -> dict[int, list[int]]:
    start = seed + 10_000
    if not independent:
        shared = list(range(start, start + max(sample_counts)))
        return {count: shared[:count] for count in sample_counts}
    schedule: dict[int, list[int]] = {}
    cursor = start
    for count in sample_counts:
        schedule[count] = list(range(cursor, cursor + count))
        cursor += count
    return schedule


def _dropout_condition(config: ProjectConfig) -> str:
    probability = config.model.dropout_probability
    suffix = "native" if probability is None else f"p{probability:.2f}"
    return f"{config.model.dropout_scope}_{suffix}"


def _prepare_document_embeddings(
    encoder: SentenceEmbeddingEncoder,
    dataset: IRDatasetAdapter,
    config: ProjectConfig,
    store: ArtifactStore,
    expected_ids: list[str],
) -> dict[str, Any]:
    cache = store.promote_compatible_document_cache(config)
    if cache is None:
        report = encoder.encode_to_file(
            records=dataset.iter_documents(),
            count=dataset.document_count,
            output_path=store.cached_document_embeddings,
            ids_path=store.cached_document_ids,
            prefix=config.model.document_prefix,
            seed=config.experiment.seed,
            stochastic=False,
            description="Deterministic documents (shared cache)",
        )
        store.write_document_cache_metadata(config, source="encoded")
        cache = {"source": "encoded", "cache_key": document_cache_key(config)}
    else:
        configure_dropout(encoder.model, stochastic=False)
        report = {
            **inspect_embedding_file(
                store.cached_document_embeddings,
                expected_count=dataset.document_count,
            ),
            "cached": True,
            "seed": config.experiment.seed,
            "stochastic": False,
            "enabled_dropout_modules": 0,
            "dropout_probabilities": dropout_probability_summary(
                encoder.model, encoder.config.dropout_scope
            ),
            "pooling": encoder.pooling,
        }
    metadata = json.loads(store.cached_document_metadata.read_text(encoding="utf-8"))
    if metadata.get("contract") != document_encoding_contract(config):
        raise RuntimeError("Document cache contract does not match this configuration")
    if metadata.get("embeddings_sha256") != file_sha256(
        store.cached_document_embeddings
    ):
        raise RuntimeError("Cached document embedding hash mismatch")
    if metadata.get("ids_sha256") != file_sha256(store.cached_document_ids):
        raise RuntimeError("Cached document ID hash mismatch")
    with store.cached_document_ids.open(encoding="utf-8") as handle:
        cached_ids = [json.loads(line) for line in handle]
    if cached_ids != expected_ids:
        raise RuntimeError("Cached document IDs or ordering do not match the dataset")
    report.update(
        inspect_embedding_file(
            store.cached_document_embeddings,
            expected_count=len(expected_ids),
        )
    )
    store.materialize_documents_in_run()
    report["cache"] = cache
    return report


def _validate_alignment(
    query_ids: list[str],
    document_ids: list[str],
    qrels: dict[str, dict[str, int]],
    expected_query_ids: list[str],
    expected_document_ids: list[str],
) -> None:
    if query_ids != expected_query_ids:
        raise RuntimeError("Query IDs or ordering do not match the dataset")
    if document_ids != expected_document_ids:
        raise RuntimeError("Document IDs or ordering do not match the dataset")
    if set(qrels) != set(query_ids):
        raise RuntimeError("Qrels query IDs do not match the evaluated query IDs")
    document_id_set = set(document_ids)
    unknown = {
        document_id
        for relevance in qrels.values()
        for document_id in relevance
        if document_id not in document_id_set
    }
    if unknown:
        raise RuntimeError(
            f"Qrels reference {len(unknown)} documents outside the encoded corpus"
        )


def _aggregate_condition(
    requested: set[str],
    deterministic_ranking: Rankings,
    deterministic_queries: np.ndarray,
    stochastic_queries: np.ndarray,
    sample_rankings: list[Rankings],
    retriever: DenseRetriever,
    document_embeddings: np.ndarray,
    query_ids: list[str],
    document_ids: list[str],
    qrels: dict[str, dict[str, int]],
    config: ProjectConfig,
) -> tuple[
    dict[str, Rankings],
    pd.DataFrame | None,
    pd.DataFrame | None,
    pd.DataFrame,
]:
    rankings: dict[str, Rankings] = {"deterministic": deterministic_ranking}
    trimmed_diagnostics: pd.DataFrame | None = None
    selections: pd.DataFrame | None = None
    anchor_methods = {
        "anchored_centroid_a010": 0.10,
        "anchored_centroid_a025": 0.25,
        "anchored_centroid_a050": 0.50,
        "anchored_centroid_a075": 0.75,
    }
    anchored_rankings: dict[float, Rankings] = {}
    for method, alpha in anchor_methods.items():
        if method in requested or (
            alpha == 0.25
            and {
                "margin_gated_anchor",
                "disagreement_gated_anchor",
            }
            & requested
        ):
            anchored_rankings[alpha] = retriever.search(
                anchored_centroid(
                    deterministic_queries,
                    stochastic_queries,
                    alpha,
                )
            )
        if method in requested:
            rankings[method] = anchored_rankings[alpha]

    score_depth = deterministic_ranking.scores.shape[1]
    margin_1 = (
        deterministic_score_margin(deterministic_ranking, 1)
        if score_depth > 1
        else np.full(len(query_ids), np.nan, dtype=np.float32)
    )
    margin_10 = (
        deterministic_score_margin(deterministic_ranking, 10)
        if score_depth > 10
        else margin_1.copy()
    )
    disagreement = ranking_disagreement(
        sample_rankings,
        depth=min(10, sample_rankings[0].indices.shape[1]),
    )
    (
        margin_mask,
        margin_threshold,
        disagreement_mask,
        disagreement_threshold,
    ) = quartile_gate_masks(
        margin_10,
        disagreement,
        len(sample_rankings),
    )
    if "margin_gated_anchor" in requested:
        rankings["margin_gated_anchor"] = gated_ranking(
            deterministic_ranking,
            anchored_rankings[0.25],
            margin_mask,
        )
    if "disagreement_gated_anchor" in requested:
        rankings["disagreement_gated_anchor"] = gated_ranking(
            deterministic_ranking,
            anchored_rankings[0.25],
            disagreement_mask,
        )
    stochastic_centroid = mean_embedding(stochastic_queries)
    deterministic_normalized = deterministic_queries / np.maximum(
        np.linalg.norm(deterministic_queries, axis=1, keepdims=True),
        np.finfo(np.float32).eps,
    )
    centroid_cosine = np.sum(deterministic_normalized * stochastic_centroid, axis=1)
    gate_diagnostics = pd.DataFrame(
        {
            "query_id": query_ids,
            "deterministic_margin_1_2": margin_1,
            "deterministic_margin_10_11": margin_10,
            "margin_gate_threshold": margin_threshold,
            "margin_gate_selected": margin_mask,
            "top10_jaccard_disagreement": disagreement,
            "disagreement_gate_threshold": disagreement_threshold,
            "disagreement_gate_selected": disagreement_mask,
            "deterministic_centroid_cosine": centroid_cosine,
        }
    )
    if "mean_embedding" in requested:
        rankings["mean_embedding"] = retriever.search(
            mean_embedding(stochastic_queries)
        )
    if "mean_score" in requested:
        rankings["mean_score"] = retriever.search(stochastic_queries.mean(axis=0))
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
    return (
        {method: ranking for method, ranking in rankings.items() if method in requested},
        trimmed_diagnostics,
        selections,
        gate_diagnostics,
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
    expected_ids: list[str] | None = None,
) -> dict[str, Any]:
    if output_path.exists() and ids_path.exists():
        with ids_path.open(encoding="utf-8") as handle:
            cached_ids = [json.loads(line) for line in handle]
        if expected_ids is not None and cached_ids != expected_ids:
            raise RuntimeError(
                f"Cached IDs or ordering do not match the dataset: {ids_path}"
            )
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
    report = encoder.encode_to_file(
        records=records,
        count=count,
        output_path=output_path,
        ids_path=ids_path,
        prefix=prefix,
        seed=seed,
        stochastic=stochastic,
        description=description,
    )
    if expected_ids is not None:
        with ids_path.open(encoding="utf-8") as handle:
            written_ids = [json.loads(line) for line in handle]
        if written_ids != expected_ids:
            raise RuntimeError(f"Encoded IDs or ordering are invalid: {ids_path}")
    return report


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


def _save_sample_rankings(
    store: ArtifactStore,
    rankings: list[Rankings],
    sample_count: int,
) -> None:
    directory = store.ranking_dir / "samples" / f"n_{sample_count:03d}"
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
