from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from stochastic_retrieval.artifacts import file_sha256
from stochastic_retrieval.evaluation import (
    evaluate_rankings,
    oracle_best_of_n,
    paired_bootstrap_comparisons,
    summarize,
)
from stochastic_retrieval.retrieval import DenseRetriever, Rankings, l2_normalize


def matched_noise_controls(
    deterministic: np.ndarray,
    stochastic: np.ndarray,
    query_ids: list[str],
    seed: int,
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """Generate full-covariance and isotropic controls with exact dropout angles.

    The final embeddings are unit normalized. Each artificial sample receives
    the same query-relative angular displacement (and therefore the same chord
    distance) as the corresponding dropout sample. The controls differ only in
    perturbation direction.
    """
    if stochastic.ndim != 3:
        raise ValueError("stochastic embeddings must have shape [samples, queries, dim]")
    sample_count, query_count, dimension = stochastic.shape
    if deterministic.shape != (query_count, dimension):
        raise ValueError("deterministic embeddings do not align with stochastic samples")
    if len(query_ids) != query_count:
        raise ValueError("query_ids do not align with embeddings")

    deterministic = l2_normalize(deterministic)
    stochastic = l2_normalize(stochastic)
    dropout_cosines = np.clip(
        np.einsum("sqd,qd->sq", stochastic, deterministic),
        -1.0,
        1.0,
    )
    dropout_angles = np.arccos(dropout_cosines)
    rng = np.random.default_rng(seed)
    gaussian = np.empty_like(stochastic)
    isotropic = np.empty_like(stochastic)
    diagnostic_rows: list[dict[str, float | int | str | bool]] = []

    for query_index, query_id in enumerate(query_ids):
        reference = deterministic[query_index]
        sample_vectors = stochastic[:, query_index, :]
        tangent = _tangent_components(sample_vectors, reference)
        tangent_mean = tangent.mean(axis=0)
        centered = tangent - tangent_mean
        covariance_estimable = sample_count > 1 and bool(np.any(centered))

        gaussian_directions = np.empty((sample_count, dimension), dtype=np.float32)
        isotropic_directions = np.empty_like(gaussian_directions)
        for sample_index in range(sample_count):
            if covariance_estimable:
                coefficients = rng.normal(size=sample_count)
                gaussian_raw = tangent_mean + (
                    coefficients @ centered / np.sqrt(sample_count - 1)
                )
            else:
                gaussian_raw = rng.normal(size=dimension)
            gaussian_directions[sample_index] = _unit_tangent(
                gaussian_raw,
                reference,
                rng,
            )
            isotropic_directions[sample_index] = _unit_tangent(
                rng.normal(size=dimension),
                reference,
                rng,
            )

        angles = dropout_angles[:, query_index]
        gaussian[:, query_index, :] = _place_at_angles(
            reference,
            gaussian_directions,
            angles,
        )
        isotropic[:, query_index, :] = _place_at_angles(
            reference,
            isotropic_directions,
            angles,
        )

        for control_type, control in (
            ("full_covariance_gaussian", gaussian[:, query_index, :]),
            ("isotropic_angular_matched", isotropic[:, query_index, :]),
        ):
            control_angles = np.arccos(
                np.clip(control @ reference, -1.0, 1.0)
            )
            control_tangent = _tangent_components(control, reference)
            diagnostic_rows.append(
                {
                    "query_id": query_id,
                    "control_type": control_type,
                    "sample_count": sample_count,
                    "covariance_estimable": covariance_estimable,
                    "dropout_angle_mean": float(angles.mean()),
                    "dropout_angle_std": float(angles.std()),
                    "dropout_angle_variance": float(angles.var()),
                    "dropout_chord_mean": float(
                        np.linalg.norm(sample_vectors - reference, axis=1).mean()
                    ),
                    "control_angle_mean": float(control_angles.mean()),
                    "control_angle_std": float(control_angles.std()),
                    "maximum_angle_mismatch": float(
                        np.max(np.abs(control_angles - angles))
                    ),
                    "control_chord_mean": float(
                        np.linalg.norm(control - reference, axis=1).mean()
                    ),
                    "tangent_covariance_alignment": _covariance_alignment(
                        tangent,
                        control_tangent,
                    ),
                    "tangent_mean_alignment": _vector_alignment(
                        tangent.mean(axis=0),
                        control_tangent.mean(axis=0),
                    ),
                }
            )

    return (
        {
            "full_covariance_gaussian_oracle": l2_normalize(gaussian),
            "isotropic_noise_oracle": l2_normalize(isotropic),
        },
        pd.DataFrame(diagnostic_rows),
    )


def evaluate_matched_noise_oracles(
    deterministic_embeddings: np.ndarray,
    stochastic_embeddings: np.ndarray,
    deterministic_ranking: Rankings,
    dropout_rankings: list[Rankings],
    retriever: object,
    query_ids: list[str],
    document_ids: list[str],
    qrels: dict[str, dict[str, int]],
    metric_cutoffs: tuple[int, ...],
    success_cutoffs: tuple[int, ...],
    sample_count: int,
    seed: int,
    selection_cutoff: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate dropout and magnitude-matched artificial best-of-N controls."""
    controls, diagnostics = matched_noise_controls(
        deterministic_embeddings,
        stochastic_embeddings,
        query_ids,
        seed,
    )
    rankings: dict[str, Rankings] = {"deterministic": deterministic_ranking}
    selections = []
    dropout_oracle, dropout_selections = oracle_best_of_n(
        dropout_rankings,
        query_ids,
        document_ids,
        qrels,
        selection_cutoff=selection_cutoff,
        candidate_source="dropout_stochastic",
    )
    rankings["dropout_oracle"] = dropout_oracle
    dropout_selections.insert(0, "method", "dropout_oracle")
    selections.append(dropout_selections)

    for method, embeddings in controls.items():
        sample_rankings = [
            retriever.search(embeddings[sample])
            for sample in range(sample_count)
        ]
        oracle, method_selections = oracle_best_of_n(
            sample_rankings,
            query_ids,
            document_ids,
            qrels,
            selection_cutoff=selection_cutoff,
            candidate_source=method,
        )
        rankings[method] = oracle
        method_selections.insert(0, "method", method)
        selections.append(method_selections)

    frames = []
    for method, ranking in rankings.items():
        frame = evaluate_rankings(
            method,
            ranking,
            query_ids,
            document_ids,
            qrels,
            metric_cutoffs,
            success_cutoffs=success_cutoffs,
        )
        frame.insert(1, "sample_count", sample_count)
        frames.append(frame)
    selection_frame = pd.concat(selections, ignore_index=True)
    selection_frame.insert(1, "sample_count", sample_count)
    return pd.concat(frames, ignore_index=True), selection_frame, diagnostics


def analyze_saved_noise_oracles(
    run_dir: Path,
    sample_counts: tuple[int, ...] | None = None,
) -> list[Path]:
    """Run matched-noise oracle controls from saved embeddings and rankings."""
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = manifest["config"]
    experiment = config["experiment"]
    available_counts = tuple(experiment["sample_counts"])
    selected_counts = sample_counts or available_counts
    unknown = set(selected_counts) - set(available_counts)
    if unknown:
        raise ValueError(f"Sample counts not present in run: {sorted(unknown)}")

    query_ids = _read_jsonl(run_dir / "embeddings/queries_deterministic/ids.jsonl")
    document_ids = _read_jsonl(run_dir / "embeddings/documents/ids.jsonl")
    qrels = json.loads((run_dir / "qrels.json").read_text(encoding="utf-8"))
    deterministic_embeddings = np.load(
        run_dir / "embeddings/queries_deterministic/sample_000.npy",
        mmap_mode="r",
    )
    documents = np.load(
        run_dir / "embeddings/documents/sample_000.npy",
        mmap_mode="r",
    )
    retriever = DenseRetriever(
        documents,
        experiment["retrieval_k"],
        backend=experiment["retrieval_backend"],
        query_batch_size=experiment["retrieval_query_batch_size"],
        corpus_batch_size=experiment["retrieval_corpus_batch_size"],
    )
    deterministic_ranking = retriever.search(deterministic_embeddings)
    per_query_parts = []
    selection_parts = []
    diagnostic_parts = []
    for sample_count in selected_counts:
        sample_directory = (
            run_dir / f"embeddings/queries_stochastic/n_{sample_count:03d}"
        )
        stochastic_embeddings = np.stack(
            [
                np.load(sample_directory / f"sample_{sample:03d}.npy")
                for sample in range(sample_count)
            ]
        )
        ranking_directory = run_dir / f"rankings/samples/n_{sample_count:03d}"
        dropout_rankings = []
        for sample in range(sample_count):
            with np.load(ranking_directory / f"sample_{sample:03d}.npz") as data:
                dropout_rankings.append(
                    Rankings(data["indices"].copy(), data["scores"].copy())
                )
        per_query, selections, diagnostics = evaluate_matched_noise_oracles(
            deterministic_embeddings,
            stochastic_embeddings,
            deterministic_ranking,
            dropout_rankings,
            retriever,
            query_ids,
            document_ids,
            qrels,
            metric_cutoffs=tuple(experiment["metric_cutoffs"]),
            success_cutoffs=tuple(experiment["success_cutoffs"]),
            sample_count=sample_count,
            seed=experiment["seed"] + 900_000 + sample_count,
            selection_cutoff=10,
        )
        if diagnostics["maximum_angle_mismatch"].max() > 1e-5:
            raise RuntimeError("Artificial noise does not match dropout angles")
        per_query_parts.append(per_query)
        selection_parts.append(selections)
        diagnostic_parts.append(diagnostics)

    per_query = pd.concat(per_query_parts, ignore_index=True)
    selections = pd.concat(selection_parts, ignore_index=True)
    diagnostics = pd.concat(diagnostic_parts, ignore_index=True)
    summary = summarize(per_query)
    bootstrap_parts = []
    for baseline in ("deterministic", "dropout_oracle"):
        comparison = paired_bootstrap_comparisons(
            per_query,
            baseline=baseline,
            metric="ndcg@10",
            replicates=experiment["bootstrap_replicates"],
            seed=experiment["seed"],
        )
        comparison.insert(0, "comparison", f"versus_{baseline}")
        bootstrap_parts.append(comparison)
    bootstrap = pd.concat(bootstrap_parts, ignore_index=True)

    outputs = [
        _write_parquet(
            run_dir,
            per_query,
            "metrics/noise_oracle_per_query.parquet",
            manifest,
        ),
        _write_parquet(
            run_dir,
            summary,
            "metrics/noise_oracle_summary.parquet",
            manifest,
        ),
        _write_parquet(
            run_dir,
            bootstrap,
            "metrics/noise_oracle_bootstrap.parquet",
            manifest,
        ),
        _write_parquet(
            run_dir,
            selections,
            "analyses/noise_oracle_selections.parquet",
            manifest,
        ),
        _write_parquet(
            run_dir,
            diagnostics,
            "analyses/noise_control_diagnostics.parquet",
            manifest,
        ),
    ]
    manifest["matched_noise_oracle"] = {
        "sample_counts": list(selected_counts),
        "seed_offset": 900_000,
        "selection_metric": "linear_gain_ndcg@10",
        "controls": {
            "full_covariance_gaussian_oracle": (
                "Empirical tangent mean/covariance directions with exact "
                "per-sample dropout angular displacement"
            ),
            "isotropic_noise_oracle": (
                "Uniform tangent directions with exact per-sample dropout "
                "angular displacement"
            ),
        },
        "magnitude_contract": (
            "Every artificial sample exactly matches its paired dropout angle "
            "and unit-sphere chord distance from deterministic"
        ),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return outputs


def _tangent_components(vectors: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return vectors - np.outer(vectors @ reference, reference)


def _unit_tangent(
    vector: np.ndarray,
    reference: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    tangent = vector - float(vector @ reference) * reference
    norm = float(np.linalg.norm(tangent))
    while norm <= 1e-12:
        tangent = rng.normal(size=len(reference))
        tangent -= float(tangent @ reference) * reference
        norm = float(np.linalg.norm(tangent))
    return tangent / norm


def _place_at_angles(
    reference: np.ndarray,
    directions: np.ndarray,
    angles: np.ndarray,
) -> np.ndarray:
    return (
        np.cos(angles)[:, None] * reference
        + np.sin(angles)[:, None] * directions
    )


def _covariance_alignment(left: np.ndarray, right: np.ndarray) -> float:
    left = left - left.mean(axis=0, keepdims=True)
    right = right - right.mean(axis=0, keepdims=True)
    left_norm = float(np.square(left @ left.T).sum())
    right_norm = float(np.square(right @ right.T).sum())
    if left_norm <= 1e-24 or right_norm <= 1e-24:
        return 0.0
    cross = float(np.square(left @ right.T).sum())
    return cross / np.sqrt(left_norm * right_norm)


def _vector_alignment(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-24:
        return 0.0
    return float(left @ right) / denominator


def _read_jsonl(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _write_parquet(
    run_dir: Path,
    frame: pd.DataFrame,
    relative_path: str,
    manifest: dict[str, object],
) -> Path:
    path = run_dir / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    manifest.setdefault("artifacts", {})[relative_path] = {
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }
    return path
