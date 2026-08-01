from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from stochastic_retrieval.artifacts import ArtifactStore, file_sha256
from stochastic_retrieval.config import ProjectConfig
from stochastic_retrieval.evaluation import ndcg


def audit_nfcorpus_sweep(
    configs: list[ProjectConfig],
    project_root: Path,
) -> pd.DataFrame:
    """Independently validate completed sweep artifacts and metric provenance."""
    checks: list[dict[str, object]] = []
    baseline_reference: pd.DataFrame | None = None
    cache_keys: set[str] = set()
    document_id_hashes: set[str] = set()
    for config in configs:
        store = ArtifactStore(project_root, config)
        run_id = store.run_id
        manifest = json.loads(
            (store.run_dir / "manifest.json").read_text(encoding="utf-8")
        )
        _record(checks, run_id, "manifest_completed", manifest["status"] == "completed")
        cache_keys.add(manifest["document_cache"]["cache_key"])
        document_id_hashes.add(
            manifest["artifacts"]["embeddings/documents/ids.jsonl"]["sha256"]
        )
        for relative, expected in manifest["artifacts"].items():
            path = store.run_dir / relative
            _record(
                checks,
                run_id,
                f"artifact_hash:{relative}",
                path.exists()
                and path.stat().st_size == expected["bytes"]
                and file_sha256(path) == expected["sha256"],
            )

        query_ids = _read_jsonl(
            store.run_dir / "embeddings/queries_deterministic/ids.jsonl"
        )
        document_ids = _read_jsonl(store.run_dir / "embeddings/documents/ids.jsonl")
        qrels = json.loads((store.run_dir / "qrels.json").read_text(encoding="utf-8"))
        method_count = len(config.experiment.methods)
        sample_count = config.experiment.resolved_sample_counts[0]
        per_query = pd.read_parquet(store.run_dir / "metrics/per_query.parquet")
        summary = pd.read_parquet(store.run_dir / "metrics/summary.parquet")
        bootstrap = pd.read_parquet(store.run_dir / "metrics/paired_bootstrap.parquet")
        _record(
            checks,
            run_id,
            "per_query_row_cardinality",
            len(per_query) == len(query_ids) * method_count,
        )
        _record(
            checks,
            run_id,
            "summary_row_cardinality",
            len(summary) == method_count,
        )
        ranking_rows = pq.ParquetFile(
            store.run_dir / "rankings/aggregated_rankings.parquet"
        ).metadata.num_rows
        _record(
            checks,
            run_id,
            "ranking_row_cardinality",
            ranking_rows
            == len(query_ids) * method_count * config.experiment.retrieval_k,
        )
        seeds = manifest["sample_count_protocol"]["seeds"][str(sample_count)]
        _record(
            checks,
            run_id,
            "sample_seed_uniqueness",
            len(seeds) == sample_count and len(set(seeds)) == sample_count,
        )
        _record(
            checks,
            run_id,
            "bootstrap_within_sample_count",
            set(bootstrap["sample_count"]) == {sample_count},
        )
        numeric_metrics = [
            column
            for column in per_query.select_dtypes(include="number").columns
            if column not in {"sample_count"}
        ]
        recomputed = (
            per_query.groupby(["method", "sample_count"])[numeric_metrics]
            .mean()
            .reset_index()
        )
        merged = summary.merge(
            recomputed,
            on=["method", "sample_count"],
            suffixes=("_saved", "_recomputed"),
        )
        summary_matches = all(
            np.allclose(
                merged[f"{metric}_saved"],
                merged[f"{metric}_recomputed"],
                atol=1e-12,
                equal_nan=True,
            )
            for metric in numeric_metrics
            if f"{metric}_saved" in merged
        )
        _record(checks, run_id, "summary_means_recomputed", summary_matches)

        baseline = (
            per_query[per_query["method"] == "deterministic"]
            .sort_values("query_id")
            .reset_index(drop=True)
        )
        if baseline_reference is None:
            baseline_reference = baseline
        else:
            _record(
                checks,
                run_id,
                "deterministic_baseline_invariant",
                baseline.equals(baseline_reference),
            )

        selections = pd.read_parquet(
            store.run_dir / "analyses/oracle_selections.parquet"
        )
        _record(
            checks,
            run_id,
            "oracle_candidate_source",
            set(selections["candidate_source"]) == {"stochastic_only"}
            and selections["selected_sample"].between(0, sample_count - 1).all(),
        )
        oracle_metrics = per_query[per_query["method"] == "oracle_best_of_n"].set_index(
            "query_id"
        )
        ranking_validity = True
        sample_directory = (
            store.run_dir / f"rankings/samples/n_{sample_count:03d}"
        )
        sample_paths = sorted(sample_directory.glob("sample_*.npz"))
        _record(
            checks,
            run_id,
            "sample_ranking_file_count",
            len(sample_paths) == sample_count,
        )
        oracle_utilities = np.empty(
            (sample_count, len(query_ids)),
            dtype=np.float64,
        )
        for sample_index, path in enumerate(sample_paths):
            with np.load(path) as data:
                indices = data["indices"]
                scores = data["scores"]
                ranking_validity &= bool(np.all(scores[:, :-1] >= scores[:, 1:]))
                ranking_validity &= all(
                    len(np.unique(row)) == len(row) for row in indices
                )
                for query_index, query_id in enumerate(query_ids):
                    ranked_ids = [
                        document_ids[int(index)]
                        for index in indices[query_index, :10]
                    ]
                    oracle_utilities[sample_index, query_index] = ndcg(
                        ranked_ids,
                        qrels[query_id],
                        10,
                    )
        _record(checks, run_id, "all_sample_rankings_valid", ranking_validity)
        expected_samples = np.argmax(oracle_utilities, axis=0)
        selection_by_query = selections.set_index("query_id").loc[query_ids]
        expected_utilities = oracle_utilities[
            expected_samples,
            np.arange(len(query_ids)),
        ]
        oracle_match = (
            np.array_equal(
                selection_by_query["selected_sample"].to_numpy(),
                expected_samples,
            )
            and np.allclose(
                selection_by_query["selection_ndcg@10"],
                expected_utilities,
                atol=1e-12,
            )
            and np.allclose(
                oracle_metrics.loc[query_ids, "ndcg@10"],
                expected_utilities,
                atol=1e-12,
            )
        )
        _record(checks, run_id, "all_oracle_argmax_metrics_recomputed", oracle_match)

    _record(
        checks,
        "sweep",
        "document_cache_identity",
        len(cache_keys) == 1 and len(document_id_hashes) == 1,
    )
    return pd.DataFrame(checks)


def write_nfcorpus_audit(
    configs: list[ProjectConfig],
    project_root: Path,
) -> tuple[pd.DataFrame, Path]:
    audit = audit_nfcorpus_sweep(configs, project_root)
    reports = project_root / configs[0].artifact_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    path = reports / "nfcorpus_sweep_audit.parquet"
    audit.to_parquet(path, index=False)
    return audit, path


def _record(
    checks: list[dict[str, object]],
    run_id: str,
    check: str,
    passed: bool,
) -> None:
    checks.append({"run_id": run_id, "check": check, "passed": bool(passed)})


def _read_jsonl(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]
