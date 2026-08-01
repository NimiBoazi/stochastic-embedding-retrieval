from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
from collections.abc import Iterable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from stochastic_retrieval.config import ProjectConfig


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def document_encoding_contract(config: ProjectConfig) -> dict[str, Any]:
    """Inputs that can change deterministic document embeddings or row order."""
    return {
        "model": {
            "name": config.model.name,
            "revision": config.model.revision,
            "document_prefix": config.model.document_prefix,
            "pooling": config.model.pooling,
            "expected_dimension": config.model.expected_dimension,
            "max_length": config.model.max_length,
            "normalize": config.model.normalize,
        },
        "dataset": {
            "ir_dataset_id": config.dataset.ir_dataset_id,
            "document_limit": config.dataset.document_limit,
        },
        "attention_implementation": "eager",
        "dtype": "float32",
    }


def document_cache_key(config: ProjectConfig) -> str:
    payload = json.dumps(
        document_encoding_contract(config), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


class ArtifactStore:
    def __init__(self, project_root: Path, config: ProjectConfig) -> None:
        root = Path(config.artifact_root)
        self.root = root if root.is_absolute() else project_root / root
        self.run_id = f"{config.experiment.name}-{config.fingerprint}"
        self.run_dir = self.root / "runs" / self.run_id
        self.embedding_dir = self.run_dir / "embeddings"
        self.ranking_dir = self.run_dir / "rankings"
        self.metric_dir = self.run_dir / "metrics"
        self.document_cache_dir = (
            self.root / "cache" / "documents" / document_cache_key(config)
        )
        for directory in (
            self.embedding_dir,
            self.ranking_dir,
            self.metric_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def embedding_path(
        self, kind: str, sample: int, sample_count: int | None = None
    ) -> Path:
        directory = self.embedding_dir / kind
        if sample_count is not None:
            directory = directory / f"n_{sample_count:03d}"
        return directory / f"sample_{sample:03d}.npy"

    def ids_path(self, kind: str, sample_count: int | None = None) -> Path:
        directory = self.embedding_dir / kind
        if sample_count is not None:
            directory = directory / f"n_{sample_count:03d}"
        return directory / "ids.jsonl"

    def load_embeddings(
        self, kind: str, sample: int, sample_count: int | None = None
    ) -> np.ndarray:
        return np.load(
            self.embedding_path(kind, sample, sample_count), mmap_mode="r"
        )

    def load_ids(self, kind: str, sample_count: int | None = None) -> list[str]:
        with self.ids_path(kind, sample_count).open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle]

    @property
    def cached_document_embeddings(self) -> Path:
        return self.document_cache_dir / "sample_000.npy"

    @property
    def cached_document_ids(self) -> Path:
        return self.document_cache_dir / "ids.jsonl"

    @property
    def cached_document_metadata(self) -> Path:
        return self.document_cache_dir / "metadata.json"

    def promote_compatible_document_cache(
        self, config: ProjectConfig
    ) -> dict[str, Any] | None:
        """Populate the shared cache from a compatible completed run if available."""
        if self.cached_document_embeddings.exists() or self.cached_document_ids.exists():
            if not (
                self.cached_document_embeddings.exists()
                and self.cached_document_ids.exists()
            ):
                raise RuntimeError(
                    f"Incomplete document cache at {self.document_cache_dir}"
                )
            return {"source": "shared_cache", "cache_key": document_cache_key(config)}

        contract = document_encoding_contract(config)
        for manifest_path in sorted((self.root / "runs").glob("*/manifest.json")):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                prior = manifest["config"]
                prior_contract = {
                    "model": {
                        key: prior["model"].get(key)
                        for key in contract["model"]
                    },
                    "dataset": {
                        key: prior["dataset"].get(key)
                        for key in contract["dataset"]
                    },
                    "attention_implementation": manifest.get(
                        "model_provenance", {}
                    ).get("attention_implementation"),
                    "dtype": "float32",
                }
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
            if manifest.get("status") != "completed" or prior_contract != contract:
                continue
            run_dir = manifest_path.parent
            source_embeddings = (
                run_dir / "embeddings" / "documents" / "sample_000.npy"
            )
            source_ids = run_dir / "embeddings" / "documents" / "ids.jsonl"
            if not source_embeddings.exists() or not source_ids.exists():
                continue
            self._copy_document_cache(
                source_embeddings,
                source_ids,
                contract,
                source=str(run_dir.name),
            )
            return {
                "source": "promoted_run",
                "source_run": run_dir.name,
                "cache_key": document_cache_key(config),
            }
        return None

    def write_document_cache_metadata(
        self, config: ProjectConfig, source: str
    ) -> None:
        self.document_cache_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "cache_key": document_cache_key(config),
            "contract": document_encoding_contract(config),
            "source": source,
            "embeddings_sha256": file_sha256(self.cached_document_embeddings),
            "ids_sha256": file_sha256(self.cached_document_ids),
        }
        self.cached_document_metadata.write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )

    def materialize_documents_in_run(self) -> None:
        """Hard-link validated cache files into the run, copying if required."""
        output_path = self.embedding_path("documents", 0)
        ids_path = self.ids_path("documents")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        for source, destination in (
            (self.cached_document_embeddings, output_path),
            (self.cached_document_ids, ids_path),
        ):
            if destination.exists():
                destination.unlink()
            try:
                os.link(source, destination)
            except OSError:
                shutil.copy2(source, destination)

    def _copy_document_cache(
        self,
        source_embeddings: Path,
        source_ids: Path,
        contract: dict[str, Any],
        source: str,
    ) -> None:
        self.document_cache_dir.mkdir(parents=True, exist_ok=True)
        temporary_embeddings = self.cached_document_embeddings.with_suffix(".npy.tmp")
        temporary_ids = self.cached_document_ids.with_suffix(".jsonl.tmp")
        shutil.copy2(source_embeddings, temporary_embeddings)
        shutil.copy2(source_ids, temporary_ids)
        temporary_embeddings.replace(self.cached_document_embeddings)
        temporary_ids.replace(self.cached_document_ids)
        metadata = {
            "cache_key": self.document_cache_dir.name,
            "contract": contract,
            "source": source,
            "embeddings_sha256": file_sha256(self.cached_document_embeddings),
            "ids_sha256": file_sha256(self.cached_document_ids),
        }
        self.cached_document_metadata.write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )

    def write_dataframe(self, frame: pd.DataFrame, category: str, name: str) -> Path:
        directory = self.run_dir / category
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        return path

    def write_dataframe_chunks(
        self,
        frames: Iterable[pd.DataFrame],
        category: str,
        name: str,
    ) -> Path:
        directory = self.run_dir / category
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{name}.parquet"
        writer: pq.ParquetWriter | None = None
        try:
            for frame in frames:
                if frame.empty:
                    continue
                table = pa.Table.from_pandas(frame, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(path, table.schema, compression="zstd")
                writer.write_table(table)
        finally:
            if writer is not None:
                writer.close()
        if writer is None:
            raise ValueError(f"No rows were produced for {category}/{name}")
        return path

    def write_manifest(
        self,
        config: ProjectConfig,
        project_root: Path,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        artifact_files = sorted(
            path
            for path in self.run_dir.glob("**/*")
            if path.is_file() and path.name not in {"manifest.json", "events.jsonl"}
        )
        manifest = {
            "schema_version": 2,
            "run_id": self.run_id,
            "created_at": datetime.now(UTC).isoformat(),
            "config": asdict(config),
            "config_fingerprint": config.fingerprint,
            "git_revision": git_revision(project_root),
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
            },
            "artifacts": {
                str(path.relative_to(self.run_dir)): {
                    "sha256": file_sha256(path),
                    "bytes": path.stat().st_size,
                }
                for path in artifact_files
            },
            **(extra or {}),
        }
        path = self.run_dir / "manifest.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
        return path
