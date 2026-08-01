from __future__ import annotations

import hashlib
import json
import platform
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


class ArtifactStore:
    def __init__(self, project_root: Path, config: ProjectConfig) -> None:
        root = Path(config.artifact_root)
        self.root = root if root.is_absolute() else project_root / root
        self.run_id = f"{config.experiment.name}-{config.fingerprint}"
        self.run_dir = self.root / "runs" / self.run_id
        self.embedding_dir = self.run_dir / "embeddings"
        self.ranking_dir = self.run_dir / "rankings"
        self.metric_dir = self.run_dir / "metrics"
        for directory in (
            self.embedding_dir,
            self.ranking_dir,
            self.metric_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def embedding_path(self, kind: str, sample: int) -> Path:
        return self.embedding_dir / kind / f"sample_{sample:03d}.npy"

    def ids_path(self, kind: str) -> Path:
        return self.embedding_dir / kind / "ids.jsonl"

    def load_embeddings(self, kind: str, sample: int) -> np.ndarray:
        return np.load(self.embedding_path(kind, sample), mmap_mode="r")

    def load_ids(self, kind: str) -> list[str]:
        with self.ids_path(kind).open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle]

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
        embedding_files = sorted(self.embedding_dir.glob("**/*.npy"))
        manifest = {
            "schema_version": 1,
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
                for path in embedding_files
            },
            **(extra or {}),
        }
        path = self.run_dir / "manifest.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
        return path
