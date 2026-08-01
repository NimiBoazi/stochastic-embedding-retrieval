from __future__ import annotations

import json
import random
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
from numpy.lib.format import open_memmap
from sentence_transformers import SentenceTransformer
from torch import nn
from tqdm import tqdm

from stochastic_retrieval.config import ModelConfig
from stochastic_retrieval.data import TextRecord


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_dropout(
    model: nn.Module,
    stochastic: bool,
    scope: str = "all",
    probability: float | None = None,
) -> int:
    """Enable only selected dropout modules while keeping the model otherwise in eval mode."""
    valid_scopes = {"all", "attention", "hidden"}
    if scope not in valid_scopes:
        raise ValueError(f"dropout_scope must be one of {sorted(valid_scopes)}")

    model.eval()
    if not stochastic:
        return 0

    enabled = 0
    for name, module in model.named_modules():
        if not isinstance(module, nn.Dropout):
            continue
        is_attention = "attention" in name.lower() or "attn" in name.lower()
        selected = (
            scope == "all"
            or (scope == "attention" and is_attention)
            or (scope == "hidden" and not is_attention)
        )
        if selected:
            if probability is not None:
                if not 0 <= probability < 1:
                    raise ValueError("dropout_probability must be in [0, 1)")
                module.p = probability
            module.train()
            enabled += 1
    if enabled == 0:
        raise RuntimeError(f"No dropout modules matched scope '{scope}'")
    return enabled


class SentenceEmbeddingEncoder:
    def __init__(self, config: ModelConfig, device: str) -> None:
        self.config = config
        self.device = resolve_device(device)
        self.model = SentenceTransformer(
            config.name,
            revision=config.revision,
            device=self.device,
        )
        self.model.max_seq_length = config.max_length
        dimension = self.model.get_sentence_embedding_dimension()
        if dimension is None:
            raise RuntimeError("Could not determine sentence embedding dimension")
        self.dimension = int(dimension)

    def encode_to_file(
        self,
        records: Iterable[TextRecord],
        count: int,
        output_path: Path,
        ids_path: Path,
        prefix: str,
        seed: int,
        stochastic: bool,
        description: str,
    ) -> dict[str, float | int | bool]:
        set_reproducible_seed(seed)
        enabled_dropout_modules = configure_dropout(
            self.model,
            stochastic=stochastic,
            scope=self.config.dropout_scope,
            probability=self.config.dropout_probability,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        embeddings = open_memmap(
            output_path,
            mode="w+",
            dtype=np.float32,
            shape=(count, self.dimension),
        )

        index = 0
        with ids_path.open("w", encoding="utf-8") as ids_file:
            batch: list[TextRecord] = []
            iterator = tqdm(records, total=count, desc=description, unit="items")
            for record in iterator:
                batch.append(record)
                if len(batch) == self.config.batch_size:
                    index = self._write_batch(batch, prefix, embeddings, ids_file, index)
                    batch.clear()
            if batch:
                index = self._write_batch(batch, prefix, embeddings, ids_file, index)

        embeddings.flush()
        if index != count:
            output_path.unlink(missing_ok=True)
            ids_path.unlink(missing_ok=True)
            raise RuntimeError(f"Expected {count} records but encoded {index}")
        health = inspect_embedding_file(output_path, expected_count=count)
        return {
            **health,
            "cached": False,
            "seed": seed,
            "stochastic": stochastic,
            "enabled_dropout_modules": enabled_dropout_modules,
        }

    def _write_batch(
        self,
        batch: list[TextRecord],
        prefix: str,
        destination: np.memmap,
        ids_file: object,
        start: int,
    ) -> int:
        texts = [f"{prefix}{record.text}" for record in batch]
        features = self.model.tokenize(texts)
        features = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in features.items()
        }
        with torch.inference_mode():
            output = self.model(features)["sentence_embedding"]
            if self.config.normalize:
                output = torch.nn.functional.normalize(output, p=2, dim=1)
        if not torch.isfinite(output).all():
            raise RuntimeError(f"Non-finite embedding detected near item {start}")
        if torch.linalg.vector_norm(output, dim=1).min().item() <= 1e-8:
            raise RuntimeError(f"Near-zero embedding detected near item {start}")
        array = output.detach().cpu().float().numpy()
        stop = start + len(batch)
        destination[start:stop] = array
        for record in batch:
            ids_file.write(json.dumps(record.item_id) + "\n")  # type: ignore[attr-defined]
        return stop


def inspect_embedding_file(
    path: Path,
    expected_count: int,
    chunk_size: int = 50_000,
) -> dict[str, float | int]:
    embeddings = np.load(path, mmap_mode="r")
    if embeddings.ndim != 2 or len(embeddings) != expected_count:
        raise RuntimeError(
            f"Invalid embedding shape {embeddings.shape}; expected {expected_count} rows"
        )

    minimum_norm = float("inf")
    maximum_norm = 0.0
    norm_sum = 0.0
    for start in range(0, len(embeddings), chunk_size):
        chunk = np.asarray(embeddings[start : start + chunk_size])
        if not np.isfinite(chunk).all():
            raise RuntimeError(f"Non-finite values found in {path} near row {start}")
        norms = np.linalg.norm(chunk, axis=1)
        if len(norms):
            minimum_norm = min(minimum_norm, float(norms.min()))
            maximum_norm = max(maximum_norm, float(norms.max()))
            norm_sum += float(norms.sum())
    if minimum_norm <= 1e-8:
        raise RuntimeError(f"Near-zero embedding norm found in {path}")
    return {
        "items": len(embeddings),
        "dimension": int(embeddings.shape[1]),
        "minimum_norm": minimum_norm,
        "maximum_norm": maximum_norm,
        "mean_norm": norm_sum / max(len(embeddings), 1),
    }
