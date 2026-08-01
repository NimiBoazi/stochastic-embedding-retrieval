from __future__ import annotations

import json
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.lib.format import open_memmap
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.modules import Pooling
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


class _DropoutSite:
    """One dropout application point, with the module whose training flag gates it.

    Modern HuggingFace encoders apply dropout in two distinct ways:

    - Hidden-state dropout (attention-output, feed-forward, embeddings) calls an
      ``nn.Dropout`` module directly, so that module's own training flag controls it.
    - Attention-probability dropout is applied functionally inside the attention
      forward, gated by the *attention module's* training flag. BERT-family models
      keep an ``nn.Dropout`` child at ``...attention.self.dropout`` purely as a
      probability holder; T5-family models store the probability as a plain float
      on the attention module and have no ``nn.Dropout`` child at all.

    Setting only the ``nn.Dropout`` child to train mode therefore silently
    disables attention-probability dropout; the gate module must be flipped.
    """

    def __init__(
        self,
        name: str,
        gate: nn.Module,
        holder: nn.Dropout | None,
        is_attention_probability: bool,
    ) -> None:
        self.name = name
        self.gate = gate
        self.holder = holder
        self.is_attention_probability = is_attention_probability

    @property
    def probability(self) -> float:
        if self.holder is not None:
            return float(self.holder.p)
        return float(self.gate.dropout)

    def set_probability(self, probability: float) -> None:
        if self.holder is not None:
            self.holder.p = probability
        else:
            self.gate.dropout = probability

    def enable(self) -> None:
        self.gate.train()


def _dropout_sites(model: nn.Module) -> list[_DropoutSite]:
    modules = dict(model.named_modules())
    sites: list[_DropoutSite] = []
    for name, module in modules.items():
        if isinstance(module, nn.Dropout):
            if name.endswith("attention.self.dropout"):
                # BERT-family attention-probability dropout: gated by the parent
                # self-attention module, probability held by this nn.Dropout.
                parent = modules[name.rsplit(".", 1)[0]]
                sites.append(_DropoutSite(name, parent, module, True))
            else:
                sites.append(_DropoutSite(name, module, module, False))
        elif type(module).__name__.endswith("Attention") and isinstance(
            getattr(module, "dropout", None), float
        ):
            # T5-family attention-probability dropout: applied functionally with a
            # float probability attribute; no nn.Dropout child exists.
            sites.append(_DropoutSite(name, module, None, True))
    return sites


def _site_selected(site: _DropoutSite, scope: str) -> bool:
    return (
        scope == "all"
        or (scope == "attention" and site.is_attention_probability)
        or (scope == "hidden" and not site.is_attention_probability)
    )


def configure_dropout(
    model: nn.Module,
    stochastic: bool,
    scope: str = "all",
    probability: float | None = None,
) -> int:
    """Enable only selected dropout sites while keeping the model otherwise in eval mode."""
    valid_scopes = {"all", "attention", "hidden"}
    if scope not in valid_scopes:
        raise ValueError(f"dropout_scope must be one of {sorted(valid_scopes)}")

    model.eval()
    if not stochastic:
        return 0

    enabled = 0
    for site in _dropout_sites(model):
        if not _site_selected(site, scope):
            continue
        if probability is not None:
            if not 0 <= probability < 1:
                raise ValueError("dropout_probability must be in [0, 1)")
            site.set_probability(probability)
        site.enable()
        enabled += int(site.probability > 0)
    if enabled == 0:
        raise RuntimeError(
            f"No positive-probability dropout sites matched scope '{scope}'"
        )
    return enabled


def dropout_probability_summary(model: nn.Module, scope: str) -> dict[str, int]:
    summary: dict[str, int] = {}
    for site in _dropout_sites(model):
        if _site_selected(site, scope):
            probability = f"{site.probability:.6g}"
            summary[probability] = summary.get(probability, 0) + 1
    return summary


def configure_pooling(model: SentenceTransformer, mode: str) -> str:
    valid_modes = {"model_default", "cls", "mean"}
    if mode not in valid_modes:
        raise ValueError(f"pooling must be one of {sorted(valid_modes)}")
    pooling_modules = [
        module for module in model.modules() if isinstance(module, Pooling)
    ]
    if len(pooling_modules) != 1:
        raise RuntimeError(
            f"Expected one SentenceTransformers Pooling module, found "
            f"{len(pooling_modules)}"
        )
    pooling = pooling_modules[0]
    if mode == "model_default":
        current_mode = getattr(pooling, "pooling_mode", None)
        if isinstance(current_mode, str):
            return current_mode
        if getattr(pooling, "pooling_mode_cls_token", False):
            return "cls"
        if getattr(pooling, "pooling_mode_mean_tokens", False):
            return "mean"
        return "other"

    if hasattr(pooling, "pooling_mode"):
        pooling.pooling_mode = mode
    else:  # Compatibility with SentenceTransformers before the unified mode field.
        pooling.pooling_mode_cls_token = mode == "cls"
        pooling.pooling_mode_mean_tokens = mode == "mean"
        pooling.pooling_mode_max_tokens = False
        pooling.pooling_mode_mean_sqrt_len_tokens = False
        pooling.pooling_mode_weightedmean_tokens = False
        pooling.pooling_mode_lasttoken = False
    return mode


class SentenceEmbeddingEncoder:
    def __init__(self, config: ModelConfig, device: str) -> None:
        self.config = config
        self.device = resolve_device(device)
        # Eager attention is required for stochastic encoding: SDPA applies
        # attention-probability dropout inside fused kernels, and some backends
        # (notably MPS) silently ignore dropout_p there. The eager path uses a
        # plain elementwise dropout that works on every backend.
        self.model = SentenceTransformer(
            config.name,
            revision=config.revision,
            device=self.device,
            model_kwargs={"attn_implementation": "eager"},
        )
        self.model.max_seq_length = config.max_length
        self.pooling = configure_pooling(self.model, config.pooling)
        dimension = self.model.get_sentence_embedding_dimension()
        if dimension is None:
            raise RuntimeError("Could not determine sentence embedding dimension")
        self.dimension = int(dimension)
        if (
            config.expected_dimension is not None
            and self.dimension != config.expected_dimension
        ):
            raise RuntimeError(
                f"{config.name} produced dimension {self.dimension}; "
                f"expected {config.expected_dimension}"
            )

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
    ) -> dict[str, Any]:
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
            "dropout_probabilities": dropout_probability_summary(
                self.model, self.config.dropout_scope
            ),
            "pooling": self.pooling,
        }

    @property
    def attention_implementation(self) -> str | None:
        try:
            return self.model[0].auto_model.config._attn_implementation
        except (AttributeError, IndexError, KeyError, TypeError):
            return None

    @property
    def resolved_revision(self) -> str | None:
        """The Hugging Face commit hash actually loaded, for provenance."""
        try:
            return self.model[0].auto_model.config._commit_hash
        except (AttributeError, IndexError, KeyError, TypeError):
            return None

    def verify_dropout_effect(
        self,
        probe_text: str = "Dropout effect verification probe sentence.",
    ) -> dict[str, bool]:
        """Fail loudly if any configured dropout scope does not perturb outputs.

        Guards against silent no-ops such as fused attention kernels ignoring
        dropout on some backends, or dropout gates that were never flipped.
        The 'all' scope is probed one constituent scope at a time so a dead
        attention pathway cannot hide behind an active hidden pathway.
        """
        scopes = (
            ["attention", "hidden"]
            if self.config.dropout_scope == "all"
            else [self.config.dropout_scope]
        )
        report: dict[str, bool] = {}
        for scope in scopes:
            configure_dropout(
                self.model,
                stochastic=True,
                scope=scope,
                probability=self.config.dropout_probability,
            )
            torch.manual_seed(0)
            first = self._encode_texts([probe_text])
            torch.manual_seed(1)
            second = self._encode_texts([probe_text])
            if np.array_equal(first, second):
                raise RuntimeError(
                    f"Dropout scope '{scope}' does not perturb encoder outputs on "
                    f"device '{self.device}' (attention implementation "
                    f"{self.attention_implementation!r}); the stochastic condition "
                    "would silently degenerate"
                )
            report[scope] = True
        configure_dropout(self.model, stochastic=False)
        return report

    def encode_records(
        self,
        records: list[TextRecord],
        prefix: str,
        seed: int,
        stochastic: bool,
    ) -> np.ndarray:
        """Encode a small in-memory batch under the same regime as encode_to_file."""
        set_reproducible_seed(seed)
        configure_dropout(
            self.model,
            stochastic=stochastic,
            scope=self.config.dropout_scope,
            probability=self.config.dropout_probability,
        )
        chunks = [
            self._encode_texts(
                [
                    f"{prefix}{record.text}"
                    for record in records[
                        start : start + self.config.batch_size
                    ]
                ]
            )
            for start in range(0, len(records), self.config.batch_size)
        ]
        return np.concatenate(chunks, axis=0)

    def _encode_texts(self, texts: list[str]) -> np.ndarray:
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
            raise RuntimeError("Non-finite embedding detected")
        if torch.linalg.vector_norm(output, dim=1).min().item() <= 1e-8:
            raise RuntimeError("Near-zero embedding detected")
        return output.detach().cpu().float().numpy()

    def _write_batch(
        self,
        batch: list[TextRecord],
        prefix: str,
        destination: np.memmap,
        ids_file: object,
        start: int,
    ) -> int:
        array = self._encode_texts([f"{prefix}{record.text}" for record in batch])
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
