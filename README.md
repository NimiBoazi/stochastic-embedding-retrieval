# Self-Consistency for Stochastic Embedding Retrieval

This project studies whether Monte Carlo dropout samples from a single embedding
model can improve dense retrieval through self-consistency-inspired aggregation.
It preserves embeddings, sampled rankings, per-query metrics, and provenance so
that aggregation and analysis can be repeated without another model inference run.

## Research questions

1. How much embedding and ranking diversity does Monte Carlo dropout produce?
2. Can label-free aggregation outperform deterministic retrieval?
3. What is the best-of-N oracle upper bound, and for which queries is it large?
4. How do effects vary with sample count, dropout, model, dataset, and compute?

The oracle method deliberately uses qrels. It is an academic diagnostic and is
always labeled separately from deployable methods.

## Pre-registered model design

The base-scale cross-family comparison uses:

- `BAAI/bge-base-en-v1.5`
- `intfloat/e5-base-v2`
- `facebook/contriever`
- `sentence-transformers/gtr-t5-base`

`BAAI/bge-large-en-v1.5` is a separate, secondary comparison with BGE-base. It
tests two sizes within one model family and is not treated as evidence of a
general scaling law.

Model files explicitly declare query/document prefixes, pooling, expected output
dimension, and native dropout behavior. Experiment YAML can reference reusable
model and dataset files through `model_config` and `dataset_config`.

The pre-registered retrieval datasets are SciFact (development), FiQA and
NFCorpus (confirmatory), and BEIR HotpotQA (large-scale multi-hop validation).

## Implemented pilot

- Streaming BEIR dataset access through `ir_datasets`.
- Deterministic document and query embeddings.
- Reproducible stochastic query embeddings with selected dropout modules active.
- Exact, blockwise inner-product retrieval.
- Mean embedding, mean score, medoid, reciprocal-rank fusion, majority vote, and
  oracle best-of-N aggregation.
- nDCG, recall, MAP, and MRR at configurable cutoffs.
- Paired bootstrap confidence intervals against the deterministic baseline.
- Embedding diversity diagnostics.
- Immutable run directories containing NumPy, NPZ, Parquet, JSON, and a manifest.

Stochastic document encoding is intentionally gated until the query-side pilot is
validated; full HotpotQA document sample banks can require hundreds of gigabytes.

## Setup

Python 3.10–3.12 is recommended.

```bash
cd final_project
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,cpu-index]"
```

The current exact-search implementation does not require FAISS. The optional
FAISS dependency is reserved for the large-corpus backend.

Validate the configuration without downloading data or a model:

```bash
stochastic-retrieval validate configs/experiments/scifact_bge_pilot.yaml
```

Run the SciFact pilot:

```bash
stochastic-retrieval run configs/experiments/scifact_bge_pilot.yaml
```

The first invocation downloads the BEIR collection and Hugging Face model.

The remaining SciFact screening configurations are in `configs/experiments/`.
Run each independently so that failures and artifacts remain isolated by model.

List the complete 4-model × 4-dataset matrix without starting it:

```bash
stochastic-retrieval sweep configs/sweeps/core_retrieval.yaml
```

Pass `--execute` only after reviewing the 16 conditions. Each condition remains
an independent, resumable artifact run, and one failure does not discard the
other completed conditions.

The secondary `configs/sweeps/bge_scaling.yaml` adds BGE-large on SciFact and
FiQA. Its BGE-base comparison values come from the core sweep, avoiding duplicate
inference.

Real-checkpoint contract tests are opt-in because they download all five models:

```bash
RUN_MODEL_CONTRACT_TESTS=1 pytest tests/test_model_contracts.py
```

## Outputs

Each configuration receives a deterministic fingerprint:

```text
artifacts/runs/<experiment-name>-<fingerprint>/
├── embeddings/
│   ├── documents/sample_000.npy
│   ├── queries_deterministic/sample_000.npy
│   └── queries_stochastic/sample_*.npy
├── rankings/
│   ├── samples/sample_*.npz
│   └── aggregated_rankings.parquet
├── metrics/
│   ├── per_query.parquet
│   ├── summary.parquet
│   ├── summary_by_relevance_group.parquet
│   └── paired_bootstrap.parquet
├── analyses/
│   ├── embedding_diversity.parquet
│   ├── dataset_query_diagnostics.parquet
│   └── oracle_selections.parquet
├── events.jsonl
├── qrels.json
└── manifest.json
```

Artifacts are excluded from Git. Copy completed runs to versioned GCS or Azure
Blob storage for long-term retention.

## Runtime reporting and safeguards

The runner prints and immediately persists lightweight stage events rather than
using a training-oriented dashboard. It reports stage duration, item counts,
device, embedding dimension, enabled dropout-module count, cached artifacts, and
the native dropout probabilities, configured pooling, and final aggregate metrics.

Every embedding artifact is checked for its expected shape, non-finite values,
and near-zero norms. Stochastic query banks must differ from both deterministic
embeddings and one another. `manifest.json` records `running`, `completed`, or
`failed`; failures include their exception type and message. This makes interrupted
or invalid runs distinguishable from successful experiments.

## Methodological safeguards

- The model is globally placed in evaluation mode; only selected `Dropout`
  modules return to training mode.
- Every stochastic sample has a recorded seed.
- The native trained dropout probability is the primary condition. Overridden
  dropout rates are distribution-shift ablations.
- Query/document prefixes are explicit and model-specific.
- Deterministic and stochastic representations use identical preprocessing.
- All methods are evaluated per query, enabling paired statistical tests.
- Per-query outputs record relevant-document counts and qrels overlap, and
  summaries separate single-relevance from multi-relevance queries.
- `mean_score` is retained as a named experimental condition, but for dot-product
  retrieval it is algebraically equivalent to scoring with an unnormalized mean
  query. Query-vector L2 normalization only rescales all scores for that query,
  so it also does not change ranking.

See [docs/methodology.md](docs/methodology.md) for the staged experiment design.

For large corpora, set `retrieval_backend: faiss-gpu` after installing the GPU
extra. The FAISS corpus index is built once and reused across deterministic,
sampled, and aggregated queries. Ranking rows are written to Parquet in bounded
query chunks rather than accumulated into one in-memory table.
