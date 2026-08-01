# Methodology and experiment protocol

## Claims and endpoints

The confirmatory claim is that a pre-specified, label-free stochastic aggregation
method improves retrieval over the deterministic encoder. The primary endpoint is
nDCG@10. Recall@100/1000, MAP@100, MRR@10, latency, storage, and GPU-hours are
secondary endpoints.

Oracle best-of-N is an explanatory upper bound. It must not be included when
claiming deployable performance.

## Stage 1 — pipeline validation

Use BGE-base-en-v1.5 on SciFact:

- Verify a credible deterministic baseline.
- Verify deterministic reruns are identical.
- Confirm stochastic samples differ when dropout is active.
- Confirm query ordering and qrels alignment.
- Compare mean-score and mean-embedding rankings; explain expected equivalence.
- Inspect at least 20 oracle wins and 20 oracle failures manually.

No broad conclusions should be drawn from this stage.

## Stage 2 — stochastic-query study

Models:

- BAAI/bge-base-en-v1.5
- intfloat/e5-base-v2
- Alibaba-NLP/gte-base-en-v1.5

Datasets:

- SciFact
- NFCorpus or FiQA
- BEIR HotpotQA
- One additional large BEIR collection if compute permits

Sample-count prefixes: `N = 1, 2, 4, 8, 16, 32, 64`. Generate the maximum bank
once. Estimate sample-count uncertainty with multiple fixed subsamples rather than
rerunning model inference.

Primary dropout condition uses each checkpoint's trained probability. Ablations:

- attention dropout only
- hidden dropout only
- all dropout
- overridden probabilities 0.05, 0.10, and 0.20

Overridden probabilities are stress tests because changing dropout after training
creates distribution shift.

Aggregation:

- normalized centroid
- medoid sample
- reciprocal-rank fusion
- top-k voting
- maximum score, treated cautiously because its null distribution changes with N
- agreement-weighted scoring
- oracle best-of-N

## Stage 3 — diversity and uncertainty

Embedding-space diagnostics:

- pairwise cosine dispersion
- covariance trace and effective rank
- deterministic-to-centroid displacement
- per-dimension variance

Ranking-space diagnostics:

- top-k Jaccard overlap
- rank-biased overlap
- unique-document count
- vote entropy

Fit pre-specified regressions relating diversity to deterministic query difficulty,
oracle headroom, and aggregation gain. Use a larger sample bank on a stratified
query subset for covariance, entropy, and clustering analyses.

## Stage 4 — stochastic documents

Begin on SciFact, then scale only selected conditions. Compare:

- deterministic documents
- normalized mean stochastic document
- document medoid
- variance-aware scoring
- rank fusion over stochastic document indexes

For normalized stochastic vectors and dot products:

`mean_ij(q_i · d_j) = mean(q_i) · mean(d_j)`.

Therefore, all-pairs average scoring is not an independent nonlinear aggregation
method. Renormalizing each document centroid can change rankings because the
normalization factor varies by document.

For large corpora, store float16 sample shards in object storage and maintain
float32 streaming means/second moments. Record quantization as an experimental
factor and audit its retrieval effect.

## Statistical protocol

- Freeze development queries before tuning.
- Never select methods or hyperparameters on test qrels.
- Report paired bootstrap 95% confidence intervals over queries.
- Use paired randomization tests for final confirmatory comparisons.
- Correct confirmatory p-values for the number of pre-registered comparisons.
- Report mean effect, confidence interval, win/tie/loss counts, and per-query data.
- Include all failed and null experimental conditions in the final results ledger.

For oracle best-of-N, report both the oracle metric and selection frequency. Avoid
evaluating a learned selector on the same qrels used to train it.

## Reproducibility checklist

- Pin model repository revisions.
- Fingerprint dataset, preprocessing, and experiment configuration.
- Record code revision, Python/package versions, hardware, dtype, and seeds.
- Record batch size because random-number consumption may be batch dependent.
- Persist sample embeddings and sample top-k rankings.
- Separate raw artifacts from derived analyses.
- Generate paper tables and plots from immutable Parquet outputs.
- Reproduce a subset on a second machine or cloud provider.
