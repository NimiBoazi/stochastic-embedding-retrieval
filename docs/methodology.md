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

Pre-registered base-scale cross-family models:

- BAAI/bge-base-en-v1.5
- intfloat/e5-base-v2
- facebook/contriever
- sentence-transformers/gtr-t5-base

These checkpoints are approximately base-sized, which limits model capacity as a
confound while varying model family, training data, supervision, and pooling.
Differences between them are interpreted as model-family effects, not as causal
architecture effects, because their training data and objectives also differ.

Secondary scaling comparison:

- BAAI/bge-base-en-v1.5
- BAAI/bge-large-en-v1.5

This is a pre-specified two-size comparison within one family. It is not sufficient
to claim a general scaling law. BGE-large is not part of the four-model
cross-family confirmatory comparison.

Pre-registered datasets and roles:

- SciFact: development and pipeline validation. Methodology may change after
  observing this dataset, so it is not used as the sole confirmatory evidence.
- FiQA: confirmatory cross-domain evaluation on financial question answering.
- NFCorpus: confirmatory evaluation with biomedical queries and richer
  multi-relevance judgments.
- BEIR HotpotQA: large-scale, multi-hop external validation after methods and
  hyperparameters are frozen.

The four base-scale models run on all four datasets. BGE-large is a secondary
scaling condition and is required on SciFact plus at least one confirmatory
dataset; it does not need to be run on HotpotQA unless compute permits.

For every dataset, report the number of relevant documents per query and stratify
single- versus multi-relevance results. `qrels_overlap@k` records how many
retrieved documents occur in the supplied qrels; it must not be interpreted as
complete judgment coverage because BEIR qrels are often sparse or positive-only.

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

The scope/strength ablation sweep runs all four base models on SciFact with 16
query samples per condition:

1. all dropout at native probabilities
2. attention-only at native probabilities
3. hidden-only at native probabilities
4. all dropout with probabilities 0.05, 0.10, and 0.20

This is a development analysis, not six additional confirmatory hypotheses.
Any condition selected for later use must be frozen and evaluated unchanged on
at least one confirmatory dataset. The deterministic baseline is included in
every condition, but `p=0` is not treated as stochastic dropout.

Aggregation:

- normalized centroid
- medoid sample
- reciprocal-rank fusion
- top-k voting
- maximum score, treated cautiously because its null distribution changes with N
- agreement-weighted scoring
- oracle best-of-N

