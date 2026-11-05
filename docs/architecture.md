# Architecture

```
configs/datasets.yaml  ─▶ data.registry ─▶ data.download (resume, MD5) ─▶ data/raw/<id>/   (never modified)
                                         └▶ data.inspect, data.validate
data/raw ─▶ preprocessing.pipeline
              pseudobulk: per-group statistics, chunked along the matrix's native axis
              targets:    Y_fit and Y_eval (disjoint controls), split-half, batch-disjoint, SE
              coexpression embedding (control cells only)
           ─▶ data/processed/<id>/<version>/  (response.npz, conditions.csv, genes.csv, meta.json)
data/processed ─▶ preprocessing.splits ─▶ evaluation.benchmark ─▶ results/<id>/<run>/
                  models.{baselines,mlp}   uncertainty.calibration
                  active_learning.simulate
results/<run>/serving_bundle.pkl ─▶ api.service ─▶ cli, app/streamlit_app.py
optional: networks.*, models.external ─▶ evaluation.representations
```

| Package | Responsibility |
|---|---|
| `data` | Registry, downloader, H5AD schema inspection, validation, GEO adapter, native-axis sparse reader |
| `annotations` | HGNC mapping with explicit ambiguity, Open Targets annotation |
| `networks` | STRING client and adjacency and graph features |
| `preprocessing` | Group statistics, targets, gene selection, splits and leakage checks |
| `models` | Features, baselines, MLP ensemble, external embeddings |
| `training` | Early stopping loop |
| `evaluation` | Metrics, benchmark, ablations, representation comparison |
| `uncertainty` | Interval calibration, disagreement, input support checks |
| `active_learning` | Candidate scoring and retrospective simulation |
| `api` | `PredictionService` used by the CLI and app |
| `visualization` | Altair charts |
| `future` | Interface definitions only; no implemented capability |

## Design decisions

- **Native-axis processing.** scPerturb stores X as CSC here. Row slicing a backed CSC matrix took over two minutes on a 35 MB file, so statistics are accumulated over gene chunks. Preprocessing the full Norman file takes about a minute with a 1.3 GB peak.
- **The perturbation is the training example.** Targets are pseudobulks, so a model sees a few hundred examples, not ~100k cells. This matches the claim being tested and rules out cell-level leakage.
- **One target definition, with disjoint controls.** See `scientific_assumptions.md`.
- **Serving bundles are pickles.** They are written and read locally by this package. Do not load one from an untrusted source.

## Reproducibility

Processed data are keyed by a hash of the config, raw MD5 and seed. Each run stores its config, split assignments, target digests, seed, package versions, per-perturbation metrics, logs and the serving bundle. Rerunning with the same seed on the synthetic fixture is tested to give identical metrics (`test_benchmark_is_reproducible`).
