# Implementation plan and status

Phases follow the project brief. Status reflects what was run, not what was intended.

| Phase | Deliverable | Acceptance check | Status |
|---|---|---|---|
| A | Inspect repo, hardware | Written record | Done: empty folder; Apple M3, 8 GB RAM, no GPU; Python 3.11 venv |
| B | Package, config, logging, tests | Test suite runs offline | Done |
| C | scPerturb investigation, registry, downloader, validation | Real file validated by MD5 | Done: Norman 2019 from Zenodo record 13350497 |
| D | Preprocessing, targets, splits | Target matches an independent calculation; leakage checks raise | Done |
| E | Baselines and trainable model | Same folds and targets for all models | Done |
| F | Benchmark, failure analysis, uncertainty | Per-perturbation results, ceilings, measured coverage | Done; see `benchmarking.md` |
| G | Candidate ranking | Retrospective comparison with random | Done |
| H | Interface, docs, reproducible command | App runs against the real pipeline | Done |
| I | STRING, Open Targets, pretrained | Ablation against co-expression | STRING and Gene2vec compared; Open Targets annotation only |

## MVP acceptance criteria

1. Documented public dataset accessed: **met** (Norman 2019).
2. Validated and preprocessed: **met**.
3. Baseline and trainable models trained: **met**.
4. Reproducible metrics: **met**: two complete real-data runs gave identical per-perturbation metrics on this machine, and a test covers this on the synthetic fixture.
5. Splits match the claims: **met** (`unseen_gene`, `unseen_combo`, with leakage checks in every fold).
6. Predictions and uncertainty inspectable and exportable: **met** (CLI, app CSV and JSON export).
7. Candidates ranked by a documented method: **met**.
8. Tests pass: **met**; see the README for the command.
9. Local interface runs on the actual pipeline: **met**; see the README.
10. Documentation separates demonstrated results from plans: **met** (`benchmarking.md`, `model_card.md` vs `roadmap.md`).

## Done since the MVP

- Multi-dataset study under a fixed analysis plan (Replogle K562 and RPE1, Tian neuron CRISPRa and CRISPRi); see `paper/main.pdf`.

## Not done

- Evaluation across biological replicates (not present in this file).
- Fine-tuned pretrained embeddings and large foundation-model embeddings. In the unseen-gene task, fine-tuning cannot update a held-out gene's vector because no training example contains it, so frozen vectors were compared first.
- A graph neural network. Graph-derived features were tested instead; they are the cheaper test of whether the network carries usable signal.
