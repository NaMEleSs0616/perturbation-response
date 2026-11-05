# Perturbation response prediction

Predicts how gene expression changes after a genetic perturbation, quantifies how far to trust that prediction, and helps rank follow-up experiments, using public single-cell data. This is an MVP toward multiscale biological models. It **does not simulate a cell**.

## What the evidence shows

Benchmark on Norman et al. 2019 (K562 CRISPRa; 105 single genes, 127 pairs), all on held-out perturbations. Details are in [docs/benchmarking.md](docs/benchmarking.md).

| Task | Result |
|---|---|
| New gene, no data on it | **Fails.** No model beats the mean of other perturbations, so the system abstains by default |
| New pair, both singles measured | **Works, mostly through a simple rule:** the pair ≈ 0.8 × (A + B). The MLP is not distinguishable from that rule on MSE |
| Finding strong pairs | About 2.6× enrichment over random, achieved equally by ranking on measured singles |
| Choosing experiments by uncertainty | No demonstrated practical benefit |
| Prediction intervals | About 86% coverage overall against a 90% target, but 74% on the strongest responders for pairs |

## Setup

Requires Python 3.10+ and about 2 GB of free RAM. No GPU or paid service is needed.

```bash
python3.11 -m venv .venv
```

```bash
.venv/bin/pip install -e ".[app,dev]"
```

`requirements-lock.txt` lists the versions used for the reported results. Commands below assume the virtual environment is active.

## Quick start

Laptop smoke test on a small, explicitly synthetic fixture, with no downloads. It tests the software, not biology:

```bash
python -m pytest
```

Reproduce the benchmark (a 699 MB download, then about 7 minutes including the preprocessing ablations):

```bash
python -m biophysical.cli datasets download norman2019
```

```bash
python -m biophysical.cli datasets validate norman2019
```

```bash
python -m biophysical.cli data preprocess norman2019
```

```bash
python -m biophysical.cli benchmark run --config configs/benchmark_norman.yaml
```

Compare STRING and Gene2vec representations, after downloading `gene2vec_dim_200_iter_9_w2v.txt` from https://github.com/jingcheng-du/Gene2vec into `data/external/gene2vec/`:

```bash
python -m biophysical.cli benchmark representations --gene2vec data/external/gene2vec/gene2vec_dim_200_iter_9_w2v.txt --out results/norman2019/representations
```

Launch the interface:

```bash
streamlit run app/streamlit_app.py
```

## Commands

```bash
python -m biophysical.cli datasets list
```

```bash
python -m biophysical.cli datasets inspect norman2019
```

Use a file you already have, without changing source code:

```bash
python -m biophysical.cli datasets set-local norman2019 /path/to/file.h5ad
```

Predict, with `--out` to export all genes:

```bash
python -m biophysical.cli predict CEBPA_TP73 --top 20 --out prediction.csv
```

Rank unmeasured pairs:

```bash
python -m biophysical.cli rank --budget 10 --strategy magnitude
```

Optional adapters:

```bash
python -m biophysical.cli geo summary GSE133344
```

```bash
python -m biophysical.cli string fetch --genes KLF1,GATA1,SPI1
```

```bash
python -m biophysical.cli annotate targets KLF1
```

Live-service tests, skipped by default:

```bash
BIOPHYSICAL_NETWORK_TESTS=1 python -m pytest tests/test_live_services.py
```

Set `BIOPHYSICAL_HOME` to keep `data/` and `results/` on another disk.

## Layout

| Path | Contents |
|---|---|
| `biophysical/` | Package; see [docs/architecture.md](docs/architecture.md) |
| `app/streamlit_app.py` | Local interface |
| `configs/` | Dataset registry and benchmark config |
| `tests/` | Automated tests; synthetic fixtures only, apart from the optional live-service checks |
| `data/raw/` | Downloaded source files, never modified |
| `data/processed/` | Versioned targets and metadata |
| `results/<dataset>/<run>/` | Config, splits, metrics, logs, serving model |

## Documentation

- [Implementation plan and status](docs/implementation_plan.md)
- [Public data sources](docs/data_sources.md)
- [Data provenance](docs/data_provenance.md)
- [Scientific assumptions](docs/scientific_assumptions.md)
- [Benchmark results](docs/benchmarking.md)
- [Model card](docs/model_card.md)
- [Architecture](docs/architecture.md)
- [Roadmap](docs/roadmap.md): planned work that does not exist yet

## Data and citation

Cite Norman et al., Science 2019 and Peidli et al., Nat Methods 2024 (scPerturb). See [docs/data_sources.md](docs/data_sources.md). A code license has not been chosen.
