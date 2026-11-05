# Scientific assumptions and limitations

## Target

For perturbation *p* and gene *g*, with batches *r* (10x lanes):

Δ(p, g) = Σᵣ wₚᵣ · ( mean log1p CP10k of g in p cells of r − mean log1p CP10k of g in control cells of r ), with wₚᵣ ∝ cells of p in r.

- **What it is:** a difference of pseudobulk means of log-normalised expression, in K562, under CRISPRa, at the study's single sampling time.
- **Causal reading:** this estimates an effect only if guide assignment within a batch is effectively random, and if negative-control guides are valid controls. Both are design assumptions and neither is tested here.
- **Normalisation:** library-size scaling is compositional, so a large change in a few abundant transcripts shifts all other genes slightly.
- **Log of means vs mean of logs:** these are different quantities. An ablation compares them; they are never mixed in one model or metric.

## Decisions made after inspecting the data

| Decision | Reason | Consequence |
|---|---|---|
| Keep `good_coverage` and `number_of_cells == 1` | Multiplet droplets mix expression across cells | Removes 20,277 of 111,445 cells. Four pairs fall below 50 cells and are excluded (recorded in `meta.json`). An ablation shows the effect of not filtering |
| Exclude the perturbed gene from metrics in its own condition | CRISPRa raises its own target, so including it credits a model for restating the input | Self-entries are masked in training and evaluation |
| Add target genes to the 2,000 HVGs | Gives every target a row in response-derived features, and makes on-target checks possible | 2,077 genes |
| Select genes on control cells only | Prevents selection on perturbation effects, including test perturbations | Tested: scrambling perturbed cells does not change the genes or embedding |
| Resolve target labels through HGNC and Ensembl | `KIAA1804`, `C19orf26`, `C3orf72` are not var names | They map to `RP5-862P8.2`, `CBARP`, `FOXL2NB`. Mapping is recorded, and an ambiguous mapping excludes the condition |
| Disjoint controls for training and evaluation targets | A shared control pool would give all responses a common noise term | See the ablation; the measured effect in this dataset is small |
| Report perturbation-specific metrics | Most perturbations share a large common shift from controls, so raw correlation rewards predicting only that | Primary metric is `pearson_specific`. The mean baseline has no score on it, and a skill of exactly zero |

## Features and information available to each model

Co-expression embeddings use control cells only. Target-gene baseline expression is the only varying "baseline" input: all examples share one context, so a full baseline expression vector would be a constant. No dose or time data exist in this dataset, so these covariates are unused (the model code accepts covariates).

`gene_loading_ridge` represents a target by how that gene responds across training perturbations. This is legitimate for unseen-gene prediction, since those measurements exist in the training conditions. It also means the method cannot be applied to a gene absent from the measured gene set.

## What the splits do and do not show

- `unseen_gene` holds out all data on a target. Other genes in the same family or pathway may remain in training, as they would in practice, so results do not measure extrapolation to unrelated biology. The targets were also chosen by the study, around genes with expected phenotypes, so they are not a random sample of the genome.
- `unseen_combo` holds out pairs with both singles in training. Pairs were selected by the original study, partly for expected interactions, so performance may not transfer to random pairs.
- Generalisation across **biological replicates** cannot be tested: this file has none. Batch-disjoint reproducibility is reported as a technical check.
- **New cell contexts are not tested.** The model abstains on other contexts.

## Uncertainty

Three separate quantities are reported, none of which captures everything:

- **Model disagreement:** spread across ensemble members. It misses bias shared by all members.
- **Measurement variability:** cell-level standard error of the response, available in `se`.
- **Calibrated intervals:** per-gene widths and a quantile from held-out residuals. Coverage on test perturbations is measured and reported. It is not guaranteed, because genes within a perturbation are dependent and calibration perturbations were also used for model selection.

## Not claimed

No claim is made about protein abundance, phenotype, clinical efficacy, toxicity, causal mechanism for a specific gene pair, other cell types, or other perturbation modalities. This is not a cell simulator.
