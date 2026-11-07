# Response to review

All new analyses are labelled **post hoc** in the manuscript, because they were added after the confirmatory results were seen. Numbers in the manuscript are generated from results files (`paper/numbers.tex`, `paper/numbers_revision.tex`).

One correction to the review: the interval quoted for Tian CRISPRa (−0.036 to +0.023) is the 95% interval of the mean ΔMSE×10³, not of skill. The substantive point stands and is addressed below.

## Major concerns

**1. C1: Wilcoxon versus bootstrap discrepancy.** Addressed.
- *Results 2.3* now states explicitly that the Tian CRISPRa MLP passed the Holm-corrected signed-rank test (p = 0.0098) but failed the mean-difference criterion.
- The MLP is better for 71% of perturbations, with a median ΔMSE of −0.028×10⁻³ (95% CI −0.041 to −0.019). A few large right-tail losses bring the mean to −0.008. The paper reports that the MLP was "typically, but not on average" better.
- New *Supplementary Fig. S1* shows every per-perturbation difference; new *Table S5* adds medians with intervals and the fraction better, for all datasets and models.
- *Methods 4.7* explains the composite rule: the signed-rank test concerns the location of the differences, the bootstrap concerns the average effect, and requiring both guards against declaring improvement when rare large failures raise expected error.

**2. C2: multiple testing.** Addressed.
- The plan specified no correction for C2, and the manuscript now says so.
- A post-hoc Holm correction over its eight Wilcoxon tests leaves all four Replogle results significant (largest adjusted p = 4×10⁻⁸) and none of the neuron results (adjusted p ≥ 0.06).
- The text now distinguishes "co-expression carries information" (supported) from "the information is practically useful" (gains are a few hundredths of skill).

**3. Reproducibility limits are technical.** Addressed.
- They are now called "technical reproducibility limits" throughout the text and figures.
- *Methods 4.6* lists the three assumptions behind the Spearman–Brown estimate.
- *Results 2.2* and the *Limitations* paragraph state that the limits omit biological replicate variation and describe predictability within an experiment, and that the attainable correlation against an independent repeat would be lower.

**4. Scope of models.** Addressed.
- The *Abstract* names the model classes evaluated and states that graph neural networks and foundation models were not tested.
- A new *Discussion* paragraph, "Scope of the conclusions", says the results concern the information these representations carry with these model classes. They do not show that Perturb-seq data are insufficient for stronger models.

**5. What co-expression carries.** Addressed with a new analysis (*Results 2.5, Fig. 3, Table S6*).
- Ridge and the MLP were refit with co-expression only, baseline mean only, detection rate only, both baseline features, and the full set.
- In the large screens, co-expression alone matches the full set.
- Baseline features alone give the MLP about two-thirds of the gain and ridge almost none.
- Target expression correlates only weakly with response magnitude (ρ = 0.16 in K562, 0.06 in RPE1), so we describe the MLP's use of these features as nonlinear and unresolved, rather than claiming a mechanism.
- The *Discussion* adds a biological interpretation (generic gene properties and broad co-regulated programmes, not mechanism-specific information), and states that these are associations, not causal relationships.

## Technical issues

| Issue | Change |
|---|---|
| Embedding construction | *Methods 4.5*: centring, unit-variance scaling, zero-variance handling, randomised truncated SVD, scaling by singular values / √n, standardisation on training conditions |
| Gene selection | *Methods 4.3*: dispersion = variance / mean of log-normalised expression in control cells, z-scored within 20 quantile bins of mean expression, top 2,000 |
| Hyperparameters | New *Supplementary Table S2*: all grids, architectures, seeds and early-stopping settings |
| Control splitting | *Methods 4.5* states which control cells each feature uses. New check (*Results 2.5, Table S6*): all control-derived features recomputed from training-half controls only; performance unchanged (K562 MLP skill 0.052 vs 0.051; RPE1 0.048 vs 0.052) |
| Prediction intervals | Promoted to its own Results section (2.10) and to the *Discussion*. Tested magnitude-stratified calibration (*Table S7*), which barely helps (K562 top-20 coverage 0.35 → 0.37), because the models do not anticipate which genes respond strongly |
| Pair analysis | Retitled and reworded as sub-additivity "among selected, interaction-enriched pairs"; *Abstract* updated |
| Biological replicates | Stated in *Results 2.2*, *Methods 4.6* and *Limitations* |

## Reproducibility and submission readiness

| Item | Status |
|---|---|
| Public repository | **Author decision required.** The repository is private; the Availability statement is marked AUTHOR-TO-CONFIRM |
| Analysis plan and addendum | In the repository; full SHA-256 hashes in new *Table S1* |
| Environment | `requirements-lock.txt`, hash in *Table S1*; versions in *Methods 4.10* |
| Data download and preprocessing | `python -m biophysical.cli datasets download/validate`, `data preprocess`; the study runner preprocesses automatically |
| Configurations and seeds | `configs/study.yaml`, `configs/benchmark_norman.yaml`; hashes in *Table S1*; seeds in *Table S2* |
| Figure and table scripts | `paper/scripts/` |
| Release tag and commit hash | To add when the repository is made public |
| Full supplementary results | Per-perturbation results for every dataset, model and seed are written by the study runner (`results/study/primary/`). These are not in the repository (`results/` is git-ignored) and should be deposited, e.g. on Zenodo, alongside the release |

## Writing

- **Abstract:** rewritten with fewer numbers and more emphasis on the protocol. It also distinguishes technical reproducibility and states the model scope.
- **Introduction:** a new paragraph explains how the protocol relates to Systema, Csendes et al. and Miller et al., and what it adds.
- **Discussion:** reorganised into scope, relation to the debate, what the models learn, implications for experiment prioritisation, methodological lessons and limitations.
- **References:** all checked against publisher or PubMed records; issue numbers added for Lotfollahi et al. and Roohani et al.

## Still for the author

1. Author name and affiliation.
2. The AI-assistance statement (marked AUTHOR-TO-CONFIRM).
3. Making the repository public, with a release tag.
4. Depositing the per-perturbation results archive.
