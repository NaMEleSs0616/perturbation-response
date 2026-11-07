# Response to the second review

All new analyses are **post hoc** and labelled as such. They use a further preprocessing pass that reproduces every primary target exactly (maximum difference 0). Scripts: `paper/scripts/revision2_extended.py`, `revision2_evaluate.py`, `revision2_outputs.py`. Every quoted number is generated (`paper/numbers_revision2.tex`).

## 1. Critical methodological points

**1.1 Calibration confused with reliability.** We agree, and have reframed the analysis.
- The analysis is renamed "metric discrimination" (Methods 4.6, Results 2.2, Fig. 1). The text states that MSE is a proper scoring rule and that a low win rate reflects responses small relative to noise.
- New Table S10: the MSE win rate tracks reproducibility almost perfectly. In K562 it is 0.00 in the least reproducible third of perturbations and 0.96 in the most reproducible third; in RPE1, 0.24 and 1.00.
- The 80% threshold is reported as pre-specified, but the win rate is treated as continuous, with Wilson intervals (Table S9).
- **Oracle comparator: the reviewer was right, and we have removed it.** The earlier post-hoc "best of mean and zero" took the maximum per perturbation, choosing the comparator using the outcome. Duplicate-vs-mean and duplicate-vs-zero are now reported separately (Fig. 1b, Table S9).
- The handling of undefined correlations is stated in Methods and captions.
- The claim that "MSE is poorly calibrated in four of five screens" has been removed from the Abstract and Discussion.

**1.2 Reproducibility limits.**
- (A) Tian CRISPRa has two batches, so every partition is the same contrast; this is now stated in Results 2.2 and Table S8. The cell-split limit there is higher (0.55 vs 0.46).
- (B) 95% bootstrap intervals and the range over five batch partitions are now reported (Table S8). For example, K562 0.72 (0.71–0.73), range 0.72–0.72; Tian CRISPRi 0.19 (0.16–0.21), range 0.17–0.19.
- (C) The limits are called technical throughout. Results, Discussion and Limitations restrict their interpretation to predictability within an experiment.

**1.3 Dependence and other sources of uncertainty.**
- A new block bootstrap resamples clusters of co-expressed targets (Table S13). Intervals for the primary C1 estimand (mean ΔMSE) still exclude zero for all six Replogle tests. On mean skill, the MLP stays above zero, but the linear models' intervals now include zero in RPE1. This is reported in Results.
- Five independent control-cell splits (MLP skill 0.047–0.052 in K562, 0.052–0.054 in RPE1) and five MLP seeds (0.047–0.055) are reported in Table S12.
- Fold partitions were already in Table S4. Batch partitions are in Table S8.
- Dependence along external pathway annotations is not addressed; this is stated in Limitations.

**1.4 Output-gene selection.** A new extended output space covers every gene detected in ≥1% of control cells or of any condition (Table S11; figure `paper/figures/fig8_gene_sets.pdf`).
- The Replogle conclusions hold on all genes (MLP skill 0.041 and 0.047) and on genes outside the selected set (0.034 and 0.036).
- Genes nearly silent in controls exist only in Norman and the neuron screens: the harmonised Replogle files contain only expressed genes, which we now state. No model beat the mean on those genes.

**1.5 Measurement quality.** New Table S14 reports skill by thirds of cell count, a cell-weighted mean, and the mean over conditions with ≥100 cells. Skill rises with cell count; excluding conditions with fewer than 100 cells raises MLP skill to 0.060 (K562) and 0.086 (RPE1). Uncertainty-weighted *fitting* was not attempted.

## 2. Breadth of conclusions

- **2.1** Now stated in Results 2.1, the Abstract and Limitations: the positive results come from two cell lines in one study and are not independent replications.
- **2.2** Training-set size is now described as a within-study association. The relevant section is retitled ("Within a screen, gains grow…"), and the text names the confounders across screens and the three draws per size.
- **2.3** The title now reads "How far can simple models…", and the Abstract lists the model classes tested and those not tested. kNN now appears in Fig. 2 and in Tables S11–S14. On Gene2vec, the text says only that this embedding did not help.

## 3. Individual analyses

- **3.1** Replaced with a selection-independent version (Table S16). Perturbations are selected on two cell halves within one batch set and scored against the other set, which shares no perturbed cells or batch-matched controls with the selection. The result replicates: K562 ridge skill 0.090 (0.050–0.124) and MLP 0.094 among reproducible perturbations. The caption notes the remaining shared training controls, which affect the models and the mean equally but not the zero prediction.
- **3.2** Retitled "attenuated additive projection". The text lists the alternative explanations (guide efficacy, orthogonal components, cell state and survival) and the IV assumptions, and restricts the finding to selected pairs.
- **3.3 The reviewer was right, and we have corrected the claim.** On genes selected by *predicted* deviation, coverage is close to nominal (0.85–0.90; Table S17). The earlier 35–55% came from selecting on the observed outcome. "Intervals fail where it matters" is withdrawn from the Results, Discussion and Abstract.
- **3.4** New Table S18 gives precision at 10%, 20% and 30% with bootstrap intervals, permutation p-values against random rankings, and the range over five MLP seeds. The text separates statistical enrichment (1.5- to 3-fold) from experimental utility, which remains untested.
- **3.5** Softened: the comparison is described as one pair of cell lines, two different information settings and a simple transfer model, and labelled exploratory.

## 4. C3 inconsistency
Results 2.4 now explains the K562 MLP case. The median difference is −0.009 (95% CI −0.012 to −0.004), and Gene2vec is better for only 45% of perturbations, while the mean's interval includes zero. That is the same mechanism as in C1. Table S15 reports mean and median estimates for every confirmatory test.

## 5. Reproducibility and transparency
- **5.1 Private repository:** the repository is now public, and this version is tagged `v1.0-paper`. The per-perturbation results, predictions and summary tables are attached to that release with SHA-256 checksums (Data and code availability).
- **5.2** The text now states that the plan files were first committed after the study ran, so pre-specification cannot be independently verified (Methods 4.1, Limitations, S10).
- **5.3** The text now says that identical runs show determinism, not robustness, and points to the sensitivity analyses.
- **5.4** The author, affiliation and AI-use statement are now complete.

## 6. Writing and framing
- The title distinguishes discovery from confirmation and states the model scope.
- The Abstract was rewritten to remove the over-general conclusion. It now ends with: prediction is "weak, concentrated in reproducible perturbations, and easy to mask or overstate through the choice of metric and baseline".
- The text now separates statistically detectable improvements from practical utility in Results 2.8 and the Discussion.
- References 10 and 11 are completed with identifiers and versions.

## Not done
- Benchmarking GEARS or foundation models. The claims are narrowed instead.
- Uncertainty-weighted model fitting.
- Prospective experimental validation.
- External registration of the analysis plan.
