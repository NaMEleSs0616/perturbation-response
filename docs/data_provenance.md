# Data provenance

The record of what was actually used. Machine-readable versions live next to the data: `data/raw/<id>/{download,validation,inspection}.json`, `data/processed/<id>/<version>/meta.json`, and `results/<id>/<run>/run.json`.

## Raw data

| Field | Value |
|---|---|
| Dataset | Norman et al. 2019, Science 365:786-793, doi:10.1126/science.aax4438 |
| Accession | GEO GSE133344 (PMID 31395745) |
| Source | scPerturb, Zenodo record 13350497, version 1.4 (concept DOI 10.5281/zenodo.7041848), CC-BY-4.0 |
| File | `NormanWeissman2019_filtered.h5ad`, 698,680,199 bytes |
| MD5 | `c870e6967d91c017d9da827bab183cd6` (matches Zenodo; verified by `datasets validate`, status **valid**) |
| Retrieved | 2026-11-01T13:55:38-0500 |
| URL | https://zenodo.org/api/records/13350497/files/NormanWeissman2019_filtered.h5ad/content |

The download spanned several interrupted sessions and resumed from the partial file; the MD5 confirms the completed file. Raw data are read only. The harmonised file was not compared to the raw GEO matrix.

`adamson2016_pilot` (`AdamsonWeissman2016_GSM2406675_10X001.h5ad`, MD5 `232f7e3756d41602bbe434b50662a76f`) was downloaded and checked, then **excluded** because it failed validation (no `control` label).

## Processed data

Version `931b2443c752beef`, created 2026-11-01T14:00:35-0500, seed 0.

| Step | Result |
|---|---|
| Cells in file | 111,445 |
| Removed: `good_coverage` is false | 6,938 |
| Removed: `number_of_cells` ≠ 1 | 13,339 |
| Cells after filters | 91,168 |
| Control cells used | 9,702 (labelled `control`, NegCtrl guide pairs) |
| Conditions kept | 232 |
| Conditions excluded | 4: JUN_CEBPA (49 cells), CBL_UBASH3A (48 cells), JUN_CEBPB (46 cells), C3orf72_FOXL2 (46 cells). All have fewer than 50 cells after filtering |
| Cells in kept conditions | 81,277 |
| Normalisation | library-size scaling to 10000 per cell over all genes in the file, then log1p |
| Genes | 2077 used of 33,694 in file: 2000 HVGs selected from control cells, plus perturbed targets |
| Gene selection | Dispersion of log-normalised values z-scored in 20 mean-expression bins, among genes detected in ≥ 1% of control cells |
| Target | cell-weighted mean of per-batch pseudobulk differences in log1p-normalised expression; Y_fit and Y_eval use disjoint control cells |
| Batch column | `gemgroup` (8 levels) |
| Co-expression embedding | 32 dimensions, from 8,000 randomly sampled control cells |

Target labels resolved through HGNC to Ensembl IDs, then var rows: `C19orf26` → `CBARP` (CBARP, ENSG00000099625), `C3orf72` → `FOXL2NB` (FOXL2NB, ENSG00000206262), `KIAA1804` → `RP5-862P8.2` (MAP3K21, ENSG00000143674). None were unresolved. Var has no annotation release or genome build recorded, so the source annotation version is unknown.

Per-gene and per-condition tables: `genes.csv`, `conditions.csv`. Split assignments: `results/.../split_assignments.csv`.

## External resources

| Resource | Version | Retrieved | Identity | Use |
|---|---|---|---|---|
| HGNC complete set | Not versioned by file | 2026-11-01T13:58:54-0500 | MD5 `39d46ee3ced2310056ec206019b67425` | Target resolution |
| STRING | 12.5 (https://version-12-5.string-db.org) | Cached in `data/external/string/` | Query hash in file name | Representation comparison only |
| Gene2vec | Repository commit `6236e0b21fbc367bf8ff5695ed0cc1443861bd1e` | 2026-11-01T14:10:48-0500 | MD5 `021e1141618b6b06a13963810b5ccd71` | Representation comparison only |
| GEO | Live | 2026-11-01 | GSE133344 summary | Provenance check only |
| Open Targets | API 26.9.0, data 26.09 | 2026-11-01 | Live query | Annotation; not used in any model |

## Run

| Field | Value |
|---|---|
| Run | `bench-20261101-142417-9f7a7e` |
| Target digests | Y_fit `780213a51852adff`, Y_eval `874b29925197b691` |
| Python and packages | Python 3.11.15; numpy 2.4.6, pandas 2.3.3, scipy 1.17.1, scikit-learn 1.9.1, anndata 0.12.19, h5py 3.16.0, torch 2.14.1, scanpy 1.11.5, requests 2.34.2 |
| Platform | macOS-26.5.1-arm64-arm-64bit |
