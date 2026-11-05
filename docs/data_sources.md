# Public data sources

Status terms: **used** means the resource was accessed and its data entered a reported result. **Tested** means the adapter was run against the live service but the data did not enter the core benchmark. **Registered** means a catalog entry exists and nothing has been downloaded.

## Perturbation data: scPerturb

- Site: https://scperturb.org/ · Publication: Peidli et al., Nat Methods 2024, https://www.nature.com/articles/s41592-023-02144-y · Code: https://github.com/sanderlab/scPerturb
- RNA archive: concept DOI 10.5281/zenodo.7041848. The concept DOI resolves to record **13350497, version 1.4**, which this project uses. Zenodo lists the record as CC-BY-4.0.
- ATAC archive (10.5281/zenodo.7058381) is not used: the first target is RNA.

| Registry id | File | MB | Status | Accession | Reason for its role |
|---|---|---|---|---|---|
| `norman2019` | NormanWeissman2019_filtered.h5ad | 698.7 | **Used**: primary benchmark | GEO GSE133344 | One cell line and modality, with singles and pairs, and a small enough file for a laptop |
| `adamson2016_pilot` | AdamsonWeissman2016_GSM2406675_10X001.h5ad | 34.6 | Downloaded, **fails validation** | GEO GSE90546 | See below |
| `adamson2016_upr` | AdamsonWeissman2016_GSM2406681_10X010.h5ad | 471.3 | Registered | GEO GSE90546 | CRISPRi in K562; candidate independent evaluation, not schema-checked |
| `replogle2022_rpe1` | ReplogleWeissman2022_rpe1.h5ad | 1236.9 | Registered | Figshare+ 10.25452/figshare.plus.20029387 | Cross-context candidate; requires `--allow-large` |
| `replogle2022_k562_essential` | ReplogleWeissman2022_K562_essential.h5ad | 1546.7 | Registered | same | Requires `--allow-large` |
| `dixit2016_k562_7d` | DixitRegev2016_K562_TFs_7_days.h5ad | 256.7 | Registered | GEO GSE90063 | CRISPR knockout, a different modality |
| `papalexi2021_eccite` | PapalexiSatija2021_eccite_RNA.h5ad | 147.2 | Registered | GEO GSE153056 | THP-1, a different context |

Fields in the "Registered" rows come from the Zenodo listing and the publications. Their obs schemas, control labels and replicate structure are unverified until `datasets inspect` is run on them. The datasets are not pooled: they differ in modality (CRISPRa, CRISPRi, knockout), cell type and depth.

### Norman 2019, as inspected

| Field | Observed in the file |
|---|---|
| System | K562; `cell_line` = K562, `celltype` = lymphoblasts, `disease` = chronic myelogenous leukemia |
| Organism, annotation | Human; var has gene names plus Ensembl IDs in a column spelled `ensemble_id`. The genome build is not stated in the file |
| Perturbations | CRISPRa from the publication. The file's `perturbation_type` just says `CRISPR`, which does not distinguish activation from knockout |
| Conditions | 105 single genes, 131 pairs, one `control` label |
| Controls | 11,855 cells (9,702 after QC), `nperts` = 0, from NegCtrl guide pairs |
| Cells | 111,445 in file; 91,168 after QC |
| Genes | 33,694; many have zero counts in this file |
| Matrix | Raw integer counts, CSC sparse, 361,582,621 non-zeros |
| Replicates | `gemgroup` with 8 levels (10x lanes). These are technical batches of one experiment, **not independent biological replicates** |
| Other covariates | No dose or time. `good_coverage` and `number_of_cells` describe guide-capture quality |
| Limitations | One cell line; no biological replicates; targets chosen by the study rather than at random; pairs are not a random sample of gene pairs |

### Adamson pilot: failed validation

`datasets validate adamson2016_pilot` reports `invalid` because no cell is labelled `control`. Its conditions are guide-level labels such as `62(mod)_pBA581`, and `nperts` takes values 0, 1 and 2. The control identity is not inferred; preprocessing raises an error unless `control_label` is set explicitly after checking the original study. It is excluded from all results.

## NCBI GEO

Adapter: `biophysical/data/geo.py` (E-utilities `esearch` and `esummary` on `db=gds`, plus the series supplementary directory listing). **Tested live** on GSE133344: the title matches the study, with PMID 31395745, 16 samples, and supplementary files including `GSE133344_filtered_matrix.mtx.gz` and `GSE133344_filtered_cell_identities.csv.gz`. Used for provenance only. Benchmark data come from the harmonised scPerturb file; the raw GEO matrix was not reprocessed or compared count by count.

## STRING

Adapter: `biophysical/networks/string_db.py`, API https://string-db.org/help/api/. **Tested live**, version 12.5 reported by the API. Per-channel scores are retained: `escore` (experimental), `dscore` (curated), `tscore` (text mining), and `nscore`, `fscore`, `pscore`, `ascore` (predicted associations). STRING edges are functional associations and are never treated as direct binding or causal regulation.

## Open Targets

Adapter: `biophysical/annotations/opentargets.py` (GraphQL). **Tested live**, API 26.9.0, data version 26.09. Annotation only: `python -m biophysical.cli annotate targets KLF1`. Association scores are never model labels, and they do not establish causality, efficacy or safety.

## HGNC

`biophysical/annotations/gene_ids.py` downloads `hgnc_complete_set.txt` from HGNC's public bucket. **Used** to resolve target labels; the file's retrieval date and MD5 are recorded in `data/external/hgnc/provenance.json` and each dataset's `meta.json`. Ensembl was not queried directly. Ensembl IDs come from HGNC and the H5AD var table, and they carry no release number, so the annotation release of the source matrix remains unknown.

## Gene2vec

Du et al., BMC Genomics 2019, doi:10.1186/s12864-018-5370-x. Repository https://github.com/jingcheng-du/Gene2vec, MIT license. 200-dimensional embeddings trained on co-expression in 984 GEO datasets, parsed as plain text. **Used** in the representation comparison only. Commit recorded in `data/external/gene2vec/provenance.json`.

## CELLxGENE

Not integrated. Nothing in this MVP needs an observational atlas, and such data contain no measured perturbation responses.

## Local files

```bash
python -m biophysical.cli datasets set-local norman2019 /path/to/file.h5ad
```

This records the path in `configs/local_paths.yaml`. The registry MD5 is not enforced for local files, and the file's MD5 is recorded instead.
