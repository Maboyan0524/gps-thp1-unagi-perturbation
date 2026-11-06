# GPS fine-tuning to THP-1 and UNAGI trajectory perturbation

Code accompanying the cell-state reversal component of the manuscript. Two
stages: a GPS drug-response model is retrained on LINCS THP-1 data to obtain
macrophage-specific transcriptional profiles for candidate compounds, and those
profiles are then applied as in silico perturbations on a UNAGI model of the
pro-fibrotic macrophage polarisation trajectory in silicosis.

This repository contains the pipeline code only. It does not include LINCS or
single-cell data, trained weights, predicted profiles, or result tables.

Code comments and docstrings are in Chinese.

## Stage 1 — GPS retrained on THP-1 (`gps_thp1/`)

A Python port of the GPS RCL pipeline, with the algorithm and hyperparameters
unchanged from the reference implementation. Scripts run in order.

| Script | Corresponds to | Purpose |
| --- | --- | --- |
| `lincs/httpfile.py` `lincs/fetch_rows.py` | — | Range reads over the LINCS2020 level-5 GCTX, which is uncompressed contiguous HDF5, so only the required rows are downloaded |
| `lincs/build_ds.py` | — | Assemble the THP-1 compound signature table from `siginfo` and `compoundinfo` |
| `s1_preprocess.py` | `RCL/code/pre_process.R` | Merge replicates per SMILES by correlation-weighted averaging; `cor_mean` is the mean replicate correlation; single-replicate compounds are dropped |
| `s2_predictability.py` | `RCL/code/cal_predictability.py` | Per-gene random-forest regression on Morgan fingerprints, 10-fold, real versus shuffled labels, paired t-test |
| `s3_split.py` | `RCL/code/train_test_split.R` | Keep genes with `P_MSE < 0.05`; `cor_mean > 0.7` is high quality, 10% of those form the test set; three classes at ±1.5 |
| `s4_rcl.py` | `RCL/code/main_multi.py`, `DrugGene.py`, `loss.py` | Robust co-teaching training. `--init none` trains from scratch, `--init gps_mcf7` initialises from the published MCF7 weights, `--init eval_mcf7` evaluates those weights without training |
| `s5_predict.py` | `apply_RCL.py` | Per compound and gene, up/none/down probabilities. The published rule needs a class probability above 0.95; a continuous score `s = P(up) − P(down)` is also written for ranking |
| `s6_exp.py` | — | Comparison of starting points: MCF7 weights, four-cell-line weights, and training from scratch |
| `s7_es.py` | — | **The model used in the manuscript.** Same as `s6_exp.py` with a validation set drawn from 10% of high-quality training compounds and epoch selection by validation macro-F1, so the test set plays no part in selection |
| `s8_more.py` | — | Two further transfer variants, frozen first layer and L2-SP regularisation |
| `s9_drugdisc.py` | — | Ability to discriminate between compounds: per gene, Spearman correlation between predicted score and measured z across test compounds, averaged over genes, which removes the gene prior |
| `s10_center.py` | — | The THP-1 common component, the per-gene mean prediction over the 999 training compounds. The compound-specific profile is the prediction minus this component |
| `s11_pergene_local.py` | — | Per-gene Spearman distribution of the final models against the published GPS weights |

`run_s6.sh`, `run_s7.sh` and `run_s8.sh` are the seed sweeps used for those three
stages, five seeds each.

One deliberate change from the reference code: dropout is disabled at inference.
The original `F.dropout2d` call leaves `training=True` by default, so dropout was
still active during evaluation.

## Stage 2 — UNAGI perturbation (`unagi/`)

| Script | Purpose |
| --- | --- |
| `run_v8.py` | Train the UNAGI model. Stages are the four macrophage subpopulations ordered by pseudotime, `Macro-CDKN1C → Macro-CLEC4E → Macro-SEPTIN7 → Macro-DEFB1` |
| `antisilicosis.py` | Drug perturbation scoring. Disease direction comes from the iDREM tendency, last stage minus first, at a threshold of 0.1. Drug genes are those with `P.Value < 0.05` that are present in the model and change in the direction opposite to disease. Perturbation uses `log2fc = 0.5` against a gene-count-matched random background |
| `fig5c_silicosis.py` | Figure 5c in the style of the UNAGI reference figure, with PAGA arc edges and the C, S and P nodes |
| `reversal_v2.py` | Centroid-distance reversal analysis. Fold changes are applied in linear space as `L' = log1p(2^logFC · expm1(L))`, and the main readout is the percentage of the control gap closed |
| `percell.py` | Per-cell version, since a centroid can hide a population where half the cells move the wrong way |
| `null_control.py` | Encodes the same data twice with no perturbation to measure encoding noise, plus a perturbation with all fold changes set to one |
| `sweep_res.py` | Offline sweep of the Leiden resolution over a cached checkpoint to find values that let `connectNodes` form edges |
| `root_sensitivity.R` | Trajectory order under twelve different root choices |
| `traj_quality_test2.R` | A/B comparison reproducing the original trajectory procedure |

**A parameter deviation that must be reported.** With the default
`connect_edges_cutoff` of 0.05, the smallest p value between stage 0 and stage 1
is 0.0504, so no edge forms and training fails. It is set to 0.10, which yields
13, 16 and 12 edges for the three consecutive pairs. It is a documented argument
of `run_UNAGI()`.

## Inputs

Paths are supplied through environment variables; nothing is hard coded. Copy
`env.example` and edit it.

| Variable | Contents |
| --- | --- |
| `WORK_DIR` | Working directory for the LINCS download stage |
| `GPSRCL_DIR` | Directory holding the `s*.py` scripts when run through the shell wrappers |
| `GPS_CODE` | A checkout of the GPS reference code, used for its `MLP` definition and published per-cell-line weights |
| `CONDA_BIN` | Interpreter directory used by the shell wrappers |
| `GPS_SELECTED_GENES` | `selected_genes_2198.csv` from the GPS gene-feature data |
| `FINETUNED_MODELS` | Directory holding the fine-tuned checkpoints |
| `PERGENE_OUT` | Output path for the per-gene correlation table |
| `UNAGI_SRC` | A checkout of UNAGI |
| `UNAGI_IDREM` | The iDREM directory inside that checkout |
| `UNAGI_DRUG` | The drug-signature directory inside that checkout |
| `SIL_H5AD` | Pro-fibrotic branch macrophages as AnnData |
| `SIL_MACRO_QS` | The same cells as a `qs` object, for the R scripts |
| `SIL_PLOTDATA` | Pseudotime table |
| `ITCM_DIR` | Per-compound differential expression from ITCM |
| `GPS_CONTINUOUS` | Continuous GPS profiles |
| `THP1_DRUG_GENES` | Pickled per-compound gene sets derived from stage 1 |
| `MODEL_DATA` | UNAGI model directory to read or write |
| `TRAJ_TEST_OUT` | Output directory for the trajectory tests |

External data must be obtained from its own sources: LINCS2020 level 5 from
clue.io, the GPS reference code and weights and the UNAGI code from their own
releases, and ITCM differential expression from ITCM.

## Environments

```
# stage 1
torch, rdkit, scikit-learn, pandas, numpy, scipy, joblib

# stage 2
torch 2.4.1+cu121, scanpy, anndata, pyro-ppl, leidenalg, python-igraph
```

The R scripts need `qs`, `monocle3` and `irlba` 2.4.1 or newer.

## Licence

MIT, see `LICENSE`.
