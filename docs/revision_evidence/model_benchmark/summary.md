# Real-Data Model Benchmark Evidence

## Scope

This evidence package currently contains two related benchmark components:

- black-box MLP and efficient-KAN baselines on full and selected features;
- PathNet reruns that follow the two existing example notebooks' settings and epoch counts, evaluated with a held-out test split across the same five random seeds.

The real datasets are:

- COVID-19 LC-MS metabolomics, following `example_meta.ipynb`.
- TCGA BRCA gene/miRNA data, following `example_gene.ipynb`.

For MLP/KAN, each dataset was trained on both full notebook-retained features and selected features associated with the paper's target features. For PathNet, the reruns use the original graph-constrained notebook inputs and hyperparameters.

## Data Definitions

- COVID label: `cov == Yes` and ICU `Yes` vs `No`; ICU `Yes` is encoded as 1.
- COVID full features: all 1,174 LC-MS measured features retained by PathNet preprocessing.
- COVID selected features: 27 measured LC-MS features mapped to the 10 target KEGG IDs in `example_meta.ipynb`.
- BRCA label: ER `Positive` vs `Negative`; `Indeterminate` labels were excluded for a clean binary benchmark.
- BRCA full features: 8,857 notebook-retained network genes plus filtered miRNAs.
- BRCA selected features: 63 target-gene-associated features, consisting of the 28 target genes in `example_gene.ipynb` plus filtered miRNAs mapped to those target genes.
- PathNet BRCA rerun label: follows `example_gene.ipynb` exactly, where ER `Positive` is encoded as 1 and all other non-missing labels are encoded as 0. This includes 2 `Indeterminate` samples in class 0. This is noted as a comparability caveat against the cleaner MLP/KAN BRCA binary benchmark.

## Model Grid

Each model/dataset/feature-set combination was run with five random seeds:

- MLP: `[64]`, `[128, 64]`, `[256, 128, 64]`.
- efficient-KAN: `[16]`, `[32]`, `[32, 16]` with grid size 3 and spline order 2.

The KAN backend was installed into `phynn` from:

`https://github.com/Blealtan/efficient-kan.git`

PathNet was rerun with the original notebook hyperparameters across five random seeds `[2781, 14526, 2027, 3407, 91011]`:

- COVID PathNet: maximum graph step 2, no extra dense hidden layer, dropout 0.1, learning rate 1e-3, weight decay 0, batch size 16, 200 epochs.
- BRCA PathNet: maximum graph step 2, no extra dense hidden layer, dropout 0.1, learning rate 1e-3, weight decay 0, batch size 16, 100 epochs.

The PathNet rerun uses a stratified 60/20/20 train/validation/test split for held-out test metrics. It does not apply early stopping, matching the fixed-epoch notebook style.

## Best Mean Test ROC-AUC By Model

The table below selects the best architecture within each dataset, feature set, and model family by mean test ROC-AUC.

| Dataset | Feature set | Model | Best architecture | Mean ROC-AUC | Mean balanced accuracy |
| --- | --- | --- | --- | ---: | ---: |
| BRCA | full | KAN | `kan_16_g3` | 0.965 | 0.933 |
| BRCA | full | MLP | `mlp_128_64` | 0.959 | 0.891 |
| BRCA | selected | KAN | `kan_32_g3` | 0.955 | 0.851 |
| BRCA | selected | MLP | `mlp_256_128_64` | 0.953 | 0.869 |
| COVID | full | KAN | `kan_16_g3` | 0.769 | 0.690 |
| COVID | full | MLP | `mlp_128_64` | 0.833 | 0.724 |
| COVID | selected | KAN | `kan_16_g3` | 0.605 | 0.550 |
| COVID | selected | MLP | `mlp_128_64` | 0.631 | 0.577 |

## PathNet Notebook-Setting Rerun

These rows rerun the two existing PathNet examples using their notebook settings and epoch counts, with held-out test metrics added for rebuttal evidence.

| Dataset | Notebook | Runs | Epochs | Mean test ROC-AUC | Mean test balanced accuracy | Mean test accuracy |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| COVID | `example_meta.ipynb` | 5 | 200 | 0.749 $\pm$ 0.042 | 0.691 $\pm$ 0.040 | 0.698 $\pm$ 0.033 |
| BRCA | `example_gene.ipynb` | 5 | 100 | 0.955 $\pm$ 0.018 | 0.883 $\pm$ 0.025 | 0.922 $\pm$ 0.015 |

Because the PathNet rerun intentionally preserves the notebook settings, it is useful model-comparison evidence under the external held-out benchmark split. The remaining comparability caveat for `R1.4` is BRCA label handling: the PathNet row follows `example_gene.ipynb` exactly, whereas the MLP/KAN BRCA rows exclude Indeterminate labels for a clean binary benchmark.

## Dataset-Specific Comparison Snapshots

These tables combine the PathNet notebook rerun with the best MLP/KAN results within each dataset. All rows use five random seeds.

### BRCA

| Features/scope | Model | Setting | Runs | Test ROC-AUC | Test balanced accuracy | Test accuracy | Interpretability |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| PathNet notebook graph-constrained input | PathNet | notebook hyperparameters | 5 | 0.955 | 0.883 | 0.922 | Full pathway-constrained |
| Full notebook-retained features | KAN | `kan_16_g3` | 5 | 0.965 | 0.933 | 0.946 | Partial functional |
| Full notebook-retained features | MLP | `mlp_128_64` | 5 | 0.959 | 0.891 | 0.928 | No direct pathway |
| Selected target-associated features | KAN | `kan_32_g3` | 5 | 0.955 | 0.851 | 0.913 | Partial functional |
| Selected target-associated features | MLP | `mlp_256_128_64` | 5 | 0.953 | 0.869 | 0.916 | No direct pathway |

### COVID-19

| Features/scope | Model | Setting | Runs | Test ROC-AUC | Test balanced accuracy | Test accuracy | Interpretability |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| PathNet notebook graph-constrained input | PathNet | notebook hyperparameters | 5 | 0.749 | 0.691 | 0.698 | Full pathway-constrained |
| Full notebook-retained features | MLP | `mlp_128_64` | 5 | 0.833 | 0.724 | 0.749 | No direct pathway |
| Full notebook-retained features | KAN | `kan_16_g3` | 5 | 0.769 | 0.690 | 0.719 | Partial functional |
| Selected target-associated features | MLP | `mlp_128_64` | 5 | 0.631 | 0.577 | 0.651 | No direct pathway |
| Selected target-associated features | KAN | `kan_16_g3` | 5 | 0.605 | 0.550 | 0.618 | Partial functional |

The COVID-19 PathNet row reports the external held-out benchmark rerun and does not replace the main-text 0.725 accuracy from the original manuscript analysis setting.

## Feature Reduction

| Dataset | Full features | Selected features | Retained proportion | Reduction |
| --- | ---: | ---: | ---: | ---: |
| COVID | 1,174 | 27 | 2.3% | 97.7% |
| BRCA | 8,857 | 63 | 0.7% | 99.3% |

## Interpretability Comparison

| Model family | Pathway constraint | Intrinsic interpretability | Interpretation unit |
| --- | --- | --- | --- |
| Fully connected MLP | None | No direct pathway interpretability | Prediction-level or post hoc feature attribution only |
| KAN baseline | None in this benchmark | Partial functional interpretability | Generic learned nonlinear functions |
| PathNet | Full pathway-constrained sparse connectivity | Full pathway-based interpretability by design | Curated pathway edges, learned edge functions, and retained subnetworks |

SHAP can be a useful post hoc attribution tool for black-box models, but it is not a direct replacement for pathway-constrained interpretability in high-dimensional omics settings. Exact Shapley-value attribution scales combinatorially with the number of input features, and practical model-agnostic Kernel SHAP relies on sampling many masked feature coalitions and repeated model evaluations. Direct use on full inputs such as the 1,174-feature COVID matrix or the 8,857-feature BRCA matrix would therefore require feature screening, grouping, or pathway-level aggregation.

## Reviewer-comment Coverage

- `R2.2`: substantially answered by this evidence package. It supports the response that KAN is feasible as an alternative approximator, while GNN/SHAP extensions should preserve pathway-edge semantics to retain interpretability.
- `R2.3`: partially answered. The real-data feature-reduction ratios and training times are available, but synthetic scaling and moderate-effect-size stress tests are still needed for a complete response.
- `R1.4`: answered for the real-data performance trade-off. The package now combines five-seed PathNet notebook-setting reruns with full/selected-feature MLP and KAN black-box baselines; the BRCA label-handling caveat is noted explicitly.
- `R1.5`: answered for method-detail clarification. The benchmark contributes training-time, parameter-count, sparse-connection, and tuning-grid evidence, and the manuscript now expands the assumptions, training procedure, tuning guidance, and computational-performance reporting.
- `R1.8`: partially answered. The package adds real-data results, but the requested additional simulation settings and biological interpretation are still separate tasks.

## Files

- Full per-run results: `tables/real_data_mlp_kan_results_full.csv`
- Summary by architecture: `tables/real_data_mlp_kan_summary_full.csv`
- Best architecture by model family: `tables/real_data_mlp_kan_best_by_model_full.csv`
- PathNet notebook-setting single-seed results: `tables/real_data_pathnet_results_notebook.csv`
- PathNet five-seed results: `tables/real_data_pathnet_results_full.csv`
- PathNet five-seed epoch history: `tables/real_data_pathnet_epoch_history_full.csv`
- PathNet five-seed summary: `tables/real_data_pathnet_summary_full.csv`
- PathNet/MLP/KAN comparison snapshot: `tables/real_data_pathnet_mlp_kan_comparison.csv`
- BRCA-specific comparison snapshot: `tables/real_data_brca_pathnet_mlp_kan_comparison.csv`
- COVID-specific comparison snapshot: `tables/real_data_covid_pathnet_mlp_kan_comparison.csv`
- ROC-AUC figure: `figures/real_data_mlp_kan_auc.pdf`
- Balanced accuracy figure: `figures/real_data_mlp_kan_balanced_accuracy.pdf`
- Reproducible script copy: `scripts/real_data_mlp_kan.py`
- PathNet reproducible script copy: `scripts/real_data_pathnet.py`
- MLP/KAN run metadata: `metadata.json`
- PathNet single-seed run metadata: `metadata_pathnet_notebook.json`
- PathNet five-seed run metadata: `metadata_pathnet_full.json`

## Interpretation Notes

- The MLP/KAN rows are black-box baselines. The PathNet notebook rerun now gives five-seed PathNet-vs-black-box evidence under the external held-out benchmark split; BRCA still carries the label-handling caveat noted above.
- Full-feature MLPs perform strongly on COVID; selected COVID features are much weaker for black-box prediction, which is useful context when discussing the difference between prediction and pathway-focused interpretation.
- On BRCA, selected features retain performance close to full-feature baselines, suggesting substantial predictive information in the target-gene-associated feature subset.
- KAN is competitive with MLP on BRCA, while MLP is stronger on COVID in this grid. This supports a cautious statement that KAN is a feasible alternative approximator, but not uniformly better.
- Under notebook hyperparameters, PathNet is competitive with full-feature black-box baselines on BRCA and is lower than full-feature MLP on COVID, while retaining pathway-constrained interpretability. This supports a transparent trade-off framing: the knowledge-constrained model prioritizes pathway-level interpretability and mechanistic inspection while maintaining reasonable predictive performance.
