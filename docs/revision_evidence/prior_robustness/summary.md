# Simulation Prior-Robustness Evidence

## Scope

This evidence package focuses on imperfect biological knowledge in controlled simulations. The MLP/KAN comparison is intentionally left to the real-data benchmark evidence; the simulation here tests whether PathNet benefits from a correct graph prior and appropriate centric-feature selection when ground truth is known.

## Real-Data Experiments Already Completed

- Model benchmark: PathNet, MLP, and efficient-KAN on COVID-19 and BRCA, using full and selected feature sets. Evidence: `docs/revision_evidence/model_benchmark/summary.md`.
- Real-data stress tests: original PathNet setting, random/low-signal/mixed centric features, random graph, layer-matched random sparse masks, and partial graph randomization. Evidence: `docs/revision_evidence/real_data_stress/summary.md`.
- Depth sensitivity: graph depth 1/2/3 trained on both real datasets, with larger-depth architecture scans. Evidence: `docs/revision_evidence/depth_sensitivity/summary.md`.

## Why The Old Simulation Was Ambiguous

The generator in `syn_data.ipynb` places class-dependent signal directly in source nodes. A random graph can therefore perform well if it accidentally routes even a few informative source nodes into the selected target neighborhood. The old logs under `logs/fixed_graph_simulation/` show this behavior: random or bad graph conditions are sometimes close to, or better than, the nominal graph.

This is not necessarily a bug in the model; it is a simulation-design issue. If the wrong graph is allowed to randomly include the real signal route, it is not a fully wrong prior. In the new simulation, we therefore separate two cases:

- `random_graph`: an unrestricted density-matched random graph; it can occasionally route true source nodes and is expected to be high-variance.
- `wrong_graph_decoy`: a density-matched wrong graph with no short route from selected true targets to true source nodes; this tests a genuinely wrong prior.

## Simulation Mechanism

- The graph is modular, with true centric features placed at module targets.
- Informative source nodes are three graph hops away from true targets.
- True target nodes do not directly receive the class signal; PathNet must route signed source signal through the graph neighborhood.
- Bad centric features are matched targets from disconnected decoy modules.
- The default run uses two synthetic dataset sizes, five seeds each, and 11 imperfect-knowledge conditions.

## Summary Results

| Condition | Runs | ROC-AUC | Balanced accuracy | Routed sources |
| --- | ---: | ---: | ---: | ---: |
| `correct_graph` | 10 | 0.976 +/- 0.014 | 0.970 +/- 0.020 | 36.0 |
| `correct_graph_depth1` | 10 | 0.515 +/- 0.051 | 0.501 +/- 0.041 | 0.0 |
| `correct_graph_depth3` | 10 | 0.978 +/- 0.011 | 0.970 +/- 0.018 | 36.0 |
| `random_graph` | 10 | 0.893 +/- 0.172 | 0.869 +/- 0.167 | 2.0 |
| `wrong_graph_decoy` | 10 | 0.494 +/- 0.051 | 0.497 +/- 0.045 | 0.0 |
| `random_layer_matched` | 10 | 0.860 +/- 0.168 | 0.815 +/- 0.161 | 2.3 |
| `partial_inner_random` | 10 | 0.979 +/- 0.014 | 0.972 +/- 0.022 | 36.0 |
| `bad_feature` | 10 | 0.502 +/- 0.027 | 0.503 +/- 0.033 | 0.0 |
| `mixed25_bad_feature` | 10 | 0.978 +/- 0.015 | 0.970 +/- 0.023 | 24.0 |
| `mixed50_bad_feature` | 10 | 0.976 +/- 0.013 | 0.970 +/- 0.014 | 18.0 |
| `mixed75_bad_feature` | 10 | 0.970 +/- 0.021 | 0.952 +/- 0.035 | 12.0 |

## Interpretation

- The correct graph is high and stable, while a strictly wrong graph and fully bad feature selection are close to chance.
- A shallower-than-needed depth fails because source nodes are three hops away from the centric features.
- Unrestricted random graphs can occasionally perform well; this is expected because they may randomly route some true source nodes. Their standard deviation is much larger than the correct graph condition.
- Partial inner randomization remains strong when the useful node set is preserved. This supports a careful distinction: prediction can remain robust, but edge-level interpretation should be treated cautiously when the wiring is wrong.
- Mixed bad-feature settings remain high when enough true targets are retained. This is acceptable and biologically plausible: partially correct feature choices can still preserve the relevant signal.

## Reviewer-Comment Use

- `R2.1`: direct controlled evidence for imperfect or misleading prior knowledge. Correct prior is more stable; fully wrong priors and fully bad centric features fail.
- `R1.1`: supports the response that centric-feature selection should be guided by biological rationale or wet-lab evidence.
- `R1.3`: supports depth sensitivity by showing depth 1 misses the signal while depth 2/3 recover it.
- `R1.8` and `R2.3`: contributes simulation evidence for robustness and scaling-related stress tests, alongside the real-data evidence.

## Files

- Reproducible script: `scripts/simulation_prior_robustness.py`
- Simulation-design/results figure: `figures/prior_robustness_simulation_setting.pdf`
- Figure-generation script: `scripts/plot_simulation_setting.py`
- Per-run results: `tables/simulation_prior_robustness_results_full.csv`
- Summary table: `tables/simulation_prior_robustness_summary_full.csv`
- Dataset metadata: `tables/simulation_dataset_summary_full.csv`
- Run metadata: `metadata_full.json`
