# Real-Data Feature And Graph Stress-Test Evidence

## Scope

This package stress-tests PathNet on the real COVID-19 and BRCA datasets by degrading either the centric-feature choices or the graph/connection prior while keeping the original notebook preprocessing, split protocol, hyperparameters, and epoch counts fixed.

These are real-data robustness/stability experiments, not ground-truth recovery experiments. Feature and edge recovery should be evaluated later in simulation.

## Run Configuration

- Mode: `full`
- Datasets: `covid, brca`
- Conditions: `original, random_centric, low_signal_centric, mixed25_low_signal_centric, mixed50_low_signal_centric, mixed75_low_signal_centric, random_graph, random_layer_matched, partial_outer_layer_random`
- Split: stratified 60/20/20 train/validation/test, matching the existing real-data PathNet benchmark.
- Seeds: inherited from the selected benchmark mode.

## Summary

| dataset | condition                  | n_runs | test_roc_auc_mean | test_roc_auc_std | test_balanced_accuracy_mean | test_balanced_accuracy_std | test_accuracy_mean | test_accuracy_std | target_signal_score_mean_mean |
| ------- | -------------------------- | ------ | ----------------- | ---------------- | --------------------------- | -------------------------- | ------------------ | ----------------- | ----------------------------- |
| brca    | low_signal_centric         | 5      | 0.684             | 0.218            | 0.656                       | 0.216                      | 0.835              | 0.088             | 0.001                         |
| brca    | mixed25_low_signal_centric | 5      | 0.870             | 0.207            | 0.835                       | 0.189                      | 0.907              | 0.077             | 0.731                         |
| brca    | mixed50_low_signal_centric | 5      | 0.954             | 0.025            | 0.899                       | 0.044                      | 0.928              | 0.030             | 0.478                         |
| brca    | mixed75_low_signal_centric | 5      | 0.965             | 0.010            | 0.910                       | 0.016                      | 0.929              | 0.020             | 0.258                         |
| brca    | original                   | 5      | 0.955             | 0.017            | 0.888                       | 0.035                      | 0.923              | 0.021             | 0.970                         |
| brca    | partial_outer_layer_random | 5      | 0.958             | 0.019            | 0.908                       | 0.031                      | 0.939              | 0.014             | 0.970                         |
| brca    | random_centric             | 5      | 0.752             | 0.272            | 0.751                       | 0.229                      | 0.876              | 0.096             | 0.452                         |
| brca    | random_graph               | 5      | 0.955             | 0.017            | 0.888                       | 0.040                      | 0.923              | 0.025             | 0.970                         |
| brca    | random_layer_matched       | 5      | 0.961             | 0.016            | 0.889                       | 0.040                      | 0.928              | 0.024             | 0.970                         |
| covid   | low_signal_centric         | 5      | 0.656             | 0.049            | 0.617                       | 0.048                      | 0.636              | 0.046             | 0.002                         |
| covid   | mixed25_low_signal_centric | 5      | 0.694             | 0.054            | 0.632                       | 0.044                      | 0.651              | 0.042             | 0.119                         |
| covid   | mixed50_low_signal_centric | 5      | 0.683             | 0.060            | 0.644                       | 0.048                      | 0.663              | 0.036             | 0.088                         |
| covid   | mixed75_low_signal_centric | 5      | 0.671             | 0.027            | 0.608                       | 0.037                      | 0.640              | 0.041             | 0.034                         |
| covid   | original                   | 5      | 0.746             | 0.054            | 0.698                       | 0.055                      | 0.707              | 0.051             | 0.158                         |
| covid   | partial_outer_layer_random | 5      | 0.721             | 0.057            | 0.637                       | 0.041                      | 0.653              | 0.042             | 0.158                         |
| covid   | random_centric             | 5      | 0.633             | 0.035            | 0.591                       | 0.044                      | 0.623              | 0.034             | 0.134                         |
| covid   | random_graph               | 5      | 0.690             | 0.050            | 0.641                       | 0.028                      | 0.666              | 0.027             | 0.158                         |
| covid   | random_layer_matched       | 5      | 0.722             | 0.055            | 0.649                       | 0.045                      | 0.674              | 0.045             | 0.158                         |

## Interpretation Notes

- Poor centric-feature choices reduced real-data performance and/or stability, especially on COVID and in several BRCA splits. On COVID, random centric features reduced mean test ROC-AUC from 0.746 to 0.633 and low-signal centric features reduced it to 0.656. On BRCA, random and low-signal centric features had large seed-to-seed variability, with ROC-AUC standard deviations of 0.272 and 0.218, respectively.
- Mixed centric-feature settings were not strictly monotonic on real data. BRCA remained strong for the 50% and 75% mixed settings, while the 25% mixed condition had one unstable split. This suggests that real datasets can contain proxy signal outside the original anchors, but it also supports reporting centric-feature selection as a biological/scientific decision rather than a purely automatic step.
- Graph-prior corruption did not necessarily destroy predictive performance when the original target node set was retained. On BRCA, the random graph, random layer-matched, and partial outer-layer-random conditions all remained close to the original predictive performance. On COVID, these graph-prior stress conditions reduced balanced accuracy modestly but did not collapse.
- The graph-prior result should be interpreted carefully: preserving useful target-centered nodes can preserve prediction, but randomized edges weaken confidence in edge-level pathway interpretation. This is useful for rebuttal wording because it separates predictive robustness from biological interpretability.
- Increased standard deviation or wider min-max range should be interpreted as instability under the corresponding stress condition, not as a direct ground-truth recovery metric.

## Reviewer-Comment Use

- `R2.1`: This is useful real-data stress evidence for misleading or partially incorrect prior knowledge. It supports the answer that PathNet can preserve prediction under some graph perturbations when informative nodes remain available, but edge-level interpretation becomes less reliable and should be treated cautiously. A controlled simulation is still needed for true feature/edge recovery.
- `R1.1`: This supports the feature-selection response. Bad or low-signal centric-feature choices can reduce performance or increase instability, so users should choose centric features using wet-lab evidence, prior biological knowledge, differential analysis, or another explicit scientific rationale.
- `R2.3`: This provides partial evidence that feature-selection quality matters in real data, but it does not replace the later sample/feature scaling and moderate-effect-size simulation.
- `R1.8`: This adds additional real-data stress-test results under realistic imperfect-knowledge scenarios.

## Files

- Per-run results: `tables/real_data_stress_results.csv`
- Summary table: `tables/real_data_stress_summary.csv`
- Feature-selection stress table: `tables/real_data_feature_selection_stress.csv`
- Graph-prior stress table: `tables/real_data_graph_prior_stress.csv`
- ROC-AUC figure: `figures/real_data_stress_roc_auc.pdf`
- Balanced-accuracy figure: `figures/real_data_stress_balanced_accuracy.pdf`
