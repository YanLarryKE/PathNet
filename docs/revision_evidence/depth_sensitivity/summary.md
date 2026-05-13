# Real-Data Depth Sensitivity Evidence

## Scope

This package evaluates how the maximum graph depth from centric features affects real-data PathNet performance and model size. Preprocessing, labels, split protocol, hyperparameters, and epoch counts match the existing real-data PathNet benchmark; only `maximum_step` is varied.

## Run Configuration

- Mode: `full`
- Datasets: `covid, brca`
- Trained depths: `1, 2, 3`
- Architecture-scan depths: `1, 2, 3, 4, 5`
- Split: stratified 60/20/20 train/validation/test, matching the model benchmark.
- Seeds: inherited from the selected benchmark mode.

## Trained Depth Summary

| dataset | maximum_step | n_runs | test_roc_auc_mean | test_roc_auc_std | test_balanced_accuracy_mean | test_balanced_accuracy_std | test_accuracy_mean | test_accuracy_std | train_seconds_mean | n_parameters_mean | n_sparse_connections_mean |
| ------- | ------------ | ------ | ----------------- | ---------------- | --------------------------- | -------------------------- | ------------------ | ----------------- | ------------------ | ----------------- | ------------------------- |
| brca    | 1            | 5      | 0.792             | 0.352            | 0.818                       | 0.183                      | 0.895              | 0.073             | 51.858             | 15183430.000      | 61206.000                 |
| brca    | 2            | 5      | 0.955             | 0.018            | 0.883                       | 0.025                      | 0.922              | 0.015             | 218.651            | 70801138.000      | 73084.000                 |
| brca    | 3            | 5      | 0.953             | 0.021            | 0.905                       | 0.027                      | 0.937              | 0.019             | 795.212            | 255823382.000     | 76504.000                 |
| covid   | 1            | 5      | 0.693             | 0.075            | 0.646                       | 0.068                      | 0.671              | 0.061             | 53.132             | 4435150.000       | 2441.000                  |
| covid   | 2            | 5      | 0.749             | 0.042            | 0.691                       | 0.040                      | 0.698              | 0.033             | 66.144             | 4666458.000       | 2559.000                  |
| covid   | 3            | 5      | 0.689             | 0.025            | 0.628                       | 0.039                      | 0.655              | 0.032             | 77.534             | 5122606.000       | 2661.000                  |

## Architecture Growth

| dataset | maximum_step | connection_layer_sizes          | n_sparse_connections | n_parameters | run_status |
| ------- | ------------ | ------------------------------- | -------------------- | ------------ | ---------- |
| covid   | 1            | 913|40|10                       | 2441                 | 4435150      | ok         |
| covid   | 2            | 913|99|40|10                    | 2559                 | 4666458      | ok         |
| covid   | 3            | 913|202|99|40|10                | 2661                 | 5122606      | ok         |
| covid   | 4            | 913|288|202|99|40|10            | 2729                 | 5669382      | ok         |
| covid   | 5            | 913|350|288|202|99|40|10        | 2774                 | 6299006      | ok         |
| brca    | 1            | 8523|111|28                     | 61206                | 15183430     | ok         |
| brca    | 2            | 8523|1720|111|28                | 73084                | 70801138     | ok         |
| brca    | 3            | 8523|5947|1720|111|28           | 76504                | 255823382    | ok         |
| brca    | 4            | 8523|7733|5947|1720|111|28      | 76948                | 500664298    | ok         |
| brca    | 5            | 8523|8081|7733|5947|1720|111|28 | 77027                | 762489806    | ok         |

## Interpretation Notes

- Depth controls how broad a graph neighborhood around the centric features is included.
- Depth 2 is the original notebook setting for both real datasets and is therefore the primary reference.
- Larger depths can include more indirect pathway context, but they also increase the number of included nodes, sparse connections, parameters, and training time.
- On COVID, depth 2 gave the strongest mean held-out performance among the trained depths: ROC-AUC 0.749 and balanced accuracy 0.691. Depth 1 was lower, and depth 3 did not improve performance despite the larger neighborhood.
- On BRCA, depth 2 and depth 3 had similar ROC-AUC (0.955 and 0.953), while depth 3 had somewhat higher balanced accuracy (0.905 vs 0.883). However, this came with a large computational cost: mean training time increased from 218.7 s to 795.2 s and parameter count increased from 70.8M to 255.8M.
- BRCA depth 1 was unstable across seeds because one split collapsed to low ROC-AUC, leading to a large ROC-AUC standard deviation (0.352). This suggests that overly shallow neighborhoods can omit useful pathway context for some train/test splits.
- BRCA grows sharply after depth 3 in the architecture scan: depth 4 would require about 500.7M parameters and depth 5 about 762.5M parameters. These larger depths are recorded as model-size evidence but are not trained by default in the full sweep.
- Overall, the real-data evidence supports choosing graph depth by validation performance, neighborhood size, and computational budget. In these two datasets, the original depth 2 setting is a reasonable default because it improves over an overly shallow depth and avoids the much larger cost of deeper BRCA neighborhoods.
- These real-data results address predictive and computational sensitivity. Ground-truth subnetwork recovery still requires simulation.

## Reviewer-Comment Use

- `R1.3`: direct real-data evidence for graph-depth sensitivity and practical depth selection.
- `R1.5`: additional computational-performance evidence as graph neighborhoods grow.
- `R2.3`: partial scalability evidence because depth changes pathway-neighborhood size and parameter count.

## Files

- Per-run results: `tables/real_data_depth_results.csv`
- Summary table: `tables/real_data_depth_summary.csv`
- Architecture scan: `tables/real_data_depth_architecture.csv`
- ROC-AUC figure: `figures/real_data_depth_roc_auc.pdf`
- Balanced-accuracy figure: `figures/real_data_depth_balanced_accuracy.pdf`
- Architecture-growth figure: `figures/real_data_depth_model_size.pdf`
