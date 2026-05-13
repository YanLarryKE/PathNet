# Real-Data Feature And Graph Stress-Test Outline

## Purpose

This experiment package tests how PathNet behaves on the real COVID-19 metabolomics and BRCA gene/miRNA datasets when either the centric-feature choice or the pathway graph prior is intentionally degraded. The goal is not to claim ground-truth subnetwork recovery on real data, because the true causal pathway is unknown. Instead, the goal is to quantify predictive performance, run-to-run stability, and interpretability sensitivity under realistic imperfect-user and imperfect-knowledge scenarios.

These experiments are exploratory real-data stress tests. The final simulation package should later repeat the same logic in a setting with known ground truth, so that feature recovery and edge/subnetwork recovery can be evaluated directly.

## Base Setting To Keep Fixed

- Datasets: the same real COVID-19 LC-MS metabolomics and TCGA BRCA gene/miRNA datasets used in the current PathNet benchmark.
- Preprocessing: follow `example_meta.ipynb`, `example_gene.ipynb`, and `docs/revision_evidence/model_benchmark/scripts/real_data_pathnet.py`.
- Splits: use the existing stratified 60/20/20 train/validation/test split.
- Seeds: use the same five benchmark seeds, `2781`, `14526`, `2027`, `3407`, and `91011`.
- Training: keep the notebook hyperparameters and epoch counts fixed unless a condition becomes infeasible:
  - COVID: maximum graph step 2, dropout 0.1, learning rate 1e-3, batch size 16, 200 epochs.
  - BRCA: maximum graph step 2, dropout 0.1, learning rate 1e-3, batch size 16, 100 epochs.
- Baseline comparison: report the already completed original PathNet benchmark as the reference condition, rather than rerunning it unless needed for code consistency.

Keeping these choices fixed makes the stress conditions interpretable: changes in mean performance and standard deviation can be attributed primarily to feature-selection or graph-prior degradation.

## Experiment 1: Centric-Feature Selection Quality

### Conditions

1. Original centric features.
   - Use the paper's current target KEGG IDs for COVID and target genes for BRCA.
   - This is the reference condition.

2. Random centric features.
   - Replace the original centric features with randomly sampled graph nodes that are present in the measured/mapped data.
   - Match the number of centric features to the original setting.
   - Repeat random centric-feature sampling across several selection seeds, or tie one selection seed to each train/test seed.

3. Low-signal centric features.
   - Select graph nodes with weak univariate association with the outcome, computed using only the training split to avoid leakage.
   - Match the number of centric features to the original setting.
   - This represents the user's concern that some selected anchors may be almost meaningless.

4. Mixed centric features.
   - Replace a fixed fraction of the original centric features with random or low-signal features.
   - Suggested replacement levels: 25%, 50%, and 75%.
   - This tests the realistic case where some selected features are biologically meaningful while others are not.

### Hypothesis

Good centric features should give the strongest and most stable performance. Completely random or low-signal centric features should reduce accuracy/ROC-AUC and increase run-to-run variability. Mixed centric features should degrade gradually rather than catastrophically if enough meaningful anchors remain.

### Interpretation

This directly supports the R1.1 response: feature selection is intentionally flexible, but the starting features should be anchored in biological knowledge, wet-lab evidence, differential analysis, or another external rationale. It also helps R2.3 by showing what happens when feature selection quality is imperfect.

## Experiment 2: Completely Wrong Graph Prior

### Conditions

1. Original graph.
   - Use the curated KEGG or gene interaction graph.

2. Fully random graph with matched size.
   - Keep the same node set as the original graph, but replace edges with random node pairs.
   - Prefer matching the total number of edges and, if straightforward, approximately matching the degree distribution.
   - This corresponds to a misleading prior where any feature can be connected without biological constraint.

3. Random neighborhood expansion.
   - Preserve the number of nodes per PathNet layer from the original graph, but connect newly included nodes at random.
   - This is close to the current `abla_graph` behavior in `getPartitionMatricesList`, but should be made deterministic by seed and explicitly labeled.

### Hypothesis

A completely wrong graph should produce weaker and less stable performance. The most useful signal here is not only lower mean accuracy or ROC-AUC, but also larger standard deviation, wider min-max range across seeds, and less stable selected subnetworks or sparse connections.

### Interpretation

This gives real-data stress evidence for R2.1: misleading prior knowledge can hurt both prediction and interpretability. Because real data do not provide the true underlying subnetwork, the answer should frame this as predictive/stability evidence and reserve feature/edge recovery claims for the later simulation package.

## Experiment 3: Partially Wrong Graph Prior

### Conditions

1. Original graph.
   - Reference condition.

2. Depth-specific randomization between step 2 and step 3.
   - Preserve the graph-derived nodes up to the earlier neighborhood.
   - For features introduced between step 2 and step 3, randomize their connections rather than using the curated pathway edges.
   - This matches the user's previous exploratory setting: the node set may still contain useful biology, but the local wiring is partially wrong.

3. Edge-rewiring gradient.
   - Rewire a controlled fraction of eligible edges while preserving the node set.
   - Suggested corruption levels: 10%, 25%, 50%, and 100%.
   - If computation is a concern, start with 25%, 50%, and 100%.

### Hypothesis

Partially wrong graph priors may still perform reasonably well when the informative node set is preserved, especially if the corrupted region is farther from the centric features. However, edge-level biological interpretation should be weakened as rewiring increases. This distinction is important: preserving useful nodes can maintain prediction, while corrupting edges can reduce confidence in pathway-level explanations.

### Interpretation

This complements R2.1 by separating two failure modes:

- wrong feature/node inclusion, which affects what information reaches the model;
- wrong edge wiring, which affects pathway-level interpretation even when predictive signal remains.

It also supports R1.2 and R1.7 indirectly by clarifying that graph-based interpretation depends on how the graph is used and displayed.

## Metrics

For each dataset and condition, report:

- test accuracy, balanced accuracy, and ROC-AUC;
- mean, standard deviation, and min-max range across seeds;
- validation metrics for diagnosing overfitting or unstable training;
- number of graph nodes included by the PathNet construction;
- number of sparse graph connections;
- trainable parameter count;
- training time;
- if available, overlap/Jaccard stability of retained nodes or edges after pruning.

The main table should emphasize mean performance and stability. A figure can show mean ROC-AUC or balanced accuracy with error bars, grouped by stress condition.

## Evidence Outputs

Planned durable outputs:

- `docs/revision_evidence/real_data_stress/tables/real_data_feature_selection_stress.csv`
- `docs/revision_evidence/real_data_stress/tables/real_data_graph_prior_stress.csv`
- `docs/revision_evidence/real_data_stress/tables/real_data_stress_summary.csv`
- `not included in this public snapshot`
- `not included in this public snapshot`
- `docs/revision_evidence/real_data_stress/metadata.json`

Experiment code should live under:

- `docs/revision_evidence/real_data_stress/scripts/`

## Reviewer Comments Supported

- `R2.1`: primary real-data stress evidence for misleading or partially incorrect prior knowledge.
- `R1.1`: evidence that centric-feature choice matters and should be biologically/question driven.
- `R2.3`: partial evidence for feature-selection quality and information-loss concerns.
- `R1.8`: additional real-data results under realistic imperfect-knowledge scenarios.
- `R1.4`: contextual support for why pathway-constrained models should be evaluated jointly on predictive performance and interpretability, not accuracy alone.

## Limitations To State In The Rebuttal

- These real-data experiments do not establish true subnetwork recovery because the ground-truth biological network is unknown.
- Lower or more variable performance under corrupted priors supports the importance of prior quality, but it does not identify exactly which real pathway edges are correct.
- The later simulation package is still needed to quantify feature recovery, edge recovery, and behavior under controlled moderate effect sizes.

## Suggested Implementation Notes

- Start by extending the existing PathNet real-data benchmark rather than writing a disconnected script.
- Add explicit condition labels such as `original`, `random_centric`, `low_signal_centric`, `mixed_50`, `random_graph`, and `partial_step23_randomized`.
- Seed all random feature and graph perturbations separately but record both the train/test seed and perturbation seed.
- For low-signal centric features, compute univariate association only on the training samples for each split.
- Make graph perturbation functions return both the perturbed graph and a metadata record with node count, edge count, corruption level, and whether degree distribution was approximately preserved.
