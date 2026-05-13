# Revision Evidence

This directory contains generated and derived materials from the manuscript
revision. It is intended to make the revision analyses easier to inspect without
bundling large raw omics matrices in the code repository.

The analyses use the public workflows and data formats shown in the example
notebooks. Raw public datasets should be obtained from their original sources:
TCGA/GDC for BRCA and Metabolomics Workbench study ST001849 for COVID-19.

## Contents

- `model_benchmark/`: MLP, KAN, and PathNet benchmark summaries for the COVID-19
  metabolomics and TCGA BRCA examples.
- `depth_sensitivity/`: real-data sensitivity analyses for the maximum graph
  depth parameter.
- `real_data_stress/`: real-data stress tests for centric-feature choices and
  graph-prior perturbations.
- `prior_robustness/`: controlled simulation tables, scripts, and summaries for
  evaluating correct, random, wrong, and partially corrupted graph priors.

Each subdirectory includes summary text, CSV tables, and the scripts used to
generate the reported derived results. These materials are derived outputs for
review and reproducibility; they are not a replacement for the raw public data
repositories.

The scripts are archived from the manuscript revision workflow and retain some
internal directory names used during that workflow. They are included to document
the analysis logic behind the tables; running them from this public repository
may require adapting input-data paths to local copies of the public datasets.
