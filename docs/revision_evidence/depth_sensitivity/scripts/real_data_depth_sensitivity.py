#!/usr/bin/env python
"""Real-data PathNet graph-depth sensitivity experiment.

This runner reuses the real-data PathNet benchmark loaders and training loop.
It keeps preprocessing, labels, splits, hyperparameters, and epoch counts fixed,
and varies only the maximum graph depth from the centric features.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[3]
MODEL_BENCHMARK_DIR = ROOT / "rebuttal_pipeline" / "experiments" / "model_benchmark"
if str(MODEL_BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_BENCHMARK_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from real_data_pathnet import (  # noqa: E402
    PathNetConfig,
    build_pathnet_model,
    configs_for_mode,
    count_parameters,
    count_sparse_connections,
    load_bundles,
    notebook_config,
    train_one,
)


EVIDENCE_DIR = ROOT / "rebuttal_pipeline" / "evidence" / "depth_sensitivity"
TABLE_DIR = EVIDENCE_DIR / "tables"
FIGURE_DIR = EVIDENCE_DIR / "figures"
SCRIPT_COPY_DIR = EVIDENCE_DIR / "scripts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "notebook", "full"], default="full")
    parser.add_argument("--datasets", nargs="+", choices=["covid", "brca"], default=["covid", "brca"])
    parser.add_argument("--depths", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--architecture-depths", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=EVIDENCE_DIR)
    parser.add_argument("--log-every", type=int, default=0)
    parser.add_argument("--save-history", action="store_true")
    parser.add_argument("--max-parameters", type=int, default=300_000_000)
    return parser.parse_args()


def depth_config(dataset: str, mode: str, depth: int) -> list[PathNetConfig]:
    configs = configs_for_mode(dataset, mode)
    return [PathNetConfig(**{**asdict(cfg), "maximum_step": depth}) for cfg in configs]


def architecture_scan(bundles, depths: list[int]) -> pd.DataFrame:
    rows = []
    for bundle in bundles:
        base = notebook_config(bundle.dataset)
        for depth in depths:
            cfg = PathNetConfig(**{**asdict(base), "maximum_step": depth, "max_epochs": 1})
            row = {
                "dataset": bundle.dataset,
                "maximum_step": depth,
                "n_pathway_nodes": int(bundle.knowledge_graph.vcount()),
                "n_pathway_edges": int(bundle.knowledge_graph.ecount()),
                "run_status": "ok",
                "run_error": "",
            }
            try:
                started = time.perf_counter()
                model, partition_mtx_dict, _residual_connection_dict, connection_list = build_pathnet_model(bundle, cfg)
                row.update(
                    {
                        "build_seconds": float(time.perf_counter() - started),
                        "n_connection_layers": int(len(connection_list)),
                        "connection_layer_sizes": "|".join(str(len(layer)) for layer in connection_list),
                        "n_sparse_connections": int(count_sparse_connections(partition_mtx_dict)),
                        "n_parameters": int(count_parameters(model)),
                        "partition_shapes": json.dumps(
                            {key: list(value.shape) for key, value in partition_mtx_dict.items() if key != "p0"},
                            sort_keys=True,
                        ),
                    }
                )
                del model
            except Exception as exc:
                row.update(
                    {
                        "run_status": "failed",
                        "run_error": repr(exc),
                        "build_seconds": float("nan"),
                        "n_connection_layers": 0,
                        "connection_layer_sizes": "",
                        "n_sparse_connections": float("nan"),
                        "n_parameters": float("nan"),
                        "partition_shapes": "{}",
                    }
                )
            rows.append(row)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "test_accuracy",
        "test_balanced_accuracy",
        "test_roc_auc",
        "val_accuracy",
        "val_balanced_accuracy",
        "val_roc_auc",
        "train_seconds",
        "n_parameters",
        "n_sparse_connections",
        "n_connection_layers",
    ]
    rows = []
    grouped = results.groupby(["dataset", "maximum_step"], dropna=False)
    for (dataset, depth), frame in grouped:
        row = {"dataset": dataset, "maximum_step": int(depth), "n_runs": int(len(frame))}
        for col in metric_cols:
            row[f"{col}_mean"] = float(frame[col].mean())
            row[f"{col}_std"] = float(frame[col].std(ddof=1)) if len(frame) > 1 else 0.0
            row[f"{col}_min"] = float(frame[col].min())
            row[f"{col}_max"] = float(frame[col].max())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["dataset", "maximum_step"])


def markdown_table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    if df.empty:
        return "_No rows available._"
    headers = list(df.columns)
    rows = []
    for _, row in df.iterrows():
        values = []
        for col in headers:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                values.append(format(float(value), floatfmt))
            else:
                values.append(str(value))
        rows.append(values)
    widths = [max(len(header), *(len(row[idx]) for row in rows)) for idx, header in enumerate(headers)]
    header_line = "| " + " | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)) + " |"
    sep_line = "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |"
    body_lines = [
        "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep_line, *body_lines])


def plot_depth_metric(summary: pd.DataFrame, metric: str, output_path: Path, ylabel: str) -> None:
    datasets = list(summary["dataset"].drop_duplicates())
    fig, axes = plt.subplots(1, len(datasets), figsize=(max(6, 4.0 * len(datasets)), 3.8), sharey=False)
    if len(datasets) == 1:
        axes = [axes]
    for ax, dataset in zip(axes, datasets):
        sub = summary[summary["dataset"] == dataset].sort_values("maximum_step")
        x = sub["maximum_step"].to_numpy()
        y = sub[f"{metric}_mean"].to_numpy()
        err = sub[f"{metric}_std"].to_numpy()
        ax.errorbar(x, y, yerr=err, marker="o", linewidth=1.8, capsize=3, color="#4C78A8")
        ax.set_title(dataset.upper())
        ax.set_xlabel("Maximum graph depth")
        ax.set_xticks(x)
        ax.grid(alpha=0.25)
        if "accuracy" in metric or "auc" in metric:
            ax.set_ylim(0.0, 1.0)
    axes[0].set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_architecture_growth(architecture: pd.DataFrame, output_path: Path) -> None:
    datasets = list(architecture["dataset"].drop_duplicates())
    fig, axes = plt.subplots(1, len(datasets), figsize=(max(6, 4.0 * len(datasets)), 3.8), sharey=False)
    if len(datasets) == 1:
        axes = [axes]
    for ax, dataset in zip(axes, datasets):
        sub = architecture[architecture["dataset"] == dataset].sort_values("maximum_step")
        ax.plot(sub["maximum_step"], sub["n_parameters"], marker="o", color="#F58518", label="Parameters")
        ax.set_title(dataset.upper())
        ax.set_xlabel("Maximum graph depth")
        ax.set_xticks(sub["maximum_step"])
        ax.set_ylabel("Trainable parameters")
        ax.grid(alpha=0.25)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def write_summary_markdown(
    output_dir: Path,
    args: argparse.Namespace,
    summary: pd.DataFrame,
    architecture: pd.DataFrame,
) -> None:
    summary_cols = [
        "dataset",
        "maximum_step",
        "n_runs",
        "test_roc_auc_mean",
        "test_roc_auc_std",
        "test_balanced_accuracy_mean",
        "test_balanced_accuracy_std",
        "test_accuracy_mean",
        "test_accuracy_std",
        "train_seconds_mean",
        "n_parameters_mean",
        "n_sparse_connections_mean",
    ]
    arch_cols = [
        "dataset",
        "maximum_step",
        "connection_layer_sizes",
        "n_sparse_connections",
        "n_parameters",
        "run_status",
    ]
    lines = [
        "# Real-Data Depth Sensitivity Evidence",
        "",
        "## Scope",
        "",
        "This package evaluates how the maximum graph depth from centric features affects real-data PathNet performance and model size. Preprocessing, labels, split protocol, hyperparameters, and epoch counts match the existing real-data PathNet benchmark; only `maximum_step` is varied.",
        "",
        "## Run Configuration",
        "",
        f"- Mode: `{args.mode}`",
        f"- Datasets: `{', '.join(args.datasets)}`",
        f"- Trained depths: `{', '.join(map(str, args.depths))}`",
        f"- Architecture-scan depths: `{', '.join(map(str, args.architecture_depths))}`",
        "- Split: stratified 60/20/20 train/validation/test, matching the model benchmark.",
        "- Seeds: inherited from the selected benchmark mode.",
        "",
        "## Trained Depth Summary",
        "",
        markdown_table(summary[summary_cols], floatfmt=".3f"),
        "",
        "## Architecture Growth",
        "",
        markdown_table(architecture[arch_cols], floatfmt=".3f"),
        "",
        "## Interpretation Notes",
        "",
        "- Depth controls how broad a graph neighborhood around the centric features is included.",
        "- Depth 2 is the original notebook setting for both real datasets and is therefore the primary reference.",
        "- Larger depths can include more indirect pathway context, but they also increase the number of included nodes, sparse connections, parameters, and training time.",
        "- BRCA grows sharply after depth 3 in the architecture scan, so depths 4 and 5 are recorded as model-size evidence but are not trained by default in the full sweep.",
        "- These real-data results address predictive and computational sensitivity. Ground-truth subnetwork recovery still requires simulation.",
        "",
        "## Reviewer-Comment Use",
        "",
        "- `R1.3`: direct real-data evidence for graph-depth sensitivity and practical depth selection.",
        "- `R1.5`: additional computational-performance evidence as graph neighborhoods grow.",
        "- `R2.3`: partial scalability evidence because depth changes pathway-neighborhood size and parameter count.",
        "",
        "## Files",
        "",
        "- Per-run results: `tables/real_data_depth_results.csv`",
        "- Summary table: `tables/real_data_depth_summary.csv`",
        "- Architecture scan: `tables/real_data_depth_architecture.csv`",
        "- ROC-AUC figure: `figures/real_data_depth_roc_auc.pdf`",
        "- Balanced-accuracy figure: `figures/real_data_depth_balanced_accuracy.pdf`",
        "- Architecture-growth figure: `figures/real_data_depth_model_size.pdf`",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_metadata(output_dir: Path, bundles, args: argparse.Namespace, device: torch.device) -> None:
    metadata = {
        "purpose": "Real-data graph-depth sensitivity for PathNet.",
        "mode": args.mode,
        "command": " ".join(sys.argv),
        "device": str(device),
        "depths": args.depths,
        "architecture_depths": args.architecture_depths,
        "max_parameters": args.max_parameters,
        "datasets": {bundle.dataset: bundle.metadata for bundle in bundles},
        "notes": [
            "Preprocessing and labels follow the existing real_data_pathnet.py benchmark loaders.",
            "Depth 2 is the original notebook setting for both real datasets.",
            "Training keeps notebook hyperparameters and epoch counts fixed while varying only maximum_step.",
            "Architecture scan records larger depths even when they are not trained in the full sweep.",
        ],
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    script_copy_dir = output_dir / "scripts"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    script_copy_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), script_copy_dir / Path(__file__).name)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    bundles = load_bundles(args.datasets)
    write_metadata(output_dir, bundles, args, device)

    architecture = architecture_scan(bundles, args.architecture_depths)
    architecture.to_csv(table_dir / "real_data_depth_architecture.csv", index=False)

    rows = []
    epoch_rows = []
    total = sum(len(depth_config(bundle.dataset, args.mode, depth)) for bundle in bundles for depth in args.depths)
    current = 0
    for bundle in bundles:
        for depth in args.depths:
            configs = depth_config(bundle.dataset, args.mode, depth)
            for cfg in configs:
                current += 1
                arch_row = architecture[
                    (architecture["dataset"] == bundle.dataset) & (architecture["maximum_step"] == depth)
                ].iloc[0]
                if arch_row["run_status"] != "ok":
                    row = {
                        "dataset": bundle.dataset,
                        "model": "PathNet",
                        "maximum_step": depth,
                        "seed": cfg.seed,
                        "run_status": "failed_architecture",
                        "run_error": arch_row["run_error"],
                    }
                    rows.append(row)
                    continue
                if int(arch_row["n_parameters"]) > args.max_parameters:
                    row = {
                        "dataset": bundle.dataset,
                        "model": "PathNet",
                        "maximum_step": depth,
                        "seed": cfg.seed,
                        "run_status": "skipped_parameter_limit",
                        "run_error": f"n_parameters={int(arch_row['n_parameters'])} exceeds max_parameters={args.max_parameters}",
                    }
                    rows.append(row)
                    continue

                print(
                    f"[run {current}/{total}] {bundle.dataset} depth={depth} "
                    f"seed={cfg.seed}, epochs={cfg.max_epochs}, params={int(arch_row['n_parameters'])}",
                    flush=True,
                )
                started = time.perf_counter()
                try:
                    row, history = train_one(bundle, cfg, device, args.log_every)
                    row["run_status"] = "ok"
                    row["run_error"] = ""
                    row["wall_seconds_total"] = float(time.perf_counter() - started)
                    rows.append(row)
                    if args.save_history:
                        for hist in history:
                            hist["maximum_step"] = depth
                        epoch_rows.extend(history)
                    print(
                        f"  test_auc={row['test_roc_auc']:.3f} "
                        f"test_bacc={row['test_balanced_accuracy']:.3f} "
                        f"test_acc={row['test_accuracy']:.3f} "
                        f"sec={row['train_seconds']:.1f}",
                        flush=True,
                    )
                except Exception as exc:
                    rows.append(
                        {
                            "dataset": bundle.dataset,
                            "model": "PathNet",
                            "maximum_step": depth,
                            "seed": cfg.seed,
                            "run_status": "failed",
                            "run_error": repr(exc),
                            "wall_seconds_total": float(time.perf_counter() - started),
                        }
                    )
                    print(f"  failed: {exc!r}", flush=True)
                pd.DataFrame(rows).to_csv(table_dir / "real_data_depth_results_partial.csv", index=False)

    results = pd.DataFrame(rows)
    results_path = table_dir / "real_data_depth_results.csv"
    results.to_csv(results_path, index=False)
    successful = results[results["run_status"] == "ok"].copy()
    if successful.empty:
        raise RuntimeError("No depth-sensitivity runs completed successfully.")
    summary = summarize(successful)
    summary.to_csv(table_dir / "real_data_depth_summary.csv", index=False)
    plot_depth_metric(summary, "test_roc_auc", figure_dir / "real_data_depth_roc_auc.pdf", "Test ROC-AUC")
    plot_depth_metric(
        summary,
        "test_balanced_accuracy",
        figure_dir / "real_data_depth_balanced_accuracy.pdf",
        "Test balanced accuracy",
    )
    plot_architecture_growth(architecture, figure_dir / "real_data_depth_model_size.pdf")
    write_summary_markdown(output_dir, args, summary, architecture)

    if args.save_history and epoch_rows:
        pd.DataFrame(epoch_rows).to_csv(table_dir / "real_data_depth_epoch_history.csv", index=False)
    partial_path = table_dir / "real_data_depth_results_partial.csv"
    if partial_path.exists():
        partial_path.unlink()

    print(f"[done] results: {results_path}", flush=True)
    print(f"[done] summary: {table_dir / 'real_data_depth_summary.csv'}", flush=True)
    print(f"[done] figures: {figure_dir}", flush=True)


if __name__ == "__main__":
    main()
