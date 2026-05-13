#!/usr/bin/env python
"""Real-data stress tests for PathNet centric-feature and graph-prior quality.

The data loaders, train/validation/test split, hyperparameters, and epoch counts
follow the existing real-data PathNet benchmark. This script only changes the
centric-feature set or graph/connection prior so that the stress conditions are
directly comparable to the notebook-setting PathNet reruns.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from torch import nn


ROOT = Path(__file__).resolve().parents[3]
MODEL_BENCHMARK_DIR = ROOT / "rebuttal_pipeline" / "experiments" / "model_benchmark"
if str(MODEL_BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_BENCHMARK_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from real_data_pathnet import (  # noqa: E402
    PathNetBundle,
    PathNetConfig,
    configs_for_mode,
    count_parameters,
    count_sparse_connections,
    evaluate,
    load_bundles,
    make_loader,
    safe_auc,
    set_seed,
    split_train_val_test,
)


EVIDENCE_DIR = ROOT / "rebuttal_pipeline" / "evidence" / "real_data_stress"
TABLE_DIR = EVIDENCE_DIR / "tables"
FIGURE_DIR = EVIDENCE_DIR / "figures"
SCRIPT_COPY_DIR = EVIDENCE_DIR / "scripts"

DEFAULT_CONDITIONS = [
    "original",
    "random_centric",
    "low_signal_centric",
    "mixed25_low_signal_centric",
    "mixed50_low_signal_centric",
    "mixed75_low_signal_centric",
    "random_graph",
    "random_layer_matched",
    "partial_outer_layer_random",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "notebook", "full"], default="full")
    parser.add_argument("--datasets", nargs="+", choices=["covid", "brca"], default=["covid", "brca"])
    parser.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=EVIDENCE_DIR)
    parser.add_argument("--log-every", type=int, default=0)
    parser.add_argument("--selection-pool-multiplier", type=int, default=8)
    parser.add_argument("--save-history", action="store_true")
    return parser.parse_args()


def graph_node_names(graph: ig.Graph) -> list[str]:
    return [str(name) for name in graph.vs["name"]]


def validate_targets(targets: Iterable[str], graph: ig.Graph, maximum_step: int) -> bool:
    from PathNet.utils import getPartitionMatricesList

    try:
        getPartitionMatricesList(list(targets), graph, maximum_step, abla_graph=False)
        return True
    except Exception:
        return False


def partition_for_targets(
    targets: list[str],
    graph: ig.Graph,
    maximum_step: int,
) -> tuple[dict, dict, list[list[int]]]:
    from PathNet.utils import getPartitionMatricesList

    return getPartitionMatricesList(targets, graph, maximum_step, abla_graph=False)


def node_values_for_signal(bundle: PathNetBundle, x: np.ndarray) -> np.ndarray:
    """Return sample-by-graph-node values used to rank weak centric features."""

    if bundle.model_kind == "meta":
        matching = np.asarray(bundle.feature_meta, dtype=np.float32)
        col_sums = matching.sum(axis=0)
        col_sums[col_sums == 0] = 1.0
        weights = matching / col_sums
        return x @ weights
    if bundle.model_kind == "gene_mirna":
        if bundle.n_primary_features is None:
            raise RuntimeError("BRCA bundle is missing n_primary_features.")
        return x[:, : bundle.n_primary_features]
    raise ValueError(bundle.model_kind)


def univariate_effect_scores(bundle: PathNetBundle, x_train: np.ndarray, y_train: np.ndarray) -> dict[str, float]:
    values = node_values_for_signal(bundle, x_train)
    names = graph_node_names(bundle.knowledge_graph)
    scores: dict[str, float] = {}
    for idx, name in enumerate(names):
        col = values[:, idx]
        class0 = col[y_train == 0]
        class1 = col[y_train == 1]
        if len(class0) == 0 or len(class1) == 0:
            scores[name] = float("inf")
            continue
        pooled_std = float(np.std(col)) + 1e-8
        scores[name] = float(abs(np.mean(class1) - np.mean(class0)) / pooled_std)
    return scores


def choose_random_targets(
    bundle: PathNetBundle,
    cfg: PathNetConfig,
    rng: np.random.Generator,
    count: int,
    exclude: set[str],
    pool_multiplier: int,
) -> list[str]:
    candidates = [name for name in graph_node_names(bundle.knowledge_graph) if name not in exclude]
    if len(candidates) < count:
        raise RuntimeError(f"Only {len(candidates)} candidate targets are available for {count} requested targets.")

    for _ in range(500):
        sample = rng.choice(candidates, size=count, replace=False).tolist()
        if validate_targets(sample, bundle.knowledge_graph, cfg.maximum_step):
            return [str(v) for v in sample]

    # If pure random attempts fail, sample from a larger candidate pool and walk
    # through deterministic windows. This usually only matters for sparse graphs.
    pool_size = min(len(candidates), max(count, count * pool_multiplier))
    pool = rng.choice(candidates, size=pool_size, replace=False).tolist()
    for offset in range(0, len(pool) - count + 1):
        sample = pool[offset : offset + count]
        if validate_targets(sample, bundle.knowledge_graph, cfg.maximum_step):
            return [str(v) for v in sample]
    raise RuntimeError("Could not find a valid random centric-feature set after repeated attempts.")


def choose_low_signal_targets(
    bundle: PathNetBundle,
    cfg: PathNetConfig,
    x_train: np.ndarray,
    y_train: np.ndarray,
    count: int,
    exclude: set[str],
    pool_multiplier: int,
) -> list[str]:
    scores = univariate_effect_scores(bundle, x_train, y_train)
    ordered = [name for name, _score in sorted(scores.items(), key=lambda item: (item[1], item[0])) if name not in exclude]
    if len(ordered) < count:
        raise RuntimeError(f"Only {len(ordered)} low-signal candidates are available for {count} requested targets.")

    for offset in range(0, min(len(ordered) - count + 1, count * pool_multiplier)):
        sample = ordered[offset : offset + count]
        if validate_targets(sample, bundle.knowledge_graph, cfg.maximum_step):
            return [str(v) for v in sample]

    # Fall back to the lowest-scoring pool with random combinations.
    rng = np.random.default_rng(cfg.seed + 29017)
    pool = ordered[: min(len(ordered), max(count, count * pool_multiplier))]
    for _ in range(500):
        sample = rng.choice(pool, size=count, replace=False).tolist()
        if validate_targets(sample, bundle.knowledge_graph, cfg.maximum_step):
            return [str(v) for v in sample]
    raise RuntimeError("Could not find a valid low-signal centric-feature set.")


def choose_mixed_targets(
    bundle: PathNetBundle,
    cfg: PathNetConfig,
    x_train: np.ndarray,
    y_train: np.ndarray,
    rng: np.random.Generator,
    replace_fraction: float,
    pool_multiplier: int,
) -> list[str]:
    originals = list(bundle.target_features)
    n_replace = max(1, int(round(len(originals) * replace_fraction)))
    keep_count = len(originals) - n_replace
    original_set = set(originals)
    low_signal = choose_low_signal_targets(
        bundle,
        cfg,
        x_train,
        y_train,
        n_replace,
        exclude=original_set,
        pool_multiplier=pool_multiplier,
    )
    for _ in range(200):
        kept = rng.choice(originals, size=keep_count, replace=False).tolist() if keep_count else []
        sample = [str(v) for v in kept + low_signal]
        if validate_targets(sample, bundle.knowledge_graph, cfg.maximum_step):
            return sample
    raise RuntimeError("Could not find a valid mixed centric-feature set.")


def random_graph_like(graph: ig.Graph, rng: np.random.Generator) -> ig.Graph:
    n_nodes = graph.vcount()
    n_edges = graph.ecount()
    edges: set[tuple[int, int]] = set()
    while len(edges) < n_edges:
        source = int(rng.integers(0, n_nodes))
        target = int(rng.integers(0, n_nodes - 1))
        if target >= source:
            target += 1
        edge = tuple(sorted((source, target)))
        edges.add(edge)
    perturbed = ig.Graph(n=n_nodes, edges=sorted(edges), directed=False)
    perturbed.vs["name"] = graph_node_names(graph)
    perturbed.simplify()
    return perturbed


def choose_random_graph(
    bundle: PathNetBundle,
    cfg: PathNetConfig,
    rng: np.random.Generator,
) -> ig.Graph:
    for _ in range(200):
        graph = random_graph_like(bundle.knowledge_graph, rng)
        if validate_targets(bundle.target_features, graph, cfg.maximum_step):
            return graph
    raise RuntimeError("Could not generate a valid random graph for the original targets.")


def randomize_mask_like(
    matrix: np.ndarray,
    residual_rows: list[int],
    rng: np.random.Generator,
) -> np.ndarray:
    randomized = np.zeros_like(matrix)
    n_ones = int(np.count_nonzero(matrix))
    if n_ones == 0:
        return randomized
    allowed_rows = np.ones(matrix.shape[0], dtype=bool)
    if residual_rows:
        allowed_rows[np.asarray(residual_rows, dtype=int)] = False
    allowed_positions = np.argwhere(allowed_rows[:, None] & np.ones(matrix.shape, dtype=bool))
    if len(allowed_positions) == 0:
        return randomized
    n_sample = min(n_ones, len(allowed_positions))
    picked = rng.choice(len(allowed_positions), size=n_sample, replace=False)
    coords = allowed_positions[picked]
    randomized[coords[:, 0], coords[:, 1]] = 1
    return randomized


def apply_connection_randomization(
    partition_mtx_dict: dict,
    residual_connection_dict: dict,
    mode: str,
    rng: np.random.Generator,
) -> dict:
    randomized = {key: value.copy() for key, value in partition_mtx_dict.items()}
    layer_keys = sorted(
        [key for key in partition_mtx_dict if key.startswith("p") and key != "p0"],
        key=lambda key: int(key[1:]),
    )
    if mode == "random_layer_matched":
        keys_to_randomize = layer_keys
    elif mode == "partial_outer_layer_random":
        keys_to_randomize = layer_keys[:1]
    else:
        return randomized

    for key in keys_to_randomize:
        randomized[key] = randomize_mask_like(
            np.asarray(partition_mtx_dict[key]),
            residual_connection_dict.get(key, []),
            rng,
        )
    return randomized


def prepare_condition(
    bundle: PathNetBundle,
    cfg: PathNetConfig,
    condition: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    pool_multiplier: int,
) -> tuple[list[str], ig.Graph, str, dict]:
    rng = np.random.default_rng(cfg.seed + stable_condition_offset(condition))
    original_set = set(bundle.target_features)
    metadata: dict = {}

    if condition == "original":
        targets = list(bundle.target_features)
        graph = bundle.knowledge_graph
        partition_mode = "original"
    elif condition == "random_centric":
        targets = choose_random_targets(
            bundle,
            cfg,
            rng,
            len(bundle.target_features),
            exclude=original_set,
            pool_multiplier=pool_multiplier,
        )
        graph = bundle.knowledge_graph
        partition_mode = "original"
    elif condition == "low_signal_centric":
        targets = choose_low_signal_targets(
            bundle,
            cfg,
            x_train,
            y_train,
            len(bundle.target_features),
            exclude=original_set,
            pool_multiplier=pool_multiplier,
        )
        graph = bundle.knowledge_graph
        partition_mode = "original"
    elif condition in {"mixed25_low_signal_centric", "mixed50_low_signal_centric", "mixed75_low_signal_centric"}:
        replace_fraction = {
            "mixed25_low_signal_centric": 0.25,
            "mixed50_low_signal_centric": 0.50,
            "mixed75_low_signal_centric": 0.75,
        }[condition]
        targets = choose_mixed_targets(
            bundle,
            cfg,
            x_train,
            y_train,
            rng,
            replace_fraction=replace_fraction,
            pool_multiplier=pool_multiplier,
        )
        graph = bundle.knowledge_graph
        partition_mode = "original"
    elif condition == "random_graph":
        targets = list(bundle.target_features)
        graph = choose_random_graph(bundle, cfg, rng)
        partition_mode = "original"
    elif condition in {"random_layer_matched", "partial_outer_layer_random"}:
        targets = list(bundle.target_features)
        graph = bundle.knowledge_graph
        partition_mode = condition
    else:
        raise ValueError(f"Unknown condition: {condition}")

    metadata["n_original_targets"] = len(bundle.target_features)
    metadata["n_targets"] = len(targets)
    metadata["n_replaced_targets"] = len(set(bundle.target_features) - set(targets))
    metadata["target_overlap_with_original"] = len(set(bundle.target_features).intersection(targets))
    metadata["graph_edges"] = int(graph.ecount())
    metadata["graph_nodes"] = int(graph.vcount())
    return targets, graph, partition_mode, metadata


def stable_condition_offset(condition: str) -> int:
    return sum((idx + 1) * ord(char) for idx, char in enumerate(condition)) + 1009


def build_pathnet_model_for_condition(
    bundle: PathNetBundle,
    cfg: PathNetConfig,
    targets: list[str],
    graph: ig.Graph,
    partition_mode: str,
    rng: np.random.Generator,
) -> tuple[nn.Module, dict, dict, list[list[int]]]:
    from PathNet.model import GeneExpressionModel, meta_Net

    partition_mtx_dict, residual_connection_dict, connection_list = partition_for_targets(
        targets,
        graph,
        cfg.maximum_step,
    )
    partition_mtx_dict = apply_connection_randomization(
        partition_mtx_dict,
        residual_connection_dict,
        partition_mode,
        rng,
    )
    partition_mtx_dict["p0"] = bundle.feature_meta

    if bundle.model_kind == "meta":
        model = meta_Net(partition_mtx_dict, residual_connection_dict, cfg.hidden_layers, cfg.dropout)
    elif bundle.model_kind == "gene_mirna":
        if bundle.n_primary_features is None:
            raise RuntimeError("n_primary_features is required for the gene/miRNA PathNet model.")
        model = GeneExpressionModel(
            partition_mtx_dict,
            residual_connection_dict,
            cfg.hidden_layers,
            cfg.dropout,
            bundle.n_primary_features,
        )
    else:
        raise ValueError(bundle.model_kind)
    return model, partition_mtx_dict, residual_connection_dict, connection_list


def train_one_condition(
    bundle: PathNetBundle,
    cfg: PathNetConfig,
    condition: str,
    device: torch.device,
    log_every: int,
    pool_multiplier: int,
) -> tuple[dict, list[dict]]:
    set_seed(cfg.seed)
    warnings.filterwarnings("ignore", category=UserWarning)

    x_train, x_val, x_test, y_train, y_val, y_test = split_train_val_test(bundle.x, bundle.y, cfg.seed)
    targets, graph, partition_mode, condition_meta = prepare_condition(
        bundle,
        cfg,
        condition,
        x_train,
        y_train,
        pool_multiplier,
    )

    perturb_rng = np.random.default_rng(cfg.seed + stable_condition_offset(condition) + 7919)
    model, partition_mtx_dict, _residual_connection_dict, connection_list = build_pathnet_model_for_condition(
        bundle,
        cfg,
        targets,
        graph,
        partition_mode,
        perturb_rng,
    )
    model = model.to(device)

    train_loader = make_loader(x_train, y_train, cfg.batch_size, shuffle=True)
    val_loader = make_loader(x_val, y_val, cfg.batch_size, shuffle=False)
    test_loader = make_loader(x_test, y_test, cfg.batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    loss_func = nn.CrossEntropyLoss()
    n_parameters = count_parameters(model)
    n_sparse_connections = count_sparse_connections(partition_mtx_dict)

    epoch_rows = []
    started = time.perf_counter()
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            output = model(xb)
            loss = loss_func(output, yb)
            loss.backward()
            optimizer.step()

        if log_every > 0 and (epoch == 1 or epoch == cfg.max_epochs or epoch % log_every == 0):
            val_metrics = evaluate(model, val_loader, device)
            print(
                f"  epoch={epoch:03d}/{cfg.max_epochs} "
                f"val_acc={val_metrics['accuracy']:.3f} "
                f"val_auc={val_metrics['roc_auc']:.3f}",
                flush=True,
            )
        if log_every > 0:
            train_metrics = evaluate(model, train_loader, device)
            val_metrics = evaluate(model, val_loader, device)
            epoch_rows.append(
                {
                    "dataset": bundle.dataset,
                    "condition": condition,
                    "seed": cfg.seed,
                    "epoch": epoch,
                    "train_accuracy": train_metrics["accuracy"],
                    "train_balanced_accuracy": train_metrics["balanced_accuracy"],
                    "train_roc_auc": train_metrics["roc_auc"],
                    "val_accuracy": val_metrics["accuracy"],
                    "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                    "val_roc_auc": val_metrics["roc_auc"],
                }
            )

    train_seconds = time.perf_counter() - started
    train_metrics = evaluate(model, train_loader, device)
    val_metrics = evaluate(model, val_loader, device)
    test_metrics = evaluate(model, test_loader, device)

    target_scores = univariate_effect_scores(bundle, x_train, y_train)
    selected_scores = [target_scores.get(target, float("nan")) for target in targets]
    layer_sizes = [len(layer) for layer in connection_list]
    partition_shapes = {
        key: list(value.shape)
        for key, value in partition_mtx_dict.items()
        if key != "p0"
    }
    partition_nonzeros = {
        key: int(np.count_nonzero(value))
        for key, value in partition_mtx_dict.items()
        if key != "p0"
    }

    result = {
        "dataset": bundle.dataset,
        "model": "PathNet",
        "condition": condition,
        "seed": cfg.seed,
        "n_samples": int(bundle.x.shape[0]),
        "n_input_features": int(bundle.x.shape[1]),
        "class_0": int((bundle.y == 0).sum()),
        "class_1": int((bundle.y == 1).sum()),
        "maximum_step": cfg.maximum_step,
        "hidden_layers": "-".join(map(str, cfg.hidden_layers)) if cfg.hidden_layers else "none",
        "dropout": cfg.dropout,
        "learning_rate": cfg.learning_rate,
        "weight_decay": cfg.weight_decay,
        "batch_size": cfg.batch_size,
        "max_epochs": cfg.max_epochs,
        "train_seconds": float(train_seconds),
        "n_parameters": int(n_parameters),
        "n_sparse_connections": int(n_sparse_connections),
        "n_connection_layers": int(len(connection_list)),
        "connection_layer_sizes": "|".join(str(size) for size in layer_sizes),
        "partition_shapes": json.dumps(partition_shapes, sort_keys=True),
        "partition_nonzeros": json.dumps(partition_nonzeros, sort_keys=True),
        "target_features": "|".join(targets),
        "target_signal_score_mean": float(np.nanmean(selected_scores)),
        "target_signal_score_median": float(np.nanmedian(selected_scores)),
        "target_signal_score_max": float(np.nanmax(selected_scores)),
        "notebook_reference": cfg.notebook_reference,
        **condition_meta,
    }
    for prefix, metrics in [("train", train_metrics), ("val", val_metrics), ("test", test_metrics)]:
        for metric, value in metrics.items():
            result[f"{prefix}_{metric}"] = value

    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result, epoch_rows


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
        "target_signal_score_mean",
    ]
    rows = []
    grouped = results.groupby(["dataset", "condition"], dropna=False)
    for (dataset, condition), frame in grouped:
        row = {"dataset": dataset, "condition": condition, "n_runs": int(len(frame))}
        for col in metric_cols:
            row[f"{col}_mean"] = float(frame[col].mean())
            row[f"{col}_std"] = float(frame[col].std(ddof=1)) if len(frame) > 1 else 0.0
            row[f"{col}_min"] = float(frame[col].min())
            row[f"{col}_max"] = float(frame[col].max())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["dataset", "condition"])


def condition_family(condition: str) -> str:
    if condition in {
        "random_centric",
        "low_signal_centric",
        "mixed25_low_signal_centric",
        "mixed50_low_signal_centric",
        "mixed75_low_signal_centric",
    }:
        return "feature_selection"
    if condition == "original":
        return "reference"
    return "graph_prior"


def write_condition_tables(summary: pd.DataFrame, table_dir: Path) -> None:
    summary.assign(condition_family=summary["condition"].map(condition_family)).to_csv(
        table_dir / "real_data_stress_summary.csv",
        index=False,
    )
    feature = summary[summary["condition"].map(condition_family).isin(["reference", "feature_selection"])]
    graph = summary[summary["condition"].map(condition_family).isin(["reference", "graph_prior"])]
    feature.to_csv(table_dir / "real_data_feature_selection_stress.csv", index=False)
    graph.to_csv(table_dir / "real_data_graph_prior_stress.csv", index=False)


def plot_metric(summary: pd.DataFrame, metric: str, output_path: Path, title: str) -> None:
    condition_order = [
        "original",
        "random_centric",
        "low_signal_centric",
        "mixed25_low_signal_centric",
        "mixed50_low_signal_centric",
        "mixed75_low_signal_centric",
        "random_graph",
        "random_layer_matched",
        "partial_outer_layer_random",
    ]
    labels = {
        "original": "Original",
        "random_centric": "Random\ncentric",
        "low_signal_centric": "Low-signal\ncentric",
        "mixed25_low_signal_centric": "Mixed 25%\ncentric",
        "mixed50_low_signal_centric": "Mixed 50%\ncentric",
        "mixed75_low_signal_centric": "Mixed 75%\ncentric",
        "random_graph": "Random\ngraph",
        "random_layer_matched": "Random\nlayers",
        "partial_outer_layer_random": "Outer layer\nrandom",
    }
    datasets = list(summary["dataset"].drop_duplicates())
    fig, axes = plt.subplots(1, len(datasets), figsize=(max(6, 4.2 * len(datasets)), 4.0), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    colors = {
        "reference": "#4C78A8",
        "feature_selection": "#F58518",
        "graph_prior": "#54A24B",
    }
    for ax, dataset in zip(axes, datasets):
        sub = summary[summary["dataset"] == dataset].set_index("condition")
        present = [condition for condition in condition_order if condition in sub.index]
        means = [sub.loc[condition, f"{metric}_mean"] for condition in present]
        stds = [sub.loc[condition, f"{metric}_std"] for condition in present]
        bar_colors = [colors[condition_family(condition)] for condition in present]
        ax.bar(range(len(present)), means, yerr=stds, capsize=3, color=bar_colors, edgecolor="black", linewidth=0.4)
        ax.set_title(dataset.upper())
        ax.set_xticks(range(len(present)))
        ax.set_xticklabels([labels.get(condition, condition) for condition in present], rotation=35, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel(metric.replace("_", " ").title())
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


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

    widths = [
        max(len(header), *(len(row[idx]) for row in rows))
        for idx, header in enumerate(headers)
    ]
    header_line = "| " + " | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)) + " |"
    sep_line = "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |"
    body_lines = [
        "| " + " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep_line, *body_lines])


def write_summary_markdown(summary: pd.DataFrame, output_dir: Path, args: argparse.Namespace) -> None:
    lines = [
        "# Real-Data Feature And Graph Stress-Test Evidence",
        "",
        "## Scope",
        "",
        "This package stress-tests PathNet on the real COVID-19 and BRCA datasets by degrading either the centric-feature choices or the graph/connection prior while keeping the original notebook preprocessing, split protocol, hyperparameters, and epoch counts fixed.",
        "",
        "These are real-data robustness/stability experiments, not ground-truth recovery experiments. Feature and edge recovery should be evaluated later in simulation.",
        "",
        "## Run Configuration",
        "",
        f"- Mode: `{args.mode}`",
        f"- Datasets: `{', '.join(args.datasets)}`",
        f"- Conditions: `{', '.join(args.conditions)}`",
        "- Split: stratified 60/20/20 train/validation/test, matching the existing real-data PathNet benchmark.",
        "- Seeds: inherited from the selected benchmark mode.",
        "",
        "## Summary",
        "",
    ]

    display_cols = [
        "dataset",
        "condition",
        "n_runs",
        "test_roc_auc_mean",
        "test_roc_auc_std",
        "test_balanced_accuracy_mean",
        "test_balanced_accuracy_std",
        "test_accuracy_mean",
        "test_accuracy_std",
        "target_signal_score_mean_mean",
    ]
    available_cols = [col for col in display_cols if col in summary.columns]
    lines.append(markdown_table(summary[available_cols], floatfmt=".3f"))
    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- `random_centric`, `low_signal_centric`, and the mixed centric-feature conditions test whether poor or mixed centric-feature choices reduce performance or stability.",
            "- `random_graph` tests a fully misleading graph prior with the original target features.",
            "- `random_layer_matched` preserves the original layer sizes but randomizes sparse inter-layer connections.",
            "- `partial_outer_layer_random` keeps the original target-centered node set and inner connections but randomizes the outermost graph-derived connection layer.",
            "- Increased standard deviation or wider min-max range should be interpreted as instability under the corresponding stress condition.",
            "",
            "## Files",
            "",
            "- Per-run results: `tables/real_data_stress_results.csv`",
            "- Summary table: `tables/real_data_stress_summary.csv`",
            "- Feature-selection stress table: `tables/real_data_feature_selection_stress.csv`",
            "- Graph-prior stress table: `tables/real_data_graph_prior_stress.csv`",
            "- ROC-AUC figure: `figures/real_data_stress_roc_auc.pdf`",
            "- Balanced-accuracy figure: `figures/real_data_stress_balanced_accuracy.pdf`",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_metadata(output_dir: Path, bundles: list[PathNetBundle], args: argparse.Namespace, device: torch.device) -> None:
    metadata = {
        "purpose": "Real-data PathNet stress tests for centric-feature quality and graph-prior quality.",
        "mode": args.mode,
        "command": " ".join(sys.argv),
        "device": str(device),
        "conditions": args.conditions,
        "datasets": {bundle.dataset: bundle.metadata for bundle in bundles},
        "notes": [
            "Preprocessing and labels follow the existing real_data_pathnet.py benchmark loaders.",
            "Stress conditions keep training hyperparameters fixed and alter only targets or graph/connection priors.",
            "Low-signal centric features are ranked using training-split univariate effect scores to avoid test leakage.",
            "These are real-data stability experiments and do not provide ground-truth subnetwork recovery metrics.",
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

    rows = []
    epoch_rows = []
    total = sum(len(configs_for_mode(bundle.dataset, args.mode)) * len(args.conditions) for bundle in bundles)
    current = 0
    for bundle in bundles:
        print(
            f"[dataset] {bundle.dataset}: n={bundle.x.shape[0]}, p={bundle.x.shape[1]}, "
            f"class_counts={bundle.metadata['class_counts']}",
            flush=True,
        )
        for cfg in configs_for_mode(bundle.dataset, args.mode):
            for condition in args.conditions:
                current += 1
                print(
                    f"[run {current}/{total}] {bundle.dataset} {condition} "
                    f"seed={cfg.seed}, epochs={cfg.max_epochs}, batch={cfg.batch_size}",
                    flush=True,
                )
                started = time.perf_counter()
                try:
                    row, history = train_one_condition(
                        bundle,
                        cfg,
                        condition,
                        device,
                        args.log_every,
                        args.selection_pool_multiplier,
                    )
                    row["run_status"] = "ok"
                    row["run_error"] = ""
                    rows.append(row)
                    if args.save_history:
                        epoch_rows.extend(history)
                    print(
                        f"  test_auc={row['test_roc_auc']:.3f} "
                        f"test_bacc={row['test_balanced_accuracy']:.3f} "
                        f"test_acc={row['test_accuracy']:.3f} "
                        f"sec={row['train_seconds']:.1f} "
                        f"params={row['n_parameters']}",
                        flush=True,
                    )
                except Exception as exc:  # keep long sweeps from losing earlier results
                    elapsed = time.perf_counter() - started
                    error_row = {
                        "dataset": bundle.dataset,
                        "model": "PathNet",
                        "condition": condition,
                        "seed": cfg.seed,
                        "run_status": "failed",
                        "run_error": repr(exc),
                        "train_seconds": elapsed,
                    }
                    rows.append(error_row)
                    print(f"  failed after {elapsed:.1f}s: {exc!r}", flush=True)

                pd.DataFrame(rows).to_csv(table_dir / "real_data_stress_results_partial.csv", index=False)

    results = pd.DataFrame(rows)
    results_path = table_dir / "real_data_stress_results.csv"
    results.to_csv(results_path, index=False)

    successful = results[results["run_status"] == "ok"].copy()
    if successful.empty:
        raise RuntimeError("No stress-test runs completed successfully.")
    summary = summarize(successful)
    write_condition_tables(summary, table_dir)
    plot_metric(summary, "test_roc_auc", figure_dir / "real_data_stress_roc_auc.pdf", "Real-data stress test: ROC-AUC")
    plot_metric(
        summary,
        "test_balanced_accuracy",
        figure_dir / "real_data_stress_balanced_accuracy.pdf",
        "Real-data stress test: balanced accuracy",
    )
    write_summary_markdown(summary, output_dir, args)

    if args.save_history and epoch_rows:
        pd.DataFrame(epoch_rows).to_csv(table_dir / "real_data_stress_epoch_history.csv", index=False)

    partial_path = table_dir / "real_data_stress_results_partial.csv"
    if partial_path.exists():
        partial_path.unlink()

    print(f"[done] results: {results_path}", flush=True)
    print(f"[done] summary: {table_dir / 'real_data_stress_summary.csv'}", flush=True)
    print(f"[done] figures: {figure_dir}", flush=True)


if __name__ == "__main__":
    main()
