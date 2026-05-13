#!/usr/bin/env python
"""Controlled simulation stress tests for PathNet prior and feature quality.

This script is intentionally separate from ``syn_data.ipynb``.  The notebook
generator lets source nodes carry class signal directly, which makes random
graphs surprisingly competitive whenever they accidentally route those sources
to the selected targets.  Here the graph is generated in modules so that the
correct prior is the only short route from informative source nodes to the
chosen centric features, while bad centric features live in disconnected decoy
modules.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import igraph as ig
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PathNet.model import pathNet  # noqa: E402
from PathNet.utils import getPartitionMatricesList  # noqa: E402


EVIDENCE_DIR = ROOT / "rebuttal_pipeline" / "evidence" / "prior_robustness"
TABLE_DIR = EVIDENCE_DIR / "tables"
SCRIPT_COPY_DIR = EVIDENCE_DIR / "scripts"


@dataclass
class SimulationConfig:
    n_modules: int
    module_size: int
    n_signal_modules: int
    n_sources_per_signal: int
    n_intermediates_per_signal: int
    n_samples: int
    source_shift: float
    source_noise: float
    label_noise: float
    decoy_shift: float
    decoy_noise: float
    maximum_step: int
    batch_size: int
    max_epochs: int
    learning_rate: float
    dropout: float
    seed: int


@dataclass
class SyntheticDataset:
    x: np.ndarray
    y: np.ndarray
    graph: ig.Graph
    true_targets: list[str]
    bad_targets: list[str]
    source_nodes: list[int]
    signal_modules: list[int]
    bad_modules: list[int]
    metadata: dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=[
            "correct_graph",
            "correct_graph_depth1",
            "correct_graph_depth3",
            "random_graph",
            "wrong_graph_decoy",
            "random_layer_matched",
            "partial_inner_random",
            "bad_feature",
            "mixed25_bad_feature",
            "mixed50_bad_feature",
            "mixed75_bad_feature",
        ],
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=EVIDENCE_DIR)
    parser.add_argument("--log-every", type=int, default=0)
    return parser.parse_args()


def configs_for_mode(mode: str) -> list[SimulationConfig]:
    if mode == "pilot":
        seeds = [2781, 14526, 2027]
        return [
            SimulationConfig(
                n_modules=24,
                module_size=28,
                n_signal_modules=5,
                n_sources_per_signal=5,
                n_intermediates_per_signal=3,
                n_samples=600,
                source_shift=1.0,
                source_noise=0.2,
                label_noise=0.45,
                decoy_shift=0.15,
                decoy_noise=1.0,
                maximum_step=2,
                batch_size=48,
                max_epochs=70,
                learning_rate=1e-3,
                dropout=0.1,
                seed=0,
            )
            for seed in seeds
            for _ in [None]
        ]
    if mode == "full":
        seeds = [2781, 14526, 2027, 3407, 91011]
        configs: list[SimulationConfig] = []
        for n_samples, module_size in [(500, 28), (900, 36)]:
            for seed in seeds:
                configs.append(
                    SimulationConfig(
                        n_modules=30,
                        module_size=module_size,
                        n_signal_modules=6,
                        n_sources_per_signal=6,
                        n_intermediates_per_signal=3,
                        n_samples=n_samples,
                        source_shift=1.0,
                        source_noise=0.2,
                        label_noise=0.45,
                        decoy_shift=0.15,
                        decoy_noise=1.0,
                        maximum_step=2,
                        batch_size=48,
                        max_epochs=90,
                        learning_rate=1e-3,
                        dropout=0.1,
                        seed=seed,
                    )
                )
        return configs
    raise ValueError(mode)


def normalize_config_seeds(configs: list[SimulationConfig], mode: str) -> list[SimulationConfig]:
    if mode == "pilot":
        seeds = [2781, 14526, 2027]
        return [SimulationConfig(**{**asdict(cfg), "seed": seed}) for cfg, seed in zip(configs, seeds)]
    return configs


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def condition_seed(condition: str) -> int:
    return sum((idx + 1) * ord(char) for idx, char in enumerate(condition))


def safe_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, score))
    except ValueError:
        return float("nan")


def module_nodes(module_id: int, module_size: int) -> np.ndarray:
    start = module_id * module_size
    return np.arange(start, start + module_size, dtype=int)


def build_modular_graph(cfg: SimulationConfig, rng: np.random.Generator) -> tuple[ig.Graph, dict]:
    n_nodes = cfg.n_modules * cfg.module_size
    edges: set[tuple[int, int]] = set()
    true_targets: list[int] = []
    bad_targets: list[int] = []
    source_nodes: list[int] = []
    signal_modules = list(range(cfg.n_signal_modules))
    bad_modules = list(range(cfg.n_signal_modules, cfg.n_signal_modules * 2))

    for module in range(cfg.n_modules):
        nodes = module_nodes(module, cfg.module_size)
        target = int(nodes[0])
        intermediates = nodes[1 : 1 + cfg.n_intermediates_per_signal]
        middle_nodes = nodes[
            1 + cfg.n_intermediates_per_signal : 1 + 2 * cfg.n_intermediates_per_signal
        ]
        sources = nodes[
            1 + 2 * cfg.n_intermediates_per_signal : 1 + 2 * cfg.n_intermediates_per_signal + cfg.n_sources_per_signal
        ]
        local_decoys = nodes[1 + 2 * cfg.n_intermediates_per_signal + cfg.n_sources_per_signal :]

        # Backbone: sources are exactly three hops away from the target. This
        # makes depth-1 insufficient under the current PathNet layer construction,
        # while depth-2 can route source -> middle -> inner -> target.
        for inter in intermediates:
            edges.add(tuple(sorted((target, int(inter)))))
        for mid_idx, mid in enumerate(middle_nodes):
            inter = int(intermediates[mid_idx % len(intermediates)])
            edges.add(tuple(sorted((int(mid), inter))))
        for source_idx, source in enumerate(sources):
            mid = int(middle_nodes[source_idx % len(middle_nodes)])
            edges.add(tuple(sorted((int(source), mid))))

        # Sparse within-module nuisance edges that do not connect modules.
        for _ in range(max(2, cfg.module_size // 5)):
            a, b = rng.choice(local_decoys, size=2, replace=False)
            edges.add(tuple(sorted((int(a), int(b)))))
        for decoy in rng.choice(local_decoys, size=min(3, len(local_decoys)), replace=False):
            inter = int(rng.choice(intermediates))
            edges.add(tuple(sorted((int(decoy), inter))))

        if module in signal_modules:
            true_targets.append(target)
            source_nodes.extend([int(s) for s in sources])
        if module in bad_modules:
            bad_targets.append(target)

    graph = ig.Graph(n=n_nodes, edges=sorted(edges), directed=False)
    graph.vs["name"] = [str(i) for i in range(n_nodes)]
    graph.simplify()
    metadata = {
        "n_nodes": n_nodes,
        "n_edges": int(graph.ecount()),
        "true_targets": [str(v) for v in true_targets],
        "bad_targets": [str(v) for v in bad_targets],
        "source_nodes": [int(v) for v in source_nodes],
        "signal_modules": signal_modules,
        "bad_modules": bad_modules,
    }
    return graph, metadata


def generate_dataset(cfg: SimulationConfig) -> SyntheticDataset:
    rng = np.random.default_rng(cfg.seed)
    graph, graph_meta = build_modular_graph(cfg, rng)
    n_nodes = cfg.n_modules * cfg.module_size
    x = rng.normal(0.0, cfg.decoy_noise, size=(cfg.n_samples, n_nodes)).astype(np.float32)

    z = rng.normal(0.0, 1.0, size=cfg.n_samples)
    class_sign = np.where(z >= 0, 1.0, -1.0)
    true_targets = [int(v) for v in graph_meta["true_targets"]]
    source_nodes = [int(v) for v in graph_meta["source_nodes"]]

    source_weights = rng.choice([-1.0, 1.0], size=len(source_nodes))
    source_weights *= rng.uniform(0.8, 1.2, size=len(source_nodes))
    x[:, source_nodes] = (
        class_sign[:, None] * cfg.source_shift * source_weights[None, :]
        + rng.normal(0.0, cfg.source_noise, size=(cfg.n_samples, len(source_nodes)))
    )

    # Give decoy modules weak nuisance shifts so that bad centric features are
    # not trivially constant, but cannot explain the labels well.
    for bad_target in graph_meta["bad_targets"]:
        module = int(bad_target) // cfg.module_size
        nodes = module_nodes(module, cfg.module_size)
        decoys = nodes[
            1 + 2 * cfg.n_intermediates_per_signal : 1 + 2 * cfg.n_intermediates_per_signal + 3
        ]
        x[:, decoys] += rng.normal(0.0, cfg.decoy_shift, size=(cfg.n_samples, len(decoys)))

    module_scores = []
    for target in true_targets:
        neighbors_3 = graph.neighborhood(target, order=3, mindist=3)
        local_sources = sorted(set(neighbors_3).intersection(source_nodes))
        source_idx = [source_nodes.index(node) for node in local_sources]
        module_scores.append((x[:, local_sources] * source_weights[source_idx][None, :]).mean(axis=1))
    latent_score = np.vstack(module_scores).mean(axis=0)
    latent_score += rng.normal(0.0, cfg.label_noise, size=cfg.n_samples)
    y = (latent_score > np.median(latent_score)).astype(np.int64)

    metadata = {
        **graph_meta,
        "label_mean": float(y.mean()),
        "source_abs_signal_separation": float(
            np.mean(np.abs(x[y == 1][:, source_nodes])) - np.mean(np.abs(x[y == 0][:, source_nodes]))
        ),
        "signed_source_signal_separation": float(
            np.mean(x[y == 1][:, source_nodes] * source_weights[None, :])
            - np.mean(x[y == 0][:, source_nodes] * source_weights[None, :])
        ),
        "source_weight_positive": int(np.sum(source_weights > 0)),
        "source_weight_negative": int(np.sum(source_weights < 0)),
        "latent_std": float(np.std(latent_score)),
        "n_samples": int(cfg.n_samples),
        "n_features": int(n_nodes),
    }
    return SyntheticDataset(
        x=x.astype(np.float32),
        y=y,
        graph=graph,
        true_targets=[str(v) for v in true_targets],
        bad_targets=[str(v) for v in graph_meta["bad_targets"]],
        source_nodes=source_nodes,
        signal_modules=graph_meta["signal_modules"],
        bad_modules=graph_meta["bad_modules"],
        metadata=metadata,
    )


def random_graph_like(graph: ig.Graph, rng: np.random.Generator) -> ig.Graph:
    n_nodes = graph.vcount()
    n_edges = graph.ecount()
    edges: set[tuple[int, int]] = set()
    while len(edges) < n_edges:
        a = int(rng.integers(0, n_nodes))
        b = int(rng.integers(0, n_nodes - 1))
        if b >= a:
            b += 1
        edges.add(tuple(sorted((a, b))))
    out = ig.Graph(n=n_nodes, edges=sorted(edges), directed=False)
    out.vs["name"] = graph.vs["name"]
    out.simplify()
    return out


def valid_partition(targets: list[str], graph: ig.Graph, maximum_step: int) -> bool:
    try:
        getPartitionMatricesList(targets, graph, maximum_step, abla_graph=False)
        return True
    except Exception:
        return False


def choose_random_graph(dataset: SyntheticDataset, cfg: SimulationConfig, rng: np.random.Generator) -> ig.Graph:
    for _ in range(500):
        graph = random_graph_like(dataset.graph, rng)
        if valid_partition(dataset.true_targets, graph, cfg.maximum_step):
            return graph
    raise RuntimeError("Could not generate a valid random graph for the true targets.")


def count_routable_sources(
    source_nodes: list[int],
    partition_mtx_dict: dict[str, np.ndarray],
    connection_list: list[list[int]],
) -> int:
    """Count true source nodes with at least one learned sparse route.

    PathNet prepends the full node set as the outer layer when the selected
    target neighborhood does not cover the graph. Therefore, a source node can
    still enter the model if its row in the first sparse mask has any non-zero
    connection to the selected neighborhood.
    """
    if len(connection_list) < 2 or "p2" not in partition_mtx_dict:
        return 0
    row_lookup = {node: idx for idx, node in enumerate(connection_list[0])}
    residual_sources = set(connection_list[1]).intersection(source_nodes)
    first_mask = partition_mtx_dict["p2"]
    count = len(residual_sources)
    for source in source_nodes:
        if source in residual_sources:
            continue
        row_idx = row_lookup.get(source)
        if row_idx is not None and np.count_nonzero(first_mask[row_idx, :]) > 0:
            count += 1
    return count


def pathnet_receptive_nodes(
    partition_mtx_dict: dict[str, np.ndarray],
    connection_list: list[list[int]],
) -> list[int]:
    if len(connection_list) < 2 or "p2" not in partition_mtx_dict:
        return sorted(connection_list[-1]) if connection_list else []

    nodes = set(connection_list[1])
    first_mask = partition_mtx_dict["p2"]
    for row_idx, node in enumerate(connection_list[0]):
        if np.count_nonzero(first_mask[row_idx, :]) > 0:
            nodes.add(node)
    return sorted(nodes)


def source_nodes_reachable(
    graph: ig.Graph,
    targets: list[str],
    source_nodes: list[int],
    maximum_step: int,
) -> bool:
    target_indices = [graph.vs.find(name).index for name in targets]
    reached: set[int] = set()
    for target in target_indices:
        reached.update(graph.neighborhood(target, order=maximum_step + 1, mindist=0))
    return not reached.isdisjoint(source_nodes)


def choose_source_avoiding_random_graph(
    dataset: SyntheticDataset,
    cfg: SimulationConfig,
    rng: np.random.Generator,
) -> ig.Graph:
    """Generate a density-matched wrong prior with no short true-source route."""
    for _ in range(3000):
        graph = random_graph_like(dataset.graph, rng)
        if not valid_partition(dataset.true_targets, graph, cfg.maximum_step):
            continue
        if source_nodes_reachable(graph, dataset.true_targets, dataset.source_nodes, cfg.maximum_step):
            continue
        return graph
    raise RuntimeError("Could not generate a source-avoiding wrong graph for the true targets.")


def split_train_val_test(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, ...]:
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=seed,
        stratify=y,
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=0.25,
        random_state=seed + 17,
        stratify=y_train_val,
    )
    return (
        x_train.astype(np.float32),
        x_val.astype(np.float32),
        x_test.astype(np.float32),
        y_train.astype(np.int64),
        y_val.astype(np.int64),
        y_test.astype(np.int64),
    )


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    loss_func = nn.CrossEntropyLoss()
    losses = []
    y_true: list[int] = []
    y_pred: list[int] = []
    y_score: list[float] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            output = model(xb)
            loss = loss_func(output, yb)
            pred = torch.argmax(output, dim=1)
            losses.append(float(loss.item()))
            y_true.extend(yb.cpu().numpy().tolist())
            y_pred.extend(pred.cpu().numpy().tolist())
            if output.shape[1] > 1:
                y_score.extend(output[:, 1].detach().cpu().numpy().tolist())
            else:
                y_score.extend(output.squeeze(-1).detach().cpu().numpy().tolist())
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    y_score_arr = np.asarray(y_score)
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred_arr)),
        "roc_auc": safe_auc(y_true_arr, y_score_arr),
    }


def count_sparse_connections(partition_mtx_dict: dict[str, np.ndarray]) -> int:
    return int(sum(np.count_nonzero(v) for k, v in partition_mtx_dict.items() if k != "p0"))


def randomize_partition(
    partition_mtx_dict: dict[str, np.ndarray],
    rng: np.random.Generator,
    keys: set[str] | None = None,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key, matrix in partition_mtx_dict.items():
        if keys is not None and key not in keys:
            out[key] = matrix.copy()
            continue
        nnz = int(np.count_nonzero(matrix))
        flat = np.zeros(matrix.size, dtype=float)
        if nnz:
            chosen = rng.choice(matrix.size, size=nnz, replace=False)
            flat[chosen] = 1.0
        out[key] = flat.reshape(matrix.shape)
    return out


def build_pathnet_for_condition(
    dataset: SyntheticDataset,
    cfg: SimulationConfig,
    condition: str,
    rng: np.random.Generator,
) -> tuple[nn.Module, dict, list[list[int]], str, list[str]]:
    targets = list(dataset.true_targets)
    graph = dataset.graph
    maximum_step = cfg.maximum_step
    partition_note = "correct_partition"

    if condition == "correct_graph_depth1":
        maximum_step = 1
        partition_note = "correct_partition_depth1"
    elif condition == "correct_graph_depth3":
        maximum_step = 3
        partition_note = "correct_partition_depth3"

    if condition == "bad_feature":
        targets = list(dataset.bad_targets[: len(dataset.true_targets)])
    elif condition in {"mixed25_bad_feature", "mixed50_bad_feature", "mixed75_bad_feature"}:
        bad_fraction = {
            "mixed25_bad_feature": 0.25,
            "mixed50_bad_feature": 0.50,
            "mixed75_bad_feature": 0.75,
        }[condition]
        n_bad = max(1, int(round(len(dataset.true_targets) * bad_fraction)))
        n_bad = min(n_bad, len(dataset.true_targets))
        targets = list(dataset.true_targets[:-n_bad] + dataset.bad_targets[:n_bad])
    elif condition == "random_graph":
        graph = choose_random_graph(dataset, cfg, rng)
        partition_note = "unrestricted_density_matched_random_graph"
    elif condition == "wrong_graph_decoy":
        graph = choose_source_avoiding_random_graph(dataset, cfg, rng)
        partition_note = "density_matched_random_graph_without_true_source_route"

    partition_mtx_dict, residual_connection_dict, connection_list = getPartitionMatricesList(
        targets,
        graph,
        maximum_step,
        abla_graph=False,
    )

    if condition == "random_layer_matched":
        partition_mtx_dict = randomize_partition(partition_mtx_dict, rng)
        partition_note = "all_sparse_masks_randomized_with_original_layer_sizes"
    elif condition == "partial_inner_random":
        inner_keys = {f"p{i}" for i in range(3, len(connection_list) + 1)}
        partition_mtx_dict = randomize_partition(partition_mtx_dict, rng, keys=inner_keys)
        partition_note = "inner_sparse_masks_randomized_with_original_layer_sizes"

    model = pathNet(partition_mtx_dict, residual_connection_dict, [], cfg.dropout)
    return model, partition_mtx_dict, connection_list, partition_note, targets


class MLP(nn.Module):
    def __init__(self, n_features: int, hidden_layers: list[int], dropout: float):
        super().__init__()
        dims = [n_features] + hidden_layers + [2]
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-2], dims[1:-1]):
            layers.extend([nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout)])
        layers.extend([nn.Linear(dims[-2], dims[-1]), nn.Softmax(dim=-1)])
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_kan(n_features: int) -> nn.Module:
    from efficient_kan import KAN

    return KAN([n_features, 16, 2], grid_size=3, spline_order=2)


def selected_nodes_for_black_box(dataset: SyntheticDataset, cfg: SimulationConfig, condition: str) -> tuple[list[int], str]:
    if condition.endswith("_full"):
        return list(range(dataset.x.shape[1])), "all_observed_features"
    if condition.endswith("_targets"):
        return [int(v) for v in dataset.true_targets], "selected_centric_targets_only"
    if condition.endswith("_pathnet_receptive"):
        partition_mtx_dict, _, connection_list = getPartitionMatricesList(
            dataset.true_targets,
            dataset.graph,
            cfg.maximum_step,
            abla_graph=False,
        )
        return pathnet_receptive_nodes(partition_mtx_dict, connection_list), "selected_pathnet_receptive_nodes"
    raise ValueError(f"Unknown black-box feature scope for condition {condition!r}")


def train_model(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    cfg: SimulationConfig,
    device: torch.device,
    log_every: int,
) -> tuple[dict[str, float], float]:
    x_train, x_val, x_test, y_train, y_val, y_test = split_train_val_test(x, y, cfg.seed)
    train_loader = make_loader(x_train, y_train, cfg.batch_size, shuffle=True)
    val_loader = make_loader(x_val, y_val, cfg.batch_size, shuffle=False)
    test_loader = make_loader(x_test, y_test, cfg.batch_size, shuffle=False)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=0.0)
    loss_func = nn.CrossEntropyLoss()
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
                f"    epoch={epoch:03d}/{cfg.max_epochs} val_auc={val_metrics['roc_auc']:.3f} "
                f"val_bacc={val_metrics['balanced_accuracy']:.3f}",
                flush=True,
            )
    seconds = time.perf_counter() - started
    metrics: dict[str, float] = {}
    for prefix, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        for metric, value in evaluate(model, loader, device).items():
            metrics[f"{prefix}_{metric}"] = value
    return metrics, seconds


def run_condition(
    dataset: SyntheticDataset,
    cfg: SimulationConfig,
    condition: str,
    device: torch.device,
    log_every: int,
) -> dict:
    set_seed(cfg.seed + condition_seed(condition))
    rng = np.random.default_rng(cfg.seed + 1009 + sum(ord(c) for c in condition))
    started = time.perf_counter()

    x_model = dataset.x
    feature_scope = "graph_constrained_full_input"
    n_model_input_features = int(dataset.x.shape[1])

    if condition.startswith("mlp_"):
        input_nodes, feature_scope = selected_nodes_for_black_box(dataset, cfg, condition)
        x_model = dataset.x[:, input_nodes]
        model = MLP(len(input_nodes), [128, 64], cfg.dropout)
        n_sparse_connections = 0
        n_parameters = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
        connection_sizes = ""
        target_features = "|".join(str(node) for node in input_nodes)
        partition_note = "fully_connected_black_box"
        n_routable_sources = len(set(input_nodes).intersection(dataset.source_nodes))
        n_model_input_features = len(input_nodes)
    elif condition.startswith("kan_"):
        input_nodes, feature_scope = selected_nodes_for_black_box(dataset, cfg, condition)
        x_model = dataset.x[:, input_nodes]
        model = build_kan(len(input_nodes))
        n_sparse_connections = 0
        n_parameters = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
        connection_sizes = ""
        target_features = "|".join(str(node) for node in input_nodes)
        partition_note = "efficient_kan_black_box"
        n_routable_sources = len(set(input_nodes).intersection(dataset.source_nodes))
        n_model_input_features = len(input_nodes)
    else:
        model, partition_mtx_dict, connection_list, partition_note, targets = build_pathnet_for_condition(
            dataset, cfg, condition, rng
        )
        n_sparse_connections = count_sparse_connections(partition_mtx_dict)
        n_routable_sources = count_routable_sources(dataset.source_nodes, partition_mtx_dict, connection_list)
        n_parameters = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
        connection_sizes = "|".join(str(len(layer)) for layer in connection_list)
        target_features = "|".join(targets)

    metrics, train_seconds = train_model(model, x_model, dataset.y, cfg, device, log_every)
    total_seconds = time.perf_counter() - started
    row = {
        "condition": condition,
        "seed": cfg.seed,
        "n_samples": cfg.n_samples,
        "n_features": int(dataset.x.shape[1]),
        "n_modules": cfg.n_modules,
        "module_size": cfg.module_size,
        "n_signal_modules": cfg.n_signal_modules,
        "maximum_step": cfg.maximum_step,
        "max_epochs": cfg.max_epochs,
        "batch_size": cfg.batch_size,
        "learning_rate": cfg.learning_rate,
        "dropout": cfg.dropout,
        "source_shift": cfg.source_shift,
        "source_noise": cfg.source_noise,
        "label_noise": cfg.label_noise,
        "partition_note": partition_note,
        "feature_scope": feature_scope,
        "target_features": target_features,
        "n_model_input_features": n_model_input_features,
        "connection_layer_sizes": connection_sizes,
        "n_sparse_connections": n_sparse_connections,
        "n_routable_sources": n_routable_sources,
        "n_parameters": n_parameters,
        "train_seconds": train_seconds,
        "total_seconds": total_seconds,
        "label_mean": dataset.metadata["label_mean"],
        "signed_source_signal_separation": dataset.metadata["signed_source_signal_separation"],
        "source_abs_signal_separation": dataset.metadata["source_abs_signal_separation"],
    }
    row.update(metrics)
    return row


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "test_roc_auc",
        "test_balanced_accuracy",
        "test_accuracy",
        "val_roc_auc",
        "val_balanced_accuracy",
        "train_seconds",
        "n_model_input_features",
        "n_sparse_connections",
        "n_routable_sources",
        "n_parameters",
    ]
    rows = []
    for condition, frame in results.groupby("condition"):
        row = {"condition": condition, "n_runs": int(len(frame))}
        for col in metric_cols:
            row[f"{col}_mean"] = float(frame[col].mean())
            row[f"{col}_std"] = float(frame[col].std(ddof=1)) if len(frame) > 1 else 0.0
            row[f"{col}_min"] = float(frame[col].min())
            row[f"{col}_max"] = float(frame[col].max())
        rows.append(row)
    return pd.DataFrame(rows).sort_values("condition")


def write_metadata(args: argparse.Namespace, configs: list[SimulationConfig], device: torch.device) -> None:
    metadata = {
        "purpose": "Controlled simulation of graph-prior quality and centric-feature quality for PathNet rebuttal evidence.",
        "mode": args.mode,
        "command": " ".join(sys.argv),
        "device": str(device),
        "mechanism": [
            "The graph is modular. True centric features are module targets.",
            "Informative source nodes are three graph hops away from true targets.",
            "Observed target nodes are not directly assigned the class signal; the correct graph must route signed source signal to targets.",
            "Bad centric features are matched targets from disconnected decoy modules.",
            "Wrong graph conditions preserve node count and edge count while altering or breaking the designed source-to-target routes.",
            "The default simulation conditions focus on imperfect knowledge priors and centric-feature quality; MLP/KAN baselines are handled by real-data experiments.",
        ],
        "configs": [asdict(cfg) for cfg in configs],
        "conditions": args.conditions,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"metadata_{args.mode}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table_dir = args.output_dir / "tables"
    script_copy_dir = args.output_dir / "scripts"
    table_dir.mkdir(parents=True, exist_ok=True)
    script_copy_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), script_copy_dir / Path(__file__).name)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    configs = normalize_config_seeds(configs_for_mode(args.mode), args.mode)
    write_metadata(args, configs, device)
    rows = []
    dataset_rows = []
    total = len(configs) * len(args.conditions)
    current = 0
    for cfg in configs:
        dataset = generate_dataset(cfg)
        dataset_rows.append({"seed": cfg.seed, **dataset.metadata})
        print(
            f"[dataset] seed={cfg.seed} n={cfg.n_samples} p={dataset.x.shape[1]} "
            f"label_mean={dataset.metadata['label_mean']:.3f} "
            f"signed_source_sep={dataset.metadata['signed_source_signal_separation']:.3f}",
            flush=True,
        )
        for condition in args.conditions:
            current += 1
            print(f"[run {current}/{total}] condition={condition} seed={cfg.seed}", flush=True)
            try:
                row = run_condition(dataset, cfg, condition, device, args.log_every)
                row["run_status"] = "ok"
                row["run_error"] = ""
                print(
                    f"  test_auc={row['test_roc_auc']:.3f} "
                    f"test_bacc={row['test_balanced_accuracy']:.3f} "
                    f"test_acc={row['test_accuracy']:.3f} "
                    f"sec={row['train_seconds']:.1f}",
                    flush=True,
                )
            except Exception as exc:  # keep exploratory sweeps from losing prior rows
                row = {
                    "condition": condition,
                    "seed": cfg.seed,
                    "n_samples": cfg.n_samples,
                    "n_features": cfg.n_modules * cfg.module_size,
                    "run_status": "error",
                    "run_error": repr(exc),
                }
                print(f"  ERROR: {exc!r}", flush=True)
            rows.append(row)

    results = pd.DataFrame(rows)
    ok_results = results.loc[results["run_status"] == "ok"].copy()
    summary = summarize(ok_results) if not ok_results.empty else pd.DataFrame()
    dataset_summary = pd.DataFrame(dataset_rows)

    results_path = table_dir / f"simulation_prior_robustness_results_{args.mode}.csv"
    summary_path = table_dir / f"simulation_prior_robustness_summary_{args.mode}.csv"
    data_path = table_dir / f"simulation_dataset_summary_{args.mode}.csv"
    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    dataset_summary.to_csv(data_path, index=False)
    print(f"[done] results: {results_path}", flush=True)
    print(f"[done] summary: {summary_path}", flush=True)
    print(f"[done] dataset summary: {data_path}", flush=True)


if __name__ == "__main__":
    main()
