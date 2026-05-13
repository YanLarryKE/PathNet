#!/usr/bin/env python
"""Real-data MLP/KAN benchmarks for the rebuttal.

This script follows the preprocessing choices in example_meta.ipynb and
example_gene.ipynb, then compares black-box MLP and efficient-KAN baselines on
full and selected feature sets.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[3]
EVIDENCE_DIR = ROOT / "rebuttal_pipeline" / "evidence" / "model_benchmark"
SCRIPT_COPY_DIR = EVIDENCE_DIR / "scripts"
TABLE_DIR = EVIDENCE_DIR / "tables"
FIGURE_DIR = EVIDENCE_DIR / "figures"

COVID_TARGET_KEGGIDS = [
    "C00328",
    "C00186",
    "C00584",
    "C02165",
    "C00696",
    "C01089",
    "C00042",
    "C06124",
    "C00249",
    "C00780",
]

BRCA_TARGET_GENES = [
    "OPRK1",
    "CAMKV",
    "MAGEB4",
    "CNGB1",
    "TLX3",
    "SCEL",
    "NR2E1",
    "CCKBR",
    "FETUB",
    "KCNJ3",
    "LIN28B",
    "PDX1",
    "PRKAG3",
    "GHRHR",
    "TRIM10",
    "FOXG1",
    "GSTA5",
    "MAGEA4",
    "AHSG",
    "TUBA3C",
    "PYY",
    "ALPP",
    "SERPINB4",
    "C1orf94",
    "GPR139",
    "TEKT1",
    "GHRH",
    "PSKH2",
]


@dataclass
class DatasetBundle:
    dataset: str
    feature_set: str
    x: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    metadata: dict


@dataclass
class RunConfig:
    model: str
    architecture: str
    hidden_layers: list[int]
    dropout: float
    learning_rate: float
    weight_decay: float
    batch_size: int
    max_epochs: int
    patience: int
    kan_grid_size: int | None = None
    kan_spline_order: int | None = None
    selected_features: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--datasets", nargs="+", default=["covid", "brca"], choices=["covid", "brca"])
    parser.add_argument("--feature-sets", nargs="+", default=["selected", "full"], choices=["selected", "full"])
    parser.add_argument("--models", nargs="+", default=["mlp", "kan"], choices=["mlp", "kan"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-dir", type=Path, default=EVIDENCE_DIR)
    parser.add_argument("--no-figures", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_numeric_matrix(df: pd.DataFrame) -> pd.DataFrame:
    df = df.apply(pd.to_numeric, errors="coerce")
    return df.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def filter_low_expr(expr_df: pd.DataFrame, min_sample_pct: float) -> pd.DataFrame:
    n_samples = expr_df.shape[0]
    min_samples = int(n_samples * min_sample_pct)
    expr_count = (expr_df > 0).sum(axis=0)
    high_expr = expr_count[expr_count >= min_samples].index
    return expr_df[high_expr]


def safe_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, score))
    except ValueError:
        return float("nan")


def load_covid_bundles() -> list[DatasetBundle]:
    sys.path.insert(0, str(ROOT))
    from PathNet.utils import data_preprocessing

    pos = pd.read_csv(ROOT / "data" / "data_processed" / "pos_processed.csv", index_col=0)
    neg = pd.read_csv(ROOT / "data" / "data_processed" / "neg_processed.csv", index_col=0)
    data_annos, matchings, _sub_graph, metabolites, _dic = data_preprocessing(
        pos=pos,
        neg=neg,
        idx_feature=4,
        match_tol_ppm=9,
        zero_threshold=0.75,
        scale=True,
    )

    info = pd.read_csv(ROOT / "data" / "labels" / "y.csv")
    info = info.iloc[
        np.where(np.logical_and(info["cov"] == "Yes", info["icu"].isin(["Yes", "No"])))[0],
        :,
    ]
    x_full = np.asarray(data_annos.iloc[:, 4:].T.values, dtype=np.float32)
    y = np.zeros(x_full.shape[0], dtype=np.int64)
    y[np.where(info["icu"] == "Yes")[0]] = 1

    feature_names = [str(v) for v in data_annos.index]
    metabolite_names = [str(v) for v in metabolites]
    matching_arr = np.asarray(matchings)
    selected_metabolite_idx = [i for i, name in enumerate(metabolite_names) if name in COVID_TARGET_KEGGIDS]
    selected_feature_mask = matching_arr[:, selected_metabolite_idx].sum(axis=1) > 0 if selected_metabolite_idx else np.zeros(matching_arr.shape[0], dtype=bool)
    selected_idx = np.where(selected_feature_mask)[0]
    if len(selected_idx) == 0:
        raise RuntimeError("No COVID measured features mapped to the selected KEGG IDs.")

    common_meta = {
        "label_definition": "cov == Yes and icu Yes vs No; Yes encoded as 1",
        "notebook_reference": "example_meta.ipynb",
        "target_keggids": COVID_TARGET_KEGGIDS,
        "n_samples": int(x_full.shape[0]),
        "class_counts": {"0": int((y == 0).sum()), "1": int((y == 1).sum())},
        "n_matched_metabolites": int(len(metabolite_names)),
    }
    return [
        DatasetBundle(
            dataset="covid",
            feature_set="full",
            x=x_full,
            y=y,
            feature_names=feature_names,
            metadata={
                **common_meta,
                "feature_definition": "all LC-MS measured features retained by PathNet preprocessing",
            },
        ),
        DatasetBundle(
            dataset="covid",
            feature_set="selected",
            x=x_full[:, selected_idx],
            y=y,
            feature_names=[feature_names[i] for i in selected_idx],
            metadata={
                **common_meta,
                "feature_definition": "measured LC-MS features mapped to the 10 target KEGG IDs in example_meta.ipynb",
                "n_selected_target_keggids_present": int(len(selected_metabolite_idx)),
                "selected_target_keggids_present": [metabolite_names[i] for i in selected_metabolite_idx],
            },
        ),
    ]


def parse_target_genes(value: object) -> list[str]:
    if not isinstance(value, str):
        return []
    value = value.strip()
    if value in {"", "nan", "[]"}:
        return []
    return [g.strip() for g in value.strip("[]").replace("'", "").replace('"', "").split(",") if g.strip()]


def load_brca_bundles() -> list[DatasetBundle]:
    gene_expr = pd.read_csv(ROOT / "data" / "data_raw" / "HiSeqV2", sep="\t", index_col=0).transpose()
    gene_net = pd.read_csv(ROOT / "data" / "data_raw" / "HomoSapiens_lcb_hq.txt", sep="\t")
    mirna_expr = pd.read_csv(ROOT / "data" / "data_raw" / "TCGA-mirna_expr.csv", index_col=0)

    gene_expr = clean_numeric_matrix(gene_expr)
    mirna_expr = clean_numeric_matrix(mirna_expr)

    mirna_expr.index = mirna_expr.index.str.slice(stop=-1)
    mirna_expr = mirna_expr[~mirna_expr.index.duplicated(keep="first")]
    mirna_expr = mirna_expr[mirna_expr.index.str.endswith("-01")]

    common_samples = np.intersect1d(gene_expr.index.values, mirna_expr.index.values)
    gene_expr = gene_expr.loc[common_samples]
    mirna_expr = mirna_expr.loc[gene_expr.index]

    gene_expr = filter_low_expr(gene_expr, min_sample_pct=0.2)
    mirna_expr = filter_low_expr(mirna_expr, min_sample_pct=0.05)

    interactions: set[frozenset[str]] = set()
    for _, row in gene_net.iterrows():
        interactions.add(frozenset({row["Gene_A"], row["Gene_B"]}))

    genes_in_expression = set(gene_expr.columns)
    valid_pairs = []
    for pair in interactions:
        if len(pair) == 1:
            continue
        gene_a, gene_b = tuple(pair)
        if gene_a in genes_in_expression and gene_b in genes_in_expression:
            valid_pairs.append((gene_a, gene_b))
    valid_pairs_df = pd.DataFrame(valid_pairs, columns=["Gene_A", "Gene_B"])
    genes_in_network = set(valid_pairs_df["Gene_A"]).union(set(valid_pairs_df["Gene_B"]))
    gene_expr = gene_expr.loc[:, gene_expr.columns.isin(genes_in_network)]

    clinical = pd.read_csv(
        ROOT / "data" / "labels" / "TCGA.BRCA.sampleMap_BRCA_clinicalMatrix",
        index_col=0,
        sep="\t",
    )
    clinical = clinical.loc[gene_expr.index, :]
    valid_index = clinical.index[clinical["ER_Status_nature2012"].isin(["Positive", "Negative"])]
    gene_expr = gene_expr.loc[valid_index, :]
    mirna_expr = mirna_expr.loc[valid_index, :]
    er_status = clinical.loc[valid_index, "ER_Status_nature2012"]
    y = np.zeros(len(valid_index), dtype=np.int64)
    y[er_status == "Positive"] = 1

    mirna_mapping = pd.read_csv(ROOT / "data" / "data_processed" / "miRNA_gene_mapping.csv")
    mirna_mapping["miRNA"] = mirna_mapping["miRNA"].str.lower()
    mirna_mapping = mirna_mapping.set_index("miRNA")
    mirna_mapping = mirna_mapping.reindex(mirna_expr.columns)

    selected_genes = [g for g in BRCA_TARGET_GENES if g in gene_expr.columns]
    selected_mirnas = []
    target_set = set(selected_genes)
    for mirna, row in mirna_mapping.iterrows():
        genes = parse_target_genes(row.get("target_genes"))
        if target_set.intersection(genes):
            selected_mirnas.append(mirna)
    selected_mirnas = [m for m in selected_mirnas if m in mirna_expr.columns]

    if not selected_genes:
        raise RuntimeError("No BRCA selected genes were present after notebook-style filtering.")

    full_df = pd.concat(
        [
            gene_expr.add_prefix("gene:"),
            mirna_expr.add_prefix("mirna:"),
        ],
        axis=1,
    )
    selected_parts = [gene_expr.loc[:, selected_genes].add_prefix("gene:")]
    if selected_mirnas:
        selected_parts.append(mirna_expr.loc[:, selected_mirnas].add_prefix("mirna:"))
    selected_df = pd.concat(selected_parts, axis=1)

    common_meta = {
        "label_definition": "ER_Status_nature2012 Positive vs Negative; Indeterminate excluded; Positive encoded as 1",
        "notebook_reference": "example_gene.ipynb",
        "target_genes": BRCA_TARGET_GENES,
        "n_samples": int(full_df.shape[0]),
        "class_counts": {"0": int((y == 0).sum()), "1": int((y == 1).sum())},
        "n_network_genes_after_filtering": int(gene_expr.shape[1]),
        "n_mirnas_after_filtering": int(mirna_expr.shape[1]),
    }

    return [
        DatasetBundle(
            dataset="brca",
            feature_set="full",
            x=np.asarray(full_df.values, dtype=np.float32),
            y=y,
            feature_names=[str(c) for c in full_df.columns],
            metadata={
                **common_meta,
                "feature_definition": "all notebook-retained network genes concatenated with filtered miRNAs",
            },
        ),
        DatasetBundle(
            dataset="brca",
            feature_set="selected",
            x=np.asarray(selected_df.values, dtype=np.float32),
            y=y,
            feature_names=[str(c) for c in selected_df.columns],
            metadata={
                **common_meta,
                "feature_definition": "28 target genes from example_gene.ipynb plus filtered miRNAs mapped to those target genes",
                "selected_target_genes_present": selected_genes,
                "n_selected_mirnas_mapped_to_target_genes": int(len(selected_mirnas)),
            },
        ),
    ]


def load_bundles(names: Iterable[str]) -> list[DatasetBundle]:
    bundles: list[DatasetBundle] = []
    if "covid" in names:
        bundles.extend(load_covid_bundles())
    if "brca" in names:
        bundles.extend(load_brca_bundles())
    return bundles


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: list[int], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for hidden in hidden_layers:
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = hidden
        layers.append(nn.Linear(prev, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_kan(input_dim: int, hidden_layers: list[int], grid_size: int, spline_order: int) -> nn.Module:
    from efficient_kan import KAN

    return KAN([input_dim, *hidden_layers, 2], grid_size=grid_size, spline_order=spline_order)


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def make_configs(mode: str, models: list[str]) -> list[RunConfig]:
    if mode == "smoke":
        configs = [
            RunConfig("mlp", "mlp_smoke_32", [32], 0.1, 1e-3, 1e-4, 32, 4, 2),
            RunConfig("kan", "kan_smoke_8_g3", [8], 0.0, 1e-3, 1e-4, 32, 4, 2, kan_grid_size=3, kan_spline_order=2),
        ]
    else:
        configs = [
            RunConfig("mlp", "mlp_64", [64], 0.1, 1e-3, 1e-4, 32, 120, 15),
            RunConfig("mlp", "mlp_128_64", [128, 64], 0.2, 1e-3, 1e-4, 32, 120, 15),
            RunConfig("mlp", "mlp_256_128_64", [256, 128, 64], 0.3, 7e-4, 1e-4, 32, 140, 18),
            RunConfig("kan", "kan_16_g3", [16], 0.0, 1e-3, 1e-4, 32, 90, 12, kan_grid_size=3, kan_spline_order=2),
            RunConfig("kan", "kan_32_g3", [32], 0.0, 7e-4, 1e-4, 32, 90, 12, kan_grid_size=3, kan_spline_order=2),
            RunConfig("kan", "kan_32_16_g3", [32, 16], 0.0, 7e-4, 1e-4, 32, 100, 14, kan_grid_size=3, kan_spline_order=2),
        ]
    return [cfg for cfg in configs if cfg.model in models]


def split_and_scale(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, ...]:
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
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)
    return (
        x_train.astype(np.float32),
        x_val.astype(np.float32),
        x_test.astype(np.float32),
        y_train.astype(np.int64),
        y_val.astype(np.int64),
        y_test.astype(np.int64),
    )


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    y_score: list[float] = []
    losses: list[float] = []
    loss_func = nn.CrossEntropyLoss()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = loss_func(logits, yb)
            probs = torch.softmax(logits, dim=1)[:, 1]
            pred = torch.argmax(logits, dim=1)
            losses.append(float(loss.item()))
            y_true.extend(yb.cpu().numpy().tolist())
            y_pred.extend(pred.cpu().numpy().tolist())
            y_score.extend(probs.cpu().numpy().tolist())
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    y_score_arr = np.asarray(y_score)
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred_arr)),
        "roc_auc": safe_auc(y_true_arr, y_score_arr),
    }


def train_one(bundle: DatasetBundle, cfg: RunConfig, seed: int, device: torch.device) -> dict:
    set_seed(seed)
    x_train, x_val, x_test, y_train, y_val, y_test = split_and_scale(bundle.x, bundle.y, seed)
    train_loader = make_loader(x_train, y_train, cfg.batch_size, shuffle=True)
    val_loader = make_loader(x_val, y_val, cfg.batch_size, shuffle=False)
    test_loader = make_loader(x_test, y_test, cfg.batch_size, shuffle=False)

    if cfg.model == "mlp":
        model = MLP(bundle.x.shape[1], cfg.hidden_layers, cfg.dropout)
    elif cfg.model == "kan":
        model = build_kan(
            bundle.x.shape[1],
            cfg.hidden_layers,
            grid_size=cfg.kan_grid_size or 3,
            spline_order=cfg.kan_spline_order or 2,
        )
    else:
        raise ValueError(cfg.model)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    loss_func = nn.CrossEntropyLoss()
    best_state = None
    best_val_loss = math.inf
    best_epoch = 0
    stale_epochs = 0
    started = time.perf_counter()

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_func(logits, yb)
            loss.backward()
            optimizer.step()

        val_metrics = evaluate(model, val_loader, device)
        val_loss = val_metrics["loss"]
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_epoch = epoch
            stale_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale_epochs += 1
        if stale_epochs >= cfg.patience:
            break

    train_seconds = time.perf_counter() - started
    if best_state is not None:
        model.load_state_dict(best_state)

    train_metrics = evaluate(model, train_loader, device)
    val_metrics = evaluate(model, val_loader, device)
    test_metrics = evaluate(model, test_loader, device)

    result = {
        "dataset": bundle.dataset,
        "feature_set": bundle.feature_set,
        "model": cfg.model,
        "architecture": cfg.architecture,
        "seed": seed,
        "n_samples": int(bundle.x.shape[0]),
        "n_features": int(bundle.x.shape[1]),
        "class_0": int((bundle.y == 0).sum()),
        "class_1": int((bundle.y == 1).sum()),
        "hidden_layers": "-".join(map(str, cfg.hidden_layers)),
        "dropout": cfg.dropout,
        "learning_rate": cfg.learning_rate,
        "weight_decay": cfg.weight_decay,
        "batch_size": cfg.batch_size,
        "max_epochs": cfg.max_epochs,
        "best_epoch": int(best_epoch),
        "train_seconds": float(train_seconds),
        "n_parameters": count_parameters(model),
        "kan_grid_size": cfg.kan_grid_size if cfg.model == "kan" else np.nan,
        "kan_spline_order": cfg.kan_spline_order if cfg.model == "kan" else np.nan,
    }
    for prefix, metrics in [("train", train_metrics), ("val", val_metrics), ("test", test_metrics)]:
        for metric, value in metrics.items():
            result[f"{prefix}_{metric}"] = value
    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "test_accuracy",
        "test_balanced_accuracy",
        "test_roc_auc",
        "val_accuracy",
        "val_balanced_accuracy",
        "val_roc_auc",
        "train_seconds",
        "n_parameters",
        "n_features",
    ]
    grouped = results.groupby(["dataset", "feature_set", "model", "architecture"], dropna=False)
    rows = []
    for keys, frame in grouped:
        row = dict(zip(["dataset", "feature_set", "model", "architecture"], keys))
        row["n_runs"] = int(len(frame))
        for col in metric_cols:
            row[f"{col}_mean"] = float(frame[col].mean())
            row[f"{col}_std"] = float(frame[col].std(ddof=1)) if len(frame) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["dataset", "feature_set", "model", "architecture"])


def write_metadata(
    output_dir: Path,
    bundles: list[DatasetBundle],
    configs: list[RunConfig],
    args: argparse.Namespace,
    seeds: list[int],
    device: torch.device,
) -> None:
    metadata = {
        "purpose": "Real-data black-box MLP and efficient-KAN benchmark for the rebuttal.",
        "mode": args.mode,
        "command": " ".join(sys.argv),
        "device": str(device),
        "seeds": seeds,
        "datasets": {
            f"{bundle.dataset}_{bundle.feature_set}": {
                "n_samples": int(bundle.x.shape[0]),
                "n_features": int(bundle.x.shape[1]),
                "class_counts": bundle.metadata.get("class_counts"),
                **bundle.metadata,
            }
            for bundle in bundles
        },
        "configs": [asdict(cfg) for cfg in configs],
        "kan_backend": "efficient-kan installed from https://github.com/Blealtan/efficient-kan.git",
        "notes": [
            "BRCA Indeterminate ER labels are excluded for a clean binary benchmark.",
            "COVID labels and target KEGG IDs follow example_meta.ipynb.",
            "BRCA preprocessing and target genes follow example_gene.ipynb.",
            "Selected black-box features are direct measured/mapped features associated with the paper's target features.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def plot_summary(summary: pd.DataFrame, figure_dir: Path) -> None:
    import matplotlib.pyplot as plt

    figure_dir.mkdir(parents=True, exist_ok=True)
    for metric, ylabel, filename in [
        ("test_roc_auc_mean", "Test ROC-AUC", "real_data_mlp_kan_auc.pdf"),
        ("test_balanced_accuracy_mean", "Test balanced accuracy", "real_data_mlp_kan_balanced_accuracy.pdf"),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for ax, dataset in zip(axes, ["covid", "brca"]):
            sub = summary[summary["dataset"] == dataset].copy()
            if sub.empty:
                ax.set_visible(False)
                continue
            sub["label"] = sub["feature_set"] + "\n" + sub["model"] + "\n" + sub["architecture"]
            x = np.arange(len(sub))
            ax.bar(x, sub[metric], yerr=sub[metric.replace("_mean", "_std")], color=["#4C78A8" if m == "mlp" else "#F58518" for m in sub["model"]])
            ax.set_title(dataset.upper())
            ax.set_xticks(x)
            ax.set_xticklabels(sub["label"], rotation=60, ha="right", fontsize=7)
            ax.set_ylim(0.0, 1.0)
            ax.grid(axis="y", alpha=0.25)
        axes[0].set_ylabel(ylabel)
        fig.tight_layout()
        fig.savefig(figure_dir / filename)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    table_dir = output_dir / "tables"
    figure_dir = output_dir / "figures"
    script_copy_dir = output_dir / "scripts"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    script_copy_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    seeds = [2781] if args.mode == "smoke" else [2781, 14526, 2027, 3407, 91011]
    configs = make_configs(args.mode, args.models)
    bundles = [b for b in load_bundles(args.datasets) if b.feature_set in args.feature_sets]

    write_metadata(output_dir, bundles, configs, args, seeds, device)
    shutil.copy2(Path(__file__), script_copy_dir / Path(__file__).name)

    rows = []
    total = len(bundles) * len(configs) * len(seeds)
    current = 0
    for bundle in bundles:
        print(
            f"[dataset] {bundle.dataset}/{bundle.feature_set}: "
            f"n={bundle.x.shape[0]}, p={bundle.x.shape[1]}, "
            f"class_counts={bundle.metadata.get('class_counts')}",
            flush=True,
        )
        for cfg in configs:
            for seed in seeds:
                current += 1
                print(f"[run {current}/{total}] {bundle.dataset}/{bundle.feature_set} {cfg.architecture} seed={seed}", flush=True)
                try:
                    row = train_one(bundle, cfg, seed, device)
                    rows.append(row)
                    print(
                        f"  test_auc={row['test_roc_auc']:.3f} "
                        f"test_bacc={row['test_balanced_accuracy']:.3f} "
                        f"sec={row['train_seconds']:.1f}",
                        flush=True,
                    )
                except RuntimeError as exc:
                    rows.append(
                        {
                            "dataset": bundle.dataset,
                            "feature_set": bundle.feature_set,
                            "model": cfg.model,
                            "architecture": cfg.architecture,
                            "seed": seed,
                            "n_samples": int(bundle.x.shape[0]),
                            "n_features": int(bundle.x.shape[1]),
                            "error": repr(exc),
                        }
                    )
                    print(f"  ERROR: {exc}", flush=True)
                    if "out of memory" in str(exc).lower() and torch.cuda.is_available():
                        torch.cuda.empty_cache()

    results = pd.DataFrame(rows)
    suffix = "smoke" if args.mode == "smoke" else "full"
    result_path = table_dir / f"real_data_mlp_kan_results_{suffix}.csv"
    summary_path = table_dir / f"real_data_mlp_kan_summary_{suffix}.csv"
    results.to_csv(result_path, index=False)
    ok_results = results[results.get("error", pd.Series(index=results.index, dtype=object)).isna()] if "error" in results.columns else results
    if not ok_results.empty:
        summary = summarize_results(ok_results)
        summary.to_csv(summary_path, index=False)
        if args.mode == "full" and not args.no_figures:
            plot_summary(summary, figure_dir)
    print(f"[done] results: {result_path}", flush=True)
    print(f"[done] summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
