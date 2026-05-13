#!/usr/bin/env python
"""Rerun the two notebook PathNet examples with benchmark-style evaluation.

The data preprocessing, target features, hyperparameters, and epoch counts
follow example_meta.ipynb and example_gene.ipynb. The evaluation wrapper uses
the same external train/validation/test split style as the MLP/KAN benchmark:
60% train, 20% validation, and 20% held-out test.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import igraph as ig
import numpy as np
import pandas as pd
import torch
from scipy.sparse import lil_matrix
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[3]
EVIDENCE_DIR = ROOT / "rebuttal_pipeline" / "evidence" / "model_benchmark"
SCRIPT_COPY_DIR = EVIDENCE_DIR / "scripts"
TABLE_DIR = EVIDENCE_DIR / "tables"

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
class PathNetConfig:
    dataset: str
    seed: int
    maximum_step: int
    hidden_layers: list[int]
    dropout: float
    learning_rate: float
    weight_decay: float
    batch_size: int
    max_epochs: int
    notebook_reference: str


@dataclass
class PathNetBundle:
    dataset: str
    x: np.ndarray
    y: np.ndarray
    feature_meta: object
    knowledge_graph: ig.Graph
    target_features: list[str]
    model_kind: str
    n_primary_features: int | None
    metadata: dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "notebook", "full"], default="notebook")
    parser.add_argument("--datasets", nargs="+", choices=["covid", "brca"], default=["covid", "brca"])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=EVIDENCE_DIR)
    parser.add_argument("--log-every", type=int, default=25)
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


def parse_target_genes(value: object) -> list[str]:
    if not isinstance(value, str):
        return []
    value = value.strip()
    if value in {"", "nan", "[]"}:
        return []
    return [g.strip() for g in value.strip("[]").replace("'", "").replace('"', "").split(",") if g.strip()]


def safe_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, score))
    except ValueError:
        return float("nan")


def load_covid_pathnet_bundle() -> PathNetBundle:
    sys.path.insert(0, str(ROOT))
    from PathNet.utils import data_preprocessing

    pos = pd.read_csv(ROOT / "data" / "data_processed" / "pos_processed.csv", index_col=0)
    neg = pd.read_csv(ROOT / "data" / "data_processed" / "neg_processed.csv", index_col=0)
    data_annos, matchings, sub_graph, metabolites, _dic = data_preprocessing(
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
    x = np.asarray(data_annos.iloc[:, 4:].T.values, dtype=np.float32)
    y = np.zeros(x.shape[0], dtype=np.int64)
    y[np.where(info["icu"] == "Yes")[0]] = 1

    metabolites = [str(v) for v in metabolites]
    present_targets = [k for k in COVID_TARGET_KEGGIDS if k in set(metabolites)]
    if len(present_targets) != len(COVID_TARGET_KEGGIDS):
        missing = sorted(set(COVID_TARGET_KEGGIDS) - set(present_targets))
        raise RuntimeError(f"COVID target KEGG IDs missing from graph-matched metabolites: {missing}")

    return PathNetBundle(
        dataset="covid",
        x=x,
        y=y,
        feature_meta=matchings,
        knowledge_graph=sub_graph,
        target_features=COVID_TARGET_KEGGIDS,
        model_kind="meta",
        n_primary_features=None,
        metadata={
            "label_definition": "cov == Yes and ICU Yes vs No; ICU Yes encoded as 1",
            "notebook_reference": "example_meta.ipynb",
            "n_samples": int(x.shape[0]),
            "n_input_features": int(x.shape[1]),
            "n_pathway_nodes": int(len(metabolites)),
            "class_counts": {"0": int((y == 0).sum()), "1": int((y == 1).sum())},
            "target_keggids": COVID_TARGET_KEGGIDS,
            "preprocessing": "PathNet data_preprocessing with match_tol_ppm=9, zero_threshold=0.75, scale=True.",
        },
    )


def load_brca_pathnet_bundle() -> PathNetBundle:
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

    genes_in_network = set()
    for gene_a, gene_b in valid_pairs:
        genes_in_network.add(gene_a)
        genes_in_network.add(gene_b)
    gene_expr = gene_expr.loc[:, gene_expr.columns.isin(genes_in_network)]

    mirna_mapping = pd.read_csv(ROOT / "data" / "data_processed" / "miRNA_gene_mapping.csv")
    mirna_mapping["miRNA"] = mirna_mapping["miRNA"].str.lower()
    mirna_mapping = mirna_mapping.set_index("miRNA").loc[mirna_expr.columns]

    gene_to_idx = {gene: idx for idx, gene in enumerate(gene_expr.columns)}
    mirna_to_idx = {mirna: idx for idx, mirna in enumerate(mirna_expr.columns)}
    reg_matrix = lil_matrix((len(gene_to_idx), len(mirna_to_idx)), dtype=np.float32)
    for mirna, row in mirna_mapping.iterrows():
        mirna_idx = mirna_to_idx[mirna]
        for gene in parse_target_genes(row["target_genes"]):
            if gene in gene_to_idx:
                reg_matrix[gene_to_idx[gene], mirna_idx] = 1

    graph_edges = []
    for gene_a, gene_b in valid_pairs:
        if gene_a in gene_to_idx and gene_b in gene_to_idx:
            graph_edges.append((gene_to_idx[gene_a], gene_to_idx[gene_b]))
    sub_graph = ig.Graph(n=len(gene_to_idx), edges=graph_edges, directed=False)
    sub_graph.simplify()
    sub_graph.vs["name"] = gene_expr.columns.values

    clinical = pd.read_csv(
        ROOT / "data" / "labels" / "TCGA.BRCA.sampleMap_BRCA_clinicalMatrix",
        index_col=0,
        sep="\t",
    )
    clinical = clinical.loc[gene_expr.index, :]
    valid_index = clinical.index[clinical["ER_Status_nature2012"].notna()]
    gene_expr = gene_expr.loc[valid_index, :]
    mirna_expr = mirna_expr.loc[valid_index, :]
    clinical = clinical.loc[valid_index, ["ER_Status_nature2012"]]
    y = np.zeros(len(gene_expr.index), dtype=np.int64)
    y[clinical["ER_Status_nature2012"] == "Positive"] = 1

    target_genes_present = [gene for gene in BRCA_TARGET_GENES if gene in gene_to_idx]
    if len(target_genes_present) != len(BRCA_TARGET_GENES):
        missing = sorted(set(BRCA_TARGET_GENES) - set(target_genes_present))
        raise RuntimeError(f"BRCA target genes missing after notebook-style filtering: {missing}")

    x = np.concatenate(
        [
            np.asarray(gene_expr.values, dtype=np.float32),
            np.asarray(mirna_expr.values, dtype=np.float32),
        ],
        axis=1,
    )
    status_counts = clinical["ER_Status_nature2012"].value_counts(dropna=False).to_dict()

    return PathNetBundle(
        dataset="brca",
        x=x,
        y=y,
        feature_meta=reg_matrix.transpose(),
        knowledge_graph=sub_graph,
        target_features=BRCA_TARGET_GENES,
        model_kind="gene_mirna",
        n_primary_features=int(gene_expr.shape[1]),
        metadata={
            "label_definition": "ER_Status_nature2012 non-missing labels; Positive encoded as 1 and all other non-missing labels encoded as 0, matching example_gene.ipynb.",
            "notebook_reference": "example_gene.ipynb",
            "n_samples": int(x.shape[0]),
            "n_input_features": int(x.shape[1]),
            "n_network_genes_after_filtering": int(gene_expr.shape[1]),
            "n_mirnas_after_filtering": int(mirna_expr.shape[1]),
            "n_mirna_gene_links": int(reg_matrix.nnz),
            "class_counts": {"0": int((y == 0).sum()), "1": int((y == 1).sum())},
            "er_status_counts": {str(k): int(v) for k, v in status_counts.items()},
            "target_genes": BRCA_TARGET_GENES,
            "preprocessing": "Network-gene and miRNA filtering follow example_gene.ipynb; no external StandardScaler is applied.",
        },
    )


def load_bundles(dataset_names: list[str]) -> list[PathNetBundle]:
    bundles = []
    if "covid" in dataset_names:
        bundles.append(load_covid_pathnet_bundle())
    if "brca" in dataset_names:
        bundles.append(load_brca_pathnet_bundle())
    return bundles


def notebook_config(dataset: str) -> PathNetConfig:
    if dataset == "covid":
        return PathNetConfig(
            dataset="covid",
            seed=14526,
            maximum_step=2,
            hidden_layers=[],
            dropout=0.1,
            learning_rate=1e-3,
            weight_decay=0.0,
            batch_size=16,
            max_epochs=200,
            notebook_reference="example_meta.ipynb",
        )
    if dataset == "brca":
        return PathNetConfig(
            dataset="brca",
            seed=2781,
            maximum_step=2,
            hidden_layers=[],
            dropout=0.1,
            learning_rate=1e-3,
            weight_decay=0.0,
            batch_size=16,
            max_epochs=100,
            notebook_reference="example_gene.ipynb",
        )
    raise ValueError(dataset)


def configs_for_mode(dataset: str, mode: str) -> list[PathNetConfig]:
    base = notebook_config(dataset)
    if mode == "smoke":
        return [PathNetConfig(**{**asdict(base), "max_epochs": 2})]
    if mode == "notebook":
        return [base]
    if mode == "full":
        seeds = [2781, 14526, 2027, 3407, 91011]
        return [PathNetConfig(**{**asdict(base), "seed": seed}) for seed in seeds]
    raise ValueError(mode)


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


def build_pathnet_model(bundle: PathNetBundle, cfg: PathNetConfig) -> tuple[nn.Module, dict, dict, list]:
    sys.path.insert(0, str(ROOT))
    from PathNet.model import GeneExpressionModel, meta_Net
    from PathNet.utils import getPartitionMatricesList

    partition_mtx_dict, residual_connection_dict, connection_list = getPartitionMatricesList(
        bundle.target_features,
        bundle.knowledge_graph,
        cfg.maximum_step,
        abla_graph=False,
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


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def count_sparse_connections(partition_mtx_dict: dict) -> int:
    total = 0
    for key, matrix in partition_mtx_dict.items():
        if key == "p0":
            if hasattr(matrix, "nnz"):
                total += int(matrix.nnz)
            else:
                total += int(np.count_nonzero(matrix))
        else:
            total += int(np.count_nonzero(matrix))
    return total


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
            y_score.extend(output[:, 1].detach().cpu().numpy().tolist())
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    y_score_arr = np.asarray(y_score)
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred_arr)),
        "roc_auc": safe_auc(y_true_arr, y_score_arr),
    }


def train_one(
    bundle: PathNetBundle,
    cfg: PathNetConfig,
    device: torch.device,
    log_every: int,
) -> tuple[dict, list[dict]]:
    set_seed(cfg.seed)
    warnings.filterwarnings("ignore", category=UserWarning)

    x_train, x_val, x_test, y_train, y_val, y_test = split_train_val_test(bundle.x, bundle.y, cfg.seed)
    train_loader = make_loader(x_train, y_train, cfg.batch_size, shuffle=True)
    val_loader = make_loader(x_val, y_val, cfg.batch_size, shuffle=False)
    test_loader = make_loader(x_test, y_test, cfg.batch_size, shuffle=False)

    model, partition_mtx_dict, _residual_connection_dict, connection_list = build_pathnet_model(bundle, cfg)
    model = model.to(device)
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

        train_metrics = evaluate(model, train_loader, device)
        val_metrics = evaluate(model, val_loader, device)
        epoch_row = {
            "dataset": bundle.dataset,
            "seed": cfg.seed,
            "epoch": epoch,
            "train_accuracy": train_metrics["accuracy"],
            "train_balanced_accuracy": train_metrics["balanced_accuracy"],
            "train_roc_auc": train_metrics["roc_auc"],
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_roc_auc": val_metrics["roc_auc"],
        }
        epoch_rows.append(epoch_row)
        if log_every > 0 and (epoch == 1 or epoch == cfg.max_epochs or epoch % log_every == 0):
            print(
                f"  epoch={epoch:03d}/{cfg.max_epochs} "
                f"train_acc={train_metrics['accuracy']:.3f} "
                f"val_acc={val_metrics['accuracy']:.3f} "
                f"val_auc={val_metrics['roc_auc']:.3f}",
                flush=True,
            )

    train_seconds = time.perf_counter() - started
    train_metrics = evaluate(model, train_loader, device)
    val_metrics = evaluate(model, val_loader, device)
    test_metrics = evaluate(model, test_loader, device)
    best_val_accuracy_epoch = max(epoch_rows, key=lambda row: row["val_accuracy"])["epoch"]
    best_val_auc_epoch = max(epoch_rows, key=lambda row: row["val_roc_auc"])["epoch"]

    result = {
        "dataset": bundle.dataset,
        "model": "PathNet",
        "setting": "notebook_hyperparameters_external_test_split",
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
        "final_epoch": cfg.max_epochs,
        "best_val_accuracy_epoch": int(best_val_accuracy_epoch),
        "best_val_auc_epoch": int(best_val_auc_epoch),
        "train_seconds": float(train_seconds),
        "n_parameters": n_parameters,
        "n_sparse_connections": n_sparse_connections,
        "n_connection_layers": int(len(connection_list)),
        "notebook_reference": cfg.notebook_reference,
    }
    for prefix, metrics in [("train", train_metrics), ("val", val_metrics), ("test", test_metrics)]:
        for metric, value in metrics.items():
            result[f"{prefix}_{metric}"] = value

    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result, epoch_rows


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "test_accuracy",
        "test_balanced_accuracy",
        "test_roc_auc",
        "val_accuracy",
        "val_balanced_accuracy",
        "val_roc_auc",
        "train_seconds",
        "n_parameters",
        "n_sparse_connections",
    ]
    rows = []
    for dataset, frame in results.groupby("dataset"):
        row = {"dataset": dataset, "model": "PathNet", "n_runs": int(len(frame))}
        for col in cols:
            row[f"{col}_mean"] = float(frame[col].mean())
            row[f"{col}_std"] = float(frame[col].std(ddof=1)) if len(frame) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows).sort_values("dataset")


def write_metadata(
    output_dir: Path,
    bundles: list[PathNetBundle],
    configs: list[PathNetConfig],
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    metadata = {
        "purpose": "PathNet notebook experiments rerun under the benchmark external train/validation/test evaluation setting.",
        "mode": args.mode,
        "command": " ".join(sys.argv),
        "device": str(device),
        "split": "Stratified 60/20/20 train/validation/test split, using the run seed for the first split and seed+17 for validation split.",
        "model_training": "Fixed epoch training with Adam and no early stopping, following the notebooks.",
        "datasets": {bundle.dataset: bundle.metadata for bundle in bundles},
        "configs": [asdict(cfg) for cfg in configs],
        "notes": [
            "COVID preprocessing and labels follow example_meta.ipynb.",
            "BRCA preprocessing and labels follow example_gene.ipynb, including coding all non-missing non-Positive ER labels as 0.",
            "No extra StandardScaler is applied beyond the preprocessing already present in the notebooks.",
            "The reported test metrics are held-out metrics from the external benchmark split, not training accuracy.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"metadata_pathnet_{args.mode}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    table_dir = output_dir / "tables"
    script_copy_dir = output_dir / "scripts"
    table_dir.mkdir(parents=True, exist_ok=True)
    script_copy_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), script_copy_dir / Path(__file__).name)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    bundles = load_bundles(args.datasets)
    configs = [cfg for bundle in bundles for cfg in configs_for_mode(bundle.dataset, args.mode)]
    write_metadata(output_dir, bundles, configs, args, device)

    rows = []
    epoch_rows = []
    total = sum(len(configs_for_mode(bundle.dataset, args.mode)) for bundle in bundles)
    current = 0
    for bundle in bundles:
        bundle_configs = configs_for_mode(bundle.dataset, args.mode)
        print(
            f"[dataset] {bundle.dataset}: n={bundle.x.shape[0]}, p={bundle.x.shape[1]}, "
            f"class_counts={bundle.metadata['class_counts']}",
            flush=True,
        )
        for cfg in bundle_configs:
            current += 1
            print(
                f"[run {current}/{total}] {bundle.dataset} PathNet "
                f"seed={cfg.seed}, epochs={cfg.max_epochs}, batch={cfg.batch_size}",
                flush=True,
            )
            row, history = train_one(bundle, cfg, device, args.log_every)
            rows.append(row)
            epoch_rows.extend(history)
            print(
                f"  test_auc={row['test_roc_auc']:.3f} "
                f"test_bacc={row['test_balanced_accuracy']:.3f} "
                f"test_acc={row['test_accuracy']:.3f} "
                f"sec={row['train_seconds']:.1f}",
                flush=True,
            )

    results = pd.DataFrame(rows)
    history = pd.DataFrame(epoch_rows)
    summary = summarize(results)
    result_path = table_dir / f"real_data_pathnet_results_{args.mode}.csv"
    history_path = table_dir / f"real_data_pathnet_epoch_history_{args.mode}.csv"
    summary_path = table_dir / f"real_data_pathnet_summary_{args.mode}.csv"
    results.to_csv(result_path, index=False)
    history.to_csv(history_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"[done] results: {result_path}", flush=True)
    print(f"[done] epoch history: {history_path}", flush=True)
    print(f"[done] summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
