"""
fl_v3_evaluation.py

V3.2 Federated Learning :
1. Charger le modele global federe sauvegarde.
2. Evaluer ce modele sur les jeux de test ICU, ER, LAB.
3. Entrainer un modele MLP centralise classique avec les memes features.
4. Utiliser le meme scaler global pour X et y.
5. Comparer Federated Learning vs approche centralisee.
6. Sauvegarder les resultats dans outputs_fl/.

Commande :
    python fl_v3_evaluation.py

Avec plus d'epochs :
    python fl_v3_evaluation.py --central-epochs 50
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fl_config import (
    BATCH_SIZE,
    CLIENTS,
    DATA_DIR,
    FEATURE_COLUMNS,
    GLOBAL_MODEL_PATH,
    GLOBAL_SCALER_PATH,
    HIDDEN_DIM,
    LEARNING_RATE,
    MODEL_DIR,
    PROJECT_DIR,
    TARGET_COL,
)

from fl_model import build_model, regression_metrics


OUTPUT_DIR = PROJECT_DIR / "outputs_fl"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEDERATED_EVAL_PATH = OUTPUT_DIR / "federated_global_evaluation.csv"
CENTRALIZED_EVAL_PATH = OUTPUT_DIR / "centralized_mlp_evaluation.csv"
COMPARISON_PATH = OUTPUT_DIR / "comparison_centralized_vs_federated.csv"
SUMMARY_PATH = OUTPUT_DIR / "v3_summary.json"

CENTRALIZED_MODEL_PATH = MODEL_DIR / "centralized_mlp_t60.pt"


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_global_scaler():
    if not GLOBAL_SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Scaler global introuvable : {GLOBAL_SCALER_PATH}\n"
            "Lance d'abord : python fl_build_global_scaler.py"
        )

    with open(GLOBAL_SCALER_PATH, "r", encoding="utf-8") as f:
        scaler = json.load(f)

    x_mean = np.array(scaler["x_mean"], dtype=np.float32)
    x_std = np.array(scaler["x_std"], dtype=np.float32)
    x_std[x_std == 0] = 1.0

    target_mean = float(scaler["target_mean"])
    target_std = float(scaler["target_std"])

    if target_std == 0:
        target_std = 1.0

    return x_mean, x_std, target_mean, target_std


def normalize_x(X, x_mean, x_std):
    return (X - x_mean) / x_std


def normalize_y(y, target_mean, target_std):
    return (y - target_mean) / target_std


def denormalize_y(y_norm, target_mean, target_std):
    return (y_norm * target_std) + target_mean


def load_client_dataframe(client_id: str) -> pd.DataFrame:
    path = DATA_DIR / f"client_{client_id.lower()}.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset client introuvable: {path}\n"
            "Lance d'abord : "
            "python fl_prepare_clients.py --csv data/hospital_energy_semisynthetic_3zones_from_PUHY_Aurora.csv"
        )

    df = pd.read_csv(path)
    df = df.dropna(subset=[TARGET_COL]).copy()

    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COL]).copy()

    return df


def split_client_raw(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS].astype(float).values
    y = df[TARGET_COL].astype(float).values

    split_idx = int(len(df) * 0.8)

    X_train = X[:split_idx]
    y_train = y[:split_idx]

    X_test = X[split_idx:]
    y_test = y[split_idx:]

    return X_train, y_train, X_test, y_test


def predict_denormalized(model, X_norm, target_mean, target_std):
    model.eval()

    with torch.no_grad():
        y_pred_norm = model(torch.tensor(X_norm, dtype=torch.float32)).numpy()

    y_pred = denormalize_y(
        y_pred_norm,
        target_mean,
        target_std,
    )

    return np.asarray(y_pred).reshape(-1)


def evaluate_federated_global_model():
    if not GLOBAL_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modele federe introuvable: {GLOBAL_MODEL_PATH}\n"
            "Lance d'abord : python run_federated_pipeline.py --rounds 10"
        )

    x_mean, x_std, target_mean, target_std = load_global_scaler()

    input_dim = len(FEATURE_COLUMNS)
    model = build_model(input_dim=input_dim, hidden_dim=HIDDEN_DIM)

    state_dict = torch.load(GLOBAL_MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)

    rows = []
    all_y_true = []
    all_y_pred = []

    print("\n========== Evaluation du modele federe global V3.2 ==========")
    print(f"[FEDERATED] Scaler global utilise : {GLOBAL_SCALER_PATH}")

    for client_id in CLIENTS:
        df = load_client_dataframe(client_id)
        _, _, X_test_raw, y_test = split_client_raw(df)

        X_test_norm = normalize_x(
            X_test_raw,
            x_mean,
            x_std,
        )

        y_pred = predict_denormalized(
            model,
            X_test_norm,
            target_mean,
            target_std,
        )

        y_test = np.asarray(y_test).reshape(-1)

        metrics = regression_metrics(y_test, y_pred)

        rows.append({
            "approach": "Federated Learning",
            "model": "MLP + FedAvg",
            "client_id": client_id,
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "r2": metrics["r2"],
            "n_test_samples": len(y_test),
            "data_centralized": False,
        })

        all_y_true.append(y_test)
        all_y_pred.append(y_pred)

        print(
            f"[FEDERATED] Client={client_id} | "
            f"MAE={metrics['mae']:.4f} | "
            f"RMSE={metrics['rmse']:.4f} | "
            f"R2={metrics['r2']:.4f}"
        )

    all_y_true = np.concatenate(all_y_true)
    all_y_pred = np.concatenate(all_y_pred)

    global_metrics = regression_metrics(all_y_true, all_y_pred)

    rows.append({
        "approach": "Federated Learning",
        "model": "MLP + FedAvg",
        "client_id": "GLOBAL",
        "mae": global_metrics["mae"],
        "rmse": global_metrics["rmse"],
        "r2": global_metrics["r2"],
        "n_test_samples": len(all_y_true),
        "data_centralized": False,
    })

    print(
        f"[FEDERATED GLOBAL] "
        f"MAE={global_metrics['mae']:.4f} | "
        f"RMSE={global_metrics['rmse']:.4f} | "
        f"R2={global_metrics['r2']:.4f}"
    )

    result_df = pd.DataFrame(rows)
    result_df.to_csv(FEDERATED_EVAL_PATH, index=False)

    return result_df


def train_centralized_mlp(epochs: int):
    print("\n========== Entrainement du modele centralise MLP V3.2 ==========")

    x_mean, x_std, target_mean, target_std = load_global_scaler()

    train_X_parts = []
    train_y_norm_parts = []

    test_by_client = {}

    for client_id in CLIENTS:
        df = load_client_dataframe(client_id)

        X_train_raw, y_train_raw, X_test_raw, y_test_raw = split_client_raw(df)

        X_train_norm = normalize_x(
            X_train_raw,
            x_mean,
            x_std,
        )

        y_train_norm = normalize_y(
            y_train_raw,
            target_mean,
            target_std,
        )

        train_X_parts.append(X_train_norm)
        train_y_norm_parts.append(y_train_norm)

        test_by_client[client_id] = {
            "X_test_raw": X_test_raw,
            "y_test_raw": y_test_raw,
        }

    X_train_all = np.vstack(train_X_parts)
    y_train_norm_all = np.concatenate(train_y_norm_parts)

    input_dim = len(FEATURE_COLUMNS)
    model = build_model(input_dim=input_dim, hidden_dim=HIDDEN_DIM)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    dataset = TensorDataset(
        torch.tensor(X_train_all, dtype=torch.float32),
        torch.tensor(y_train_norm_all, dtype=torch.float32),
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []

        for xb, yb in loader:
            optimizer.zero_grad()

            pred_norm = model(xb)
            loss = criterion(pred_norm, yb)

            loss.backward()
            optimizer.step()

            epoch_losses.append(float(loss.item()))

        if epoch == 1 or epoch == epochs or epoch % 5 == 0:
            print(
                f"[CENTRALIZED] Epoch={epoch}/{epochs} | "
                f"LossNorm={np.mean(epoch_losses):.4f}"
            )

    training_time = time.time() - start_time

    torch.save(model.state_dict(), CENTRALIZED_MODEL_PATH)

    rows = []
    all_y_true = []
    all_y_pred = []

    print("\n========== Evaluation du modele centralise MLP V3.2 ==========")
    print(f"[CENTRALIZED] Scaler global utilise : {GLOBAL_SCALER_PATH}")

    for client_id, values in test_by_client.items():
        X_test_norm = normalize_x(
            values["X_test_raw"],
            x_mean,
            x_std,
        )

        y_test = np.asarray(values["y_test_raw"]).reshape(-1)

        y_pred = predict_denormalized(
            model,
            X_test_norm,
            target_mean,
            target_std,
        )

        metrics = regression_metrics(y_test, y_pred)

        rows.append({
            "approach": "Centralized",
            "model": "MLP Centralized",
            "client_id": client_id,
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "r2": metrics["r2"],
            "n_test_samples": len(y_test),
            "training_time_seconds": training_time,
            "data_centralized": True,
        })

        all_y_true.append(y_test)
        all_y_pred.append(y_pred)

        print(
            f"[CENTRALIZED] Client={client_id} | "
            f"MAE={metrics['mae']:.4f} | "
            f"RMSE={metrics['rmse']:.4f} | "
            f"R2={metrics['r2']:.4f}"
        )

    all_y_true = np.concatenate(all_y_true)
    all_y_pred = np.concatenate(all_y_pred)

    global_metrics = regression_metrics(all_y_true, all_y_pred)

    rows.append({
        "approach": "Centralized",
        "model": "MLP Centralized",
        "client_id": "GLOBAL",
        "mae": global_metrics["mae"],
        "rmse": global_metrics["rmse"],
        "r2": global_metrics["r2"],
        "n_test_samples": len(all_y_true),
        "training_time_seconds": training_time,
        "data_centralized": True,
    })

    print(
        f"[CENTRALIZED GLOBAL] "
        f"MAE={global_metrics['mae']:.4f} | "
        f"RMSE={global_metrics['rmse']:.4f} | "
        f"R2={global_metrics['r2']:.4f}"
    )

    result_df = pd.DataFrame(rows)
    result_df.to_csv(CENTRALIZED_EVAL_PATH, index=False)

    return result_df


def build_comparison(federated_df: pd.DataFrame, centralized_df: pd.DataFrame):
    comparison_rows = []

    for client_id in list(CLIENTS) + ["GLOBAL"]:
        fed_row = federated_df[federated_df["client_id"] == client_id].iloc[0]
        central_row = centralized_df[centralized_df["client_id"] == client_id].iloc[0]

        comparison_rows.append({
            "client_id": client_id,

            "centralized_model": "MLP Centralized",
            "centralized_data_centralized": True,
            "centralized_mae": central_row["mae"],
            "centralized_rmse": central_row["rmse"],
            "centralized_r2": central_row["r2"],

            "federated_model": "MLP + FedAvg",
            "federated_data_centralized": False,
            "federated_mae": fed_row["mae"],
            "federated_rmse": fed_row["rmse"],
            "federated_r2": fed_row["r2"],

            "mae_difference_federated_minus_centralized": (
                fed_row["mae"] - central_row["mae"]
            ),
            "privacy_advantage": "High for federated, because raw data stays local",
        })

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(COMPARISON_PATH, index=False)

    return comparison_df


def save_summary(federated_df, centralized_df, comparison_df, central_epochs):
    fed_global = federated_df[federated_df["client_id"] == "GLOBAL"].iloc[0]
    central_global = centralized_df[centralized_df["client_id"] == "GLOBAL"].iloc[0]
    comparison_global = comparison_df[comparison_df["client_id"] == "GLOBAL"].iloc[0]

    summary = {
        "version": "V3.2",
        "objective": "Final evaluation with global X and y scaler",
        "target": TARGET_COL,
        "clients": CLIENTS,
        "feature_columns": FEATURE_COLUMNS,
        "global_scaler": str(GLOBAL_SCALER_PATH),
        "central_epochs": central_epochs,
        "federated_global": {
            "mae": float(fed_global["mae"]),
            "rmse": float(fed_global["rmse"]),
            "r2": float(fed_global["r2"]),
        },
        "centralized_global": {
            "mae": float(central_global["mae"]),
            "rmse": float(central_global["rmse"]),
            "r2": float(central_global["r2"]),
        },
        "mae_difference_federated_minus_centralized": float(
            comparison_global["mae_difference_federated_minus_centralized"]
        ),
        "outputs": {
            "federated_evaluation": str(FEDERATED_EVAL_PATH),
            "centralized_evaluation": str(CENTRALIZED_EVAL_PATH),
            "comparison": str(COMPARISON_PATH),
            "centralized_model": str(CENTRALIZED_MODEL_PATH),
        },
    }

    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(
        description="V3.2 evaluation: Federated global model vs centralized MLP."
    )

    parser.add_argument(
        "--central-epochs",
        type=int,
        default=30,
        help="Nombre d'epochs pour le modele MLP centralise.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    args = parser.parse_args()

    set_seed(args.seed)

    federated_df = evaluate_federated_global_model()
    centralized_df = train_centralized_mlp(epochs=args.central_epochs)
    comparison_df = build_comparison(federated_df, centralized_df)

    save_summary(
        federated_df=federated_df,
        centralized_df=centralized_df,
        comparison_df=comparison_df,
        central_epochs=args.central_epochs,
    )

    print("\n========== Fichiers V3.2 sauvegardes ==========")
    print("-", FEDERATED_EVAL_PATH)
    print("-", CENTRALIZED_EVAL_PATH)
    print("-", COMPARISON_PATH)
    print("-", SUMMARY_PATH)
    print("-", CENTRALIZED_MODEL_PATH)

    print("\n========== Comparaison globale ==========")

    global_comparison = comparison_df[
        comparison_df["client_id"] == "GLOBAL"
    ].iloc[0]

    print(
        f"Centralized MAE : {global_comparison['centralized_mae']:.4f} | "
        f"Federated MAE : {global_comparison['federated_mae']:.4f} | "
        f"Difference FL - Centralized : "
        f"{global_comparison['mae_difference_federated_minus_centralized']:.4f}"
    )

    print("\nV3.2 terminee avec succes.")


if __name__ == "__main__":
    main()