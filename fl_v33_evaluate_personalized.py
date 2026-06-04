"""
fl_v33_evaluate_personalized.py

V3.3 :
Evaluation des modeles Federated Learning personnalises.

Objectif :
- Charger les modeles locaux personnalises :
  models/fl_personalized_ICU_t60.pt
  models/fl_personalized_ER_t60.pt
  models/fl_personalized_LAB_t60.pt
- Evaluer chaque modele sur son propre client.
- Comparer avec le modele global FedAvg si disponible.
- Sauvegarder un CSV pour le rapport.

Commande :
    python fl_v33_evaluate_personalized.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fl_config import (
    CLIENTS,
    DATA_DIR,
    FEATURE_COLUMNS,
    GLOBAL_MODEL_PATH,
    GLOBAL_SCALER_PATH,
    HIDDEN_DIM,
    MODEL_DIR,
    PROJECT_DIR,
    TARGET_COL,
)

from fl_model import build_model, regression_metrics


OUTPUT_DIR = PROJECT_DIR / "outputs_fl"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PERSONALIZED_EVAL_PATH = OUTPUT_DIR / "personalized_fl_evaluation.csv"
FINAL_COMPARISON_PATH = OUTPUT_DIR / "v33_final_comparison.csv"


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


def denormalize_y(y_norm, target_mean, target_std):
    return (y_norm * target_std) + target_mean


def load_client_test_data(client_id: str):
    path = DATA_DIR / f"client_{client_id.lower()}.csv"

    if not path.exists():
        raise FileNotFoundError(f"Dataset client introuvable : {path}")

    df = pd.read_csv(path)
    df = df.dropna(subset=[TARGET_COL]).copy()

    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COL]).copy()

    X = df[FEATURE_COLUMNS].astype(float).values
    y = df[TARGET_COL].astype(float).values

    split_idx = int(len(df) * 0.8)

    X_test = X[split_idx:]
    y_test = y[split_idx:]

    return X_test, y_test


def predict_model(model, X_norm, target_mean, target_std):
    model.eval()

    with torch.no_grad():
        y_pred_norm = model(torch.tensor(X_norm, dtype=torch.float32)).numpy()

    y_pred = denormalize_y(y_pred_norm, target_mean, target_std)

    return np.asarray(y_pred).reshape(-1)


def evaluate_state_dict(model_path: Path, X_test_norm, y_test, target_mean, target_std):
    input_dim = len(FEATURE_COLUMNS)

    model = build_model(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
    )

    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)

    y_pred = predict_model(
        model=model,
        X_norm=X_test_norm,
        target_mean=target_mean,
        target_std=target_std,
    )

    y_test = np.asarray(y_test).reshape(-1)

    return regression_metrics(y_test, y_pred)


def evaluate_personalized_models():
    x_mean, x_std, target_mean, target_std = load_global_scaler()

    rows = []

    print("\n========== Evaluation V3.3 : modeles personnalises ==========")

    for client_id in CLIENTS:
        model_path = MODEL_DIR / f"fl_personalized_{client_id}_t60.pt"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Modele personnalise introuvable : {model_path}\n"
                "Relance d'abord : python run_federated_pipeline.py --install-deps --rounds 10"
            )

        X_test_raw, y_test = load_client_test_data(client_id)
        X_test_norm = normalize_x(X_test_raw, x_mean, x_std)

        metrics = evaluate_state_dict(
            model_path=model_path,
            X_test_norm=X_test_norm,
            y_test=y_test,
            target_mean=target_mean,
            target_std=target_std,
        )

        rows.append({
            "approach": "Personalized Federated Learning",
            "model": "Personalized MLP after FedAvg round",
            "client_id": client_id,
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "r2": metrics["r2"],
            "model_path": str(model_path),
            "data_centralized": False,
        })

        print(
            f"[PERSONALIZED] Client={client_id} | "
            f"MAE={metrics['mae']:.4f} | "
            f"RMSE={metrics['rmse']:.4f} | "
            f"R2={metrics['r2']:.4f}"
        )

    df = pd.DataFrame(rows)

    mean_row = {
        "approach": "Personalized Federated Learning",
        "model": "Average personalized local models",
        "client_id": "AVERAGE",
        "mae": df["mae"].mean(),
        "rmse": df["rmse"].mean(),
        "r2": df["r2"].mean(),
        "model_path": "one personalized model per client",
        "data_centralized": False,
    }

    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

    df.to_csv(PERSONALIZED_EVAL_PATH, index=False)

    print(
        f"[PERSONALIZED AVERAGE] "
        f"MAE={mean_row['mae']:.4f} | "
        f"RMSE={mean_row['rmse']:.4f} | "
        f"R2={mean_row['r2']:.4f}"
    )

    return df


def build_final_comparison(personalized_df: pd.DataFrame):
    comparison_rows = []

    centralized_path = OUTPUT_DIR / "centralized_mlp_evaluation.csv"
    fedavg_path = OUTPUT_DIR / "federated_global_evaluation.csv"

    if centralized_path.exists():
        centralized_df = pd.read_csv(centralized_path)
        central_global = centralized_df[centralized_df["client_id"] == "GLOBAL"].iloc[0]

        comparison_rows.append({
            "approach": "Centralized MLP",
            "data_centralized": True,
            "mae": central_global["mae"],
            "rmse": central_global["rmse"],
            "r2": central_global["r2"],
            "interpretation": "Meilleure performance, mais toutes les donnees sont centralisees.",
        })

    if fedavg_path.exists():
        fedavg_df = pd.read_csv(fedavg_path)
        fedavg_global = fedavg_df[fedavg_df["client_id"] == "GLOBAL"].iloc[0]

        comparison_rows.append({
            "approach": "FedAvg global model",
            "data_centralized": False,
            "mae": fedavg_global["mae"],
            "rmse": fedavg_global["rmse"],
            "r2": fedavg_global["r2"],
            "interpretation": "Modele global partage, moins adapte aux zones tres heterogenes.",
        })

    personalized_avg = personalized_df[personalized_df["client_id"] == "AVERAGE"].iloc[0]

    comparison_rows.append({
        "approach": "Personalized Federated Learning",
        "data_centralized": False,
        "mae": personalized_avg["mae"],
        "rmse": personalized_avg["rmse"],
        "r2": personalized_avg["r2"],
        "interpretation": "Bon compromis : donnees locales et modele adapte a chaque zone.",
    })

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(FINAL_COMPARISON_PATH, index=False)

    print("\n========== Comparaison finale V3.3 ==========")
    print(comparison_df.to_string(index=False))

    return comparison_df


def main():
    personalized_df = evaluate_personalized_models()
    build_final_comparison(personalized_df)

    print("\n========== Fichiers V3.3 sauvegardes ==========")
    print("-", PERSONALIZED_EVAL_PATH)
    print("-", FINAL_COMPARISON_PATH)

    print("\nV3.3 terminee avec succes.")


if __name__ == "__main__":
    main()