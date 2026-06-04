"""
fl_build_global_scaler.py

Construit un scaler global commun pour le Federated Learning.

V3.2 :
- scaler global pour les features X
- scaler global pour la cible y
- objectif : rendre FedAvg plus stable entre ICU, ER et LAB

Commande :
    python fl_build_global_scaler.py
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from fl_config import (
    CLIENTS,
    DATA_DIR,
    FEATURE_COLUMNS,
    GLOBAL_SCALER_PATH,
    MODEL_DIR,
    TARGET_COL,
)


def load_client_dataframe(client_id: str) -> pd.DataFrame:
    path = DATA_DIR / f"client_{client_id.lower()}.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset client introuvable : {path}\n"
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


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    x_train_parts = []
    y_train_parts = []

    print("========== Construction du scaler global FL V3.2 ==========")

    for client_id in CLIENTS:
        df = load_client_dataframe(client_id)

        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx].copy()

        X_train = train_df[FEATURE_COLUMNS].astype(float).values
        y_train = train_df[TARGET_COL].astype(float).values

        x_train_parts.append(X_train)
        y_train_parts.append(y_train)

        print(
            f"[SCALER] Client={client_id} | "
            f"Train samples={len(train_df)}"
        )

    X_all_train = np.vstack(x_train_parts)
    y_all_train = np.concatenate(y_train_parts)

    x_mean = X_all_train.mean(axis=0)
    x_std = X_all_train.std(axis=0)
    x_std[x_std == 0] = 1.0

    y_mean = float(y_all_train.mean())
    y_std = float(y_all_train.std())

    if y_std == 0:
        y_std = 1.0

    scaler = {
        "version": "V3.2",
        "description": "Global scaler for federated learning: features X and target y",
        "target": TARGET_COL,
        "clients": CLIENTS,
        "feature_columns": FEATURE_COLUMNS,
        "x_mean": x_mean.tolist(),
        "x_std": x_std.tolist(),
        "target_mean": y_mean,
        "target_std": y_std,
        "n_train_samples": int(X_all_train.shape[0]),
    }

    GLOBAL_SCALER_PATH.write_text(
        json.dumps(scaler, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("")
    print("[SCALER] Scaler global sauvegarde :")
    print(GLOBAL_SCALER_PATH)
    print("[SCALER] Total train samples :", X_all_train.shape[0])
    print("[SCALER] Target mean :", round(y_mean, 6))
    print("[SCALER] Target std  :", round(y_std, 6))
    print("")
    print("Scaler global V3.2 termine avec succes.")


if __name__ == "__main__":
    main()