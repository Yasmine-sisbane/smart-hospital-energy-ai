"""
Preparation des datasets locaux pour Federated Learning.

Sorties:
- data/fl_clients/client_icu.csv
- data/fl_clients/client_er.csv
- data/fl_clients/client_lab.csv

Important pour l'enonce:
- Apres cette etape, chaque client utilise uniquement son fichier local.
- Les clients n'envoient jamais ces donnees via Kafka.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fl_config import CLIENTS, CSV_PATH, DATA_DIR, FEATURE_COLUMNS, TARGET_COL, HORIZON_STEPS


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week_num"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week_num"] >= 5).astype(int)
    df["month"] = df["timestamp"].dt.month
    return df


def add_target_if_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["zone_name", "timestamp"]).copy()

    if TARGET_COL not in df.columns:
        df[TARGET_COL] = df.groupby("zone_name")["zone_energy_kwh"].shift(-HORIZON_STEPS)

    return df


def clean_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[columns] = df[columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


def prepare_clients(csv_path: Path, output_dir: Path) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV introuvable: {csv_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    required = ["timestamp", "zone_name", "zone_energy_kwh"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Colonne obligatoire manquante: {col}")

    df = add_time_features(df)
    df = add_target_if_missing(df)
    df = clean_numeric(df, FEATURE_COLUMNS + [TARGET_COL])
    df = df.dropna(subset=[TARGET_COL]).copy()

    print("Dataset source:", df.shape)
    print("Target:", TARGET_COL)
    print("Features:", FEATURE_COLUMNS)

    for client in CLIENTS:
        local_df = df[df["zone_name"] == client].copy()
        if local_df.empty:
            raise ValueError(f"Aucune donnee trouvee pour le client {client}")

        # On garde timestamp, zone, features, target pour evaluation locale.
        keep_cols = ["timestamp", "zone_name"] + FEATURE_COLUMNS + [TARGET_COL]
        local_df = local_df[keep_cols].sort_values("timestamp")

        out_path = output_dir / f"client_{client.lower()}.csv"
        local_df.to_csv(out_path, index=False)

        print(f"Client {client}: {local_df.shape} -> {out_path}")

    print("Preparation terminee.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare les donnees locales ICU/ER/LAB pour FL.")
    parser.add_argument("--csv", default=str(CSV_PATH), help="Chemin du dataset PUHY/Aurora.")
    parser.add_argument("--out", default=str(DATA_DIR), help="Dossier de sortie des clients.")
    args = parser.parse_args()

    prepare_clients(Path(args.csv), Path(args.out))
