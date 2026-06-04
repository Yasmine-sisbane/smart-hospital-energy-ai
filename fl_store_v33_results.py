"""
fl_store_v33_results.py

Stockage des résultats Federated Learning V3.3 dans TimescaleDB.

Ce script lit :
- outputs_fl/v33_final_comparison.csv
- outputs_fl/personalized_fl_evaluation.csv

Puis insère les résultats dans :
- federated_model_comparison
- federated_personalized_models

Il crée aussi federated_training_metrics si elle n'existe pas encore.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs_fl"
MODEL_DIR = PROJECT_DIR / "models"

COMPARISON_CSV = OUTPUT_DIR / "v33_final_comparison.csv"
PERSONALIZED_CSV = OUTPUT_DIR / "personalized_fl_evaluation.csv"

DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "hospital_iot",
    "user": "hospital_user",
    "password": "hospital_password",
}


def parse_bool(value):
    if isinstance(value, bool):
        return value

    if value is None:
        return None

    value_str = str(value).strip().lower()

    if value_str in {"true", "1", "yes", "oui"}:
        return True

    if value_str in {"false", "0", "no", "non"}:
        return False

    return None


def safe_float(value):
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def create_tables(conn):
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")

        cur.execute("""
            CREATE TABLE IF NOT EXISTS federated_training_metrics (
                ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                round_number INTEGER,
                client_id TEXT,
                horizon TEXT,
                loss DOUBLE PRECISION,
                mae DOUBLE PRECISION,
                rmse DOUBLE PRECISION,
                r2 DOUBLE PRECISION,
                training_time_seconds DOUBLE PRECISION,
                n_samples INTEGER
            );
        """)

        try:
            cur.execute("""
                SELECT create_hypertable(
                    'federated_training_metrics',
                    'ts',
                    if_not_exists => TRUE
                );
            """)
        except Exception:
            conn.rollback()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS federated_model_comparison (
                id BIGSERIAL PRIMARY KEY,
                ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                run_id TEXT NOT NULL,
                approach TEXT NOT NULL,
                data_centralized BOOLEAN,
                mae DOUBLE PRECISION,
                rmse DOUBLE PRECISION,
                r2 DOUBLE PRECISION,
                interpretation TEXT
            );
        """)

        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_federated_model_comparison_ts
            ON federated_model_comparison (ts DESC);
        """)

        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_federated_model_comparison_approach
            ON federated_model_comparison (approach);
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS federated_personalized_models (
                id BIGSERIAL PRIMARY KEY,
                ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                run_id TEXT NOT NULL,
                client_id TEXT NOT NULL,
                approach TEXT,
                model TEXT,
                horizon TEXT,
                model_path TEXT,
                metadata_path TEXT,
                data_centralized BOOLEAN,
                mae DOUBLE PRECISION,
                rmse DOUBLE PRECISION,
                r2 DOUBLE PRECISION
            );
        """)

        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_federated_personalized_models_ts
            ON federated_personalized_models (ts DESC);
        """)

        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_federated_personalized_models_client
            ON federated_personalized_models (client_id);
        """)

    conn.commit()


def store_comparison(conn, run_id: str):
    if not COMPARISON_CSV.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {COMPARISON_CSV}\n"
            "Relance d'abord : python fl_v33_evaluate_personalized.py"
        )

    df = pd.read_csv(COMPARISON_CSV)

    required_cols = ["approach", "data_centralized", "mae", "rmse", "r2", "interpretation"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Colonne manquante dans {COMPARISON_CSV}: {col}")

    rows = []

    for _, row in df.iterrows():
        rows.append((
            run_id,
            str(row["approach"]),
            parse_bool(row["data_centralized"]),
            safe_float(row["mae"]),
            safe_float(row["rmse"]),
            safe_float(row["r2"]),
            str(row["interpretation"]),
        ))

    sql = """
        INSERT INTO federated_model_comparison (
            run_id,
            approach,
            data_centralized,
            mae,
            rmse,
            r2,
            interpretation
        )
        VALUES %s;
    """

    with conn.cursor() as cur:
        execute_values(cur, sql, rows)

    conn.commit()

    print(f"[OK] {len(rows)} lignes insérées dans federated_model_comparison.")


def store_personalized_models(conn, run_id: str):
    if not PERSONALIZED_CSV.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {PERSONALIZED_CSV}\n"
            "Relance d'abord : python fl_v33_evaluate_personalized.py"
        )

    df = pd.read_csv(PERSONALIZED_CSV)

    required_cols = [
        "approach",
        "model",
        "client_id",
        "mae",
        "rmse",
        "r2",
        "model_path",
        "data_centralized",
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Colonne manquante dans {PERSONALIZED_CSV}: {col}")

    rows = []

    for _, row in df.iterrows():
        client_id = str(row["client_id"])
        model_path = str(row["model_path"])

        metadata_path = None

        if client_id not in {"AVERAGE", "GLOBAL"}:
            possible_metadata = MODEL_DIR / f"fl_personalized_{client_id}_t60_metadata.json"
            if possible_metadata.exists():
                metadata_path = str(possible_metadata)

        rows.append((
            run_id,
            client_id,
            str(row["approach"]),
            str(row["model"]),
            "T+1h",
            model_path,
            metadata_path,
            parse_bool(row["data_centralized"]),
            safe_float(row["mae"]),
            safe_float(row["rmse"]),
            safe_float(row["r2"]),
        ))

    sql = """
        INSERT INTO federated_personalized_models (
            run_id,
            client_id,
            approach,
            model,
            horizon,
            model_path,
            metadata_path,
            data_centralized,
            mae,
            rmse,
            r2
        )
        VALUES %s;
    """

    with conn.cursor() as cur:
        execute_values(cur, sql, rows)

    conn.commit()

    print(f"[OK] {len(rows)} lignes insérées dans federated_personalized_models.")


def show_counts(conn):
    queries = [
        ("federated_training_metrics", "SELECT COUNT(*) FROM federated_training_metrics;"),
        ("federated_model_comparison", "SELECT COUNT(*) FROM federated_model_comparison;"),
        ("federated_personalized_models", "SELECT COUNT(*) FROM federated_personalized_models;"),
    ]

    print("\n========== Vérification TimescaleDB ==========")

    with conn.cursor() as cur:
        for table_name, sql in queries:
            cur.execute(sql)
            count = cur.fetchone()[0]
            print(f"{table_name}: {count} lignes")


def main():
    parser = argparse.ArgumentParser(
        description="Stocker les résultats FL V3.3 dans TimescaleDB."
    )

    parser.add_argument(
        "--run-id",
        default=None,
        help="Identifiant du run. Par défaut : v33_YYYYMMDD_HHMMSS",
    )

    args = parser.parse_args()

    run_id = args.run_id or datetime.now().strftime("v33_%Y%m%d_%H%M%S")

    print("========== Stockage Federated Learning V3.3 ==========")
    print("Run ID:", run_id)
    print("Comparison CSV:", COMPARISON_CSV)
    print("Personalized CSV:", PERSONALIZED_CSV)

    conn = get_connection()

    try:
        create_tables(conn)
        store_comparison(conn, run_id)
        store_personalized_models(conn, run_id)
        show_counts(conn)

        print("\n[SUCCESS] Résultats FL V3.3 stockés dans TimescaleDB.")

    finally:
        conn.close()


if __name__ == "__main__":
    main()