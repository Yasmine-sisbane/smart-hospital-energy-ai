# =========================
# run_pipeline_puhy_aurora.py
# Lance tout le pipeline Kafka + ML + TimescaleDB
# =========================

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import psycopg2


# ============================================================
# 1. Configuration projet
# ============================================================

PROJECT_DIR = Path(
    r"C:\Users\lenovo\OneDrive\Documents\hospital_kafka_project"
)

ML_SERVICE_SCRIPT = PROJECT_DIR / "ml_prediction_service.py"
PRODUCER_SCRIPT = PROJECT_DIR / "producer_hospital_puhy_aurora.py"

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "hospital_iot")
DB_USER = os.getenv("DB_USER", "hospital_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "hospital_password")


# ============================================================
# 2. Database helpers
# ============================================================

def get_connection():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )


def reset_predictions_table():
    print("\n[DB] Nettoyage table energy_predictions...")

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE energy_predictions;")

    print("[DB] Table energy_predictions vidée.")


def count_predictions():
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM energy_predictions;")
            return cur.fetchone()[0]


def print_prediction_summary():
    print("\n========== Résumé des prédictions ==========")

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM energy_predictions;")
            total = cur.fetchone()[0]

            print(f"Total prédictions : {total}")

            cur.execute(
                """
                SELECT
                    anomaly_severity,
                    COUNT(*)
                FROM energy_predictions
                GROUP BY anomaly_severity
                ORDER BY COUNT(*) DESC;
                """
            )

            print("\nRépartition par sévérité :")
            rows = cur.fetchall()

            for severity, count in rows:
                print(f"- {severity}: {count}")

            cur.execute(
                """
                SELECT
                    event_time,
                    target_time,
                    zone_name,
                    energy_current,
                    predicted_energy_kwh,
                    actual_energy_kwh,
                    abs_error_kwh,
                    anomaly_severity
                FROM energy_predictions
                ORDER BY event_time DESC
                LIMIT 10;
                """
            )

            print("\nDernières prédictions :")
            latest = cur.fetchall()

            for row in latest:
                print(row)


# ============================================================
# 3. Process helpers
# ============================================================

def check_files():
    missing = []

    if not ML_SERVICE_SCRIPT.exists():
        missing.append(str(ML_SERVICE_SCRIPT))

    if not PRODUCER_SCRIPT.exists():
        missing.append(str(PRODUCER_SCRIPT))

    if missing:
        raise FileNotFoundError(
            "Fichiers manquants :\n" + "\n".join(missing)
        )


def start_process(script_path, name):
    print(f"\n[{name}] Démarrage : {script_path.name}")

    process = subprocess.Popen(
        [sys.executable, str(script_path)],
        cwd=str(PROJECT_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    return process


def stream_process_output(process, name, stop_when_finished=True):
    if process.stdout is None:
        return

    for line in process.stdout:
        print(f"[{name}] {line}", end="")

    if stop_when_finished:
        return_code = process.wait()
        print(f"\n[{name}] Process terminé avec code : {return_code}")


def terminate_process(process, name):
    if process is None:
        return

    if process.poll() is None:
        print(f"\n[{name}] Arrêt du process...")
        process.terminate()

        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            print(f"[{name}] Kill forcé...")
            process.kill()

    print(f"[{name}] Arrêté.")


# ============================================================
# 4. Pipeline
# ============================================================

def run_pipeline(reset=False, expected_rows=15000, wait_timeout=600):
    check_files()

    print("========== Pipeline PUHY/Aurora Kafka + ML ==========")
    print("Projet :", PROJECT_DIR)
    print("ML service :", ML_SERVICE_SCRIPT)
    print("Producer :", PRODUCER_SCRIPT)
    print("DB :", f"{DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

    # Vérifier DB
    try:
        initial_count = count_predictions()
        print(f"\n[DB] Connexion OK. Prédictions actuelles : {initial_count}")
    except Exception as e:
        print("\n[ERREUR DB] Impossible de se connecter à PostgreSQL/TimescaleDB.")
        print("Détail :", e)
        print("\nVérifie que Docker/TimescaleDB est lancé.")
        return

    if reset:
        reset_predictions_table()
        initial_count = 0
    else:
        initial_count = count_predictions()

    ml_process = None

    try:
        # 1. Lancer le service ML
        ml_process = start_process(ML_SERVICE_SCRIPT, "ML")

        print("\n[PIPELINE] Attente démarrage ML service...")
        time.sleep(6)

        # 2. Lancer le producer
        producer_process = start_process(PRODUCER_SCRIPT, "PRODUCER")

        # Lire sortie producer jusqu'à fin
        stream_process_output(producer_process, "PRODUCER")

        # 3. Attendre les insertions en DB
        print("\n[PIPELINE] Attente des prédictions en base...")

        start_time = time.time()
        target_count = initial_count + expected_rows

        while True:
            current_count = count_predictions()

            print(
                f"[DB] Prédictions : {current_count}/{target_count}",
                end="\r"
            )

            if current_count >= target_count:
                print()
                break

            if time.time() - start_time > wait_timeout:
                print("\n[WARN] Timeout atteint avant le nombre attendu.")
                break

            time.sleep(2)

        # 4. Résumé
        print_prediction_summary()

        print("\n========== Pipeline terminé avec succès ==========")

    except KeyboardInterrupt:
        print("\n[PIPELINE] Interrompu par utilisateur.")

    except Exception as e:
        print("\n[PIPELINE] Erreur :", e)

    finally:
        terminate_process(ml_process, "ML")


# ============================================================
# 5. Entrée CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Lancer tout le pipeline Kafka + ML + TimescaleDB."
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help="Vider energy_predictions avant de lancer le pipeline."
    )

    parser.add_argument(
        "--expected",
        type=int,
        default=15000,
        help="Nombre de prédictions attendues. Par défaut : 15000 = 5000 événements x 3 horizons."
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Temps maximum d'attente en secondes. Par défaut : 600."
    )

    args = parser.parse_args()

    run_pipeline(
        reset=args.reset,
        expected_rows=args.expected,
        wait_timeout=args.timeout,
    )