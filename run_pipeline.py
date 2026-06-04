# =========================
# run_pipeline.py
# Pipeline Kafka hospitalier uniquement
# Sans ML / Sans XGBoost / Sans PUHY-Aurora
# =========================

import os
import subprocess
import sys
import time
import threading
from pathlib import Path


# =========================
# Configuration globale
# =========================

os.environ["PYTHONIOENCODING"] = "utf-8"

PROJECT_DIR = Path(__file__).resolve().parent

# Mettre True si tu veux que ce script lance aussi Docker Compose
START_DOCKER_COMPOSE = True

# Arrêter automatiquement quand producer_hospital.py termine
AUTO_STOP_AFTER_HOSPITAL = True

# Temps d'attente après la fin du producer hospitalier
# pour laisser analytics et Timescale écrire les derniers messages
FINAL_DRAIN_SECONDS = 20

# Mettre False si tu ne veux pas lancer monitor_outputs.py
RUN_MONITOR = True


# =========================
# Services du pipeline Kafka hospitalier
# =========================
# IMPORTANT :
# Ce pipeline ne lance PAS :
# - ml_kafka_consumer_pro.py
# - ml_prediction_service.py
# - producer_hospital_puhy_aurora.py
#
# Ces scripts appartiennent au pipeline PUHY/Aurora séparé.

SERVICES = [
    ("ENRICHMENT", "enrichment_service.py", 5),
    ("ANALYTICS", "analytics_service.py", 5),
    ("TIMESCALE", "timescale_writer.py", 5),
    ("WEATHER", "producer_weather.py", 5),
    ("AIR", "producer_air_quality.py", 5),
    ("HOSPITAL", "producer_hospital.py", 0),
]

if RUN_MONITOR:
    SERVICES.insert(3, ("MONITOR", "monitor_outputs.py", 3))


processes = []


# =========================
# Vérifications
# =========================

def check_required_files():
    print("\n[CHECK] Vérification des fichiers nécessaires...")

    missing_scripts = []

    for _, script_name, _ in SERVICES:
        script_path = PROJECT_DIR / script_name

        if not script_path.exists():
            missing_scripts.append(script_name)

    if missing_scripts:
        print("\n[ERREUR] Scripts manquants :")
        for script in missing_scripts:
            print(f"  - {script}")

        print("\nCorrige les fichiers manquants avant de relancer.")
        sys.exit(1)

    print("[OK] Tous les scripts Kafka hospitaliers sont présents.")


# =========================
# Docker
# =========================

def start_docker_compose():
    if not START_DOCKER_COMPOSE:
        print("\n[DOCKER] Démarrage Docker Compose ignoré.")
        return

    print("\n[DOCKER] Démarrage de docker compose...")

    try:
        result = subprocess.run(
            ["docker", "compose", "up", "-d"],
            cwd=str(PROJECT_DIR),
            text=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
        )

        if result.returncode != 0:
            print("[ERREUR] docker compose up -d a échoué.")
            print(result.stdout)
            print(result.stderr)
            sys.exit(1)

        if result.stdout.strip():
            print(result.stdout.strip())

        print("[OK] Docker Compose lancé.")

    except FileNotFoundError:
        print("[ERREUR] Docker n'est pas trouvé.")
        print("Lance Docker Desktop, puis réessaie.")
        sys.exit(1)


# =========================
# Gestion des processus
# =========================

def stream_output(name, process):
    try:
        for line in process.stdout:
            line = line.rstrip()

            if line:
                print(f"[{name}] {line}")

    except Exception as e:
        print(f"[{name}] Erreur lecture sortie : {e}")


def start_service(name, script_name):
    script_path = PROJECT_DIR / script_name

    print(f"\n[DÉMARRAGE] {name} -> {script_name}")

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"

    process = subprocess.Popen(
        [sys.executable, "-u", str(script_path)],
        cwd=str(PROJECT_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
    )

    processes.append((name, process))

    thread = threading.Thread(
        target=stream_output,
        args=(name, process),
        daemon=True,
    )
    thread.start()

    return process


def stop_all():
    print("\n[ARRÊT] Fermeture des services Python...")

    for name, process in reversed(processes):
        if process.poll() is None:
            print(f"[ARRÊT] {name}")
            process.terminate()

    time.sleep(2)

    for name, process in reversed(processes):
        if process.poll() is None:
            print(f"[FORCE STOP] {name}")
            process.kill()

    print("[OK] Tous les services Python sont arrêtés.")
    print("[INFO] Docker reste lancé.")
    print("[INFO] Pour arrêter Docker : docker compose down")


def check_processes_alive():
    """
    Vérifie si un service s'est arrêté trop tôt.
    HOSPITAL peut terminer normalement.
    """
    for name, process in processes:
        exit_code = process.poll()

        if exit_code is not None and name != "HOSPITAL":
            print(f"\n[ATTENTION] {name} s'est arrêté avec le code {exit_code}.")


# =========================
# Affichage
# =========================

def print_startup_info():
    print("=" * 90)
    print("SMART HOSPITAL KAFKA PIPELINE")
    print("=" * 90)
    print(f"Dossier projet : {PROJECT_DIR}")
    print("Ce script lance uniquement le pipeline Kafka hospitalier.")
    print("Aucun service ML / XGBoost / PUHY-Aurora n'est lancé ici.")
    print("Il s'arrête automatiquement après la fin de producer_hospital.py.")
    print("Pour arrêter manuellement : CTRL + C")
    print("=" * 90)

    print("\nOrdre de lancement :")

    for name, script_name, wait_seconds in SERVICES:
        print(f"  - {name:<12} {script_name:<30} wait={wait_seconds}s")

    print("\nPipeline exécuté :")
    print("  producer_hospital.py")
    print("        ↓")
    print("  Kafka topic : hospital_raw_data")
    print("        ↓")
    print("  enrichment_service.py + producer_weather.py + producer_air_quality.py")
    print("        ↓")
    print("  Kafka topic : hospital_enriched_data")
    print("        ↓")
    print("  analytics_service.py")
    print("        ↓")
    print("  Kafka topics : hospital_metrics / hospital_anomalies / hospital_alerts")
    print("        ↓")
    print("  timescale_writer.py")
    print("        ↓")
    print("  TimescaleDB : hospital_enriched_events / hospital_metrics / hospital_anomalies / hospital_alerts")
    print("=" * 90)


# =========================
# Main
# =========================

def main():
    print_startup_info()
    check_required_files()
    start_docker_compose()

    hospital_process = None

    try:
        print("\n[START] Lancement des services Python...")

        for name, script_name, wait_seconds in SERVICES:
            process = start_service(name, script_name)

            if name == "HOSPITAL":
                hospital_process = process

            if wait_seconds > 0:
                print(f"[WAIT] Attente {wait_seconds} secondes...")
                time.sleep(wait_seconds)
                check_processes_alive()

        print("\n[OK] Pipeline Kafka hospitalier lancé.")

        if AUTO_STOP_AFTER_HOSPITAL and hospital_process is not None:
            print("\n[INFO] Attente de la fin de producer_hospital.py...")
            print("[INFO] Si tu veux arrêter avant la fin : CTRL + C")

            hospital_exit_code = hospital_process.wait()

            print(f"\n[INFO] producer_hospital.py terminé avec code : {hospital_exit_code}")

            print(
                f"[INFO] Attente {FINAL_DRAIN_SECONDS} secondes pour laisser "
                "analytics et TimescaleDB traiter les derniers messages..."
            )

            time.sleep(FINAL_DRAIN_SECONDS)

            print("\n[INFO] Arrêt automatique du pipeline Kafka hospitalier.")
            return

        print("\n[INFO] Mode keep-alive actif.")
        print("[INFO] Pour arrêter : CTRL + C")

        while True:
            time.sleep(1)
            check_processes_alive()

    except KeyboardInterrupt:
        print("\n[INFO] CTRL+C détecté.")

    finally:
        stop_all()


if __name__ == "__main__":
    main()