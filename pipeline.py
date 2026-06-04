# =========================
# pipeline.py
# Lance tout le pipeline hospitalier
# =========================

import os
import sys
import time
import signal
import subprocess
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent

SERVICES = [
    "producer_weather.py",
    "producer_air_quality.py",
    "enrichment_service.py",
    "analytics_service.py",
    "timescale_writer.py",
    "producer_hospital.py",
]

START_DELAY_SECONDS = 5

processes = []


def check_files():
    print("Verification des fichiers...")

    missing = []

    for service in SERVICES:
        file_path = PROJECT_DIR / service
        if not file_path.exists():
            missing.append(service)

    config_path = PROJECT_DIR / "config.py"
    if not config_path.exists():
        missing.append("config.py")

    if missing:
        print("\nFichiers manquants :")
        for file in missing:
            print(f" - {file}")

        print("\nCorrige les fichiers manquants puis relance pipeline.py")
        sys.exit(1)

    print("Tous les fichiers necessaires sont presents.\n")


def start_service(script_name):
    script_path = PROJECT_DIR / script_name

    print(f"Lancement : {script_name}")

    if os.name == "nt":
        # Windows : ouvre chaque service dans une nouvelle fenetre CMD
        process = subprocess.Popen(
            [
                "cmd",
                "/k",
                f'cd /d "{PROJECT_DIR}" && python "{script_path}"'
            ],
            creationflags=subprocess.CREATE_NEW_CONSOLE
        )
    else:
        # Linux / macOS
        process = subprocess.Popen(
            [sys.executable, str(script_path)],
            cwd=str(PROJECT_DIR)
        )

    processes.append(process)


def stop_all_services():
    print("\nArret du pipeline...")

    for process in processes:
        try:
            process.terminate()
        except Exception:
            pass

    print("Pipeline arrete.")


def main():
    print("=" * 60)
    print("HOSPITAL PYTHON PIPELINE")
    print("=" * 60)
    print(f"Dossier projet : {PROJECT_DIR}")
    print()

    check_files()

    print("Ordre de lancement :")
    for service in SERVICES:
        print(f" - {service}")

    print("\nDemarrage...\n")

    for service in SERVICES:
        start_service(service)
        time.sleep(START_DELAY_SECONDS)

    print("\nTous les services sont lances.")
    print("Appuie sur CTRL + C ici pour arreter le pipeline.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop_all_services()


if __name__ == "__main__":
    main()