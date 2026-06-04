# =========================
# run_pipeline.py
# Exécuteur du pipeline Kafka hospitalier complet
# =========================

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parent

REQUIRED_FILES = [
    "config.py",
    "producer_weather.py",
    "producer_air_quality.py",
    "producer_hospital.py",
    "enrichment_service.py",
    "analytics_service.py",
    "timescale_writer.py",
]

OPTIONAL_FILES = [
    "consumer_enriched.py",
]


def check_files():
    missing = []

    for filename in REQUIRED_FILES:
        if not (PIPELINE_DIR / filename).exists():
            missing.append(filename)

    if missing:
        print("Fichiers manquants :")
        for filename in missing:
            print(f"  - {filename}")

        print("\nPlace run_pipeline.py dans le même dossier que tes scripts.")
        sys.exit(1)


def stream_logs(process_name, process):
    """
    Affiche les logs de chaque processus avec un préfixe.
    """
    for line in iter(process.stdout.readline, ""):
        if not line:
            break
        print(f"[{process_name}] {line.rstrip()}")

    process.stdout.close()


def start_process(name, script_name, processes):
    script_path = PIPELINE_DIR / script_name

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    print(f"\nDémarrage : {name} -> {script_name}")

    process = subprocess.Popen(
        [sys.executable, str(script_path)],
        cwd=str(PIPELINE_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    thread = threading.Thread(
        target=stream_logs,
        args=(name, process),
        daemon=True,
    )
    thread.start()

    processes[name] = process
    return process


def stop_processes(processes):
    print("\nArrêt du pipeline...")

    for name, process in reversed(list(processes.items())):
        if process.poll() is None:
            print(f"Arrêt : {name}")

            try:
                if os.name == "nt":
                    process.terminate()
                else:
                    process.send_signal(signal.SIGTERM)

                process.wait(timeout=8)

            except subprocess.TimeoutExpired:
                print(f"Forçage arrêt : {name}")
                process.kill()

            except Exception as e:
                print(f"Erreur arrêt {name}: {e}")

    print("Pipeline arrêté.")


def wait_and_check(processes, seconds):
    """
    Attend quelques secondes et détecte si un service crash tout de suite.
    """
    for _ in range(seconds):
        time.sleep(1)

        for name, process in processes.items():
            exit_code = process.poll()
            if exit_code is not None:
                print(f"\nERREUR : {name} s'est arrêté avec le code {exit_code}.")
                stop_processes(processes)
                sys.exit(exit_code)


def main():
    parser = argparse.ArgumentParser(
        description="Lance le pipeline Kafka hospitalier complet."
    )

    parser.add_argument(
        "--with-monitor",
        action="store_true",
        help="Lance aussi consumer_enriched.py pour afficher les données enrichies.",
    )

    parser.add_argument(
        "--context-wait",
        type=int,
        default=10,
        help="Temps d'attente avant de lancer producer_hospital.py, pour laisser météo/air produire un premier contexte.",
    )

    parser.add_argument(
        "--keep-running",
        action="store_true",
        help="Garde tous les services actifs après la fin de producer_hospital.py.",
    )

    parser.add_argument(
        "--shutdown-delay",
        type=int,
        default=15,
        help="Temps d'attente après producer_hospital.py avant d'arrêter les services si --keep-running n'est pas utilisé.",
    )

    args = parser.parse_args()

    check_files()

    processes = {}

    try:
        # 1. Writer TimescaleDB
        start_process(
            name="timescale_writer",
            script_name="timescale_writer.py",
            processes=processes,
        )

        # 2. Service d'enrichissement
        start_process(
            name="enrichment_service",
            script_name="enrichment_service.py",
            processes=processes,
        )

        # 3. Service analytics
        start_process(
            name="analytics_service",
            script_name="analytics_service.py",
            processes=processes,
        )

        # 4. Consumer de monitoring optionnel
        if args.with_monitor:
            if (PIPELINE_DIR / "consumer_enriched.py").exists():
                start_process(
                    name="consumer_enriched",
                    script_name="consumer_enriched.py",
                    processes=processes,
                )
            else:
                print("consumer_enriched.py introuvable, monitoring ignoré.")

        # Petite vérification initiale
        wait_and_check(processes, seconds=5)

        # 5. Producteurs contexte externe
        start_process(
            name="producer_weather",
            script_name="producer_weather.py",
            processes=processes,
        )

        start_process(
            name="producer_air_quality",
            script_name="producer_air_quality.py",
            processes=processes,
        )

        print(
            f"\nAttente {args.context_wait}s pour recevoir un premier contexte météo/air..."
        )
        wait_and_check(processes, seconds=args.context_wait)

        # 6. Producer hospitalier en dernier
        hospital_process = start_process(
            name="producer_hospital",
            script_name="producer_hospital.py",
            processes=processes,
        )

        print("\nPipeline lancé.")
        print("Ctrl+C pour arrêter manuellement.")

        # Le producer hospitalier termine quand le CSV est rejoué.
        hospital_exit_code = hospital_process.wait()

        if hospital_exit_code != 0:
            print(f"\nproducer_hospital terminé avec erreur : {hospital_exit_code}")
        else:
            print("\nproducer_hospital terminé : replay CSV fini.")

        if args.keep_running:
            print("\nServices maintenus actifs. Ctrl+C pour arrêter.")
            while True:
                time.sleep(2)

                for name, process in processes.items():
                    exit_code = process.poll()
                    if exit_code is not None and name != "producer_hospital":
                        print(f"Attention : {name} s'est arrêté avec le code {exit_code}")

        else:
            print(
                f"\nAttente {args.shutdown_delay}s pour laisser analytics et Timescale traiter les derniers messages..."
            )
            time.sleep(args.shutdown_delay)
            stop_processes(processes)

    except KeyboardInterrupt:
        stop_processes(processes)

    except Exception as e:
        print(f"\nErreur pipeline : {e}")
        stop_processes(processes)
        sys.exit(1)


if __name__ == "__main__":
    main()