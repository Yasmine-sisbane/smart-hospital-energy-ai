"""
run_federated_pipeline.py

Lance automatiquement tout le pipeline Federated Learning Kafka dans un seul terminal.

Version 2 :
1. Vérifie les dépendances Python dans le venv courant
2. Optionnel : prépare les datasets clients ICU / ER / LAB
3. Lance fl_metrics_writer.py
4. Lance les clients ICU, ER, LAB
5. Attend que les clients écoutent Kafka
6. Lance fl_aggregator.py
7. Affiche tous les logs dans un seul terminal
8. Arrête proprement les processus à la fin

Commandes :
    python run_federated_pipeline.py --rounds 10

Avec installation automatique des dépendances :
    python run_federated_pipeline.py --install-deps --rounds 10

Avec préparation des datasets :
    python run_federated_pipeline.py --prepare --rounds 10

Sans TimescaleDB metrics writer :
    python run_federated_pipeline.py --rounds 10 --no-metrics-writer
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from fl_config import CLIENTS, NUM_ROUNDS


PROJECT_DIR = Path(__file__).resolve().parent

REQUIRED_FILES = [
    "fl_client.py",
    "fl_aggregator.py",
    "fl_config.py",
    "fl_model.py",
]

OPTIONAL_FILES = [
    "fl_prepare_clients.py",
    "fl_metrics_writer.py",
]

BASE_DEPENDENCIES = {
    "pandas": "pandas",
    "numpy": "numpy",
    "torch": "torch",
    "kafka": "kafka-python",
}

METRICS_DEPENDENCIES = {
    "psycopg2": "psycopg2-binary",
}

processes: list[tuple[str, subprocess.Popen]] = []


def check_python_dependencies(use_metrics_writer: bool, install_deps: bool) -> None:
    print("\n[CHECK] Verification des dependances Python dans l'environnement courant...")
    print(f"[PYTHON] {sys.executable}")

    dependencies = dict(BASE_DEPENDENCIES)

    if use_metrics_writer:
        dependencies.update(METRICS_DEPENDENCIES)

    missing_packages = []

    for import_name, pip_name in dependencies.items():
        if importlib.util.find_spec(import_name) is None:
            missing_packages.append(pip_name)

    if not missing_packages:
        print("[OK] Toutes les dependances Python sont disponibles.")
        return

    print("\n[WARN] Dependances manquantes dans cet environnement :")
    for package in missing_packages:
        print(f"  - {package}")

    if install_deps:
        print("\n[INSTALL] Installation automatique des dependances manquantes...")

        command = [
            sys.executable,
            "-m",
            "pip",
            "install",
            *missing_packages,
        ]

        result = subprocess.run(
            command,
            cwd=str(PROJECT_DIR),
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        if result.returncode != 0:
            print("\n[ERREUR] Installation des dependances echouee.")
            sys.exit(result.returncode)

        importlib.invalidate_caches()
        print("[OK] Installation terminee.")
        return

    print("\n[ERREUR] Installe les dependances dans le venv courant avec :")
    print("  python -m pip install pandas numpy torch kafka-python psycopg2-binary")
    print("\nOu relance directement :")
    print("  python run_federated_pipeline.py --install-deps --rounds 10")
    sys.exit(1)


def check_files() -> None:
    print("\n[CHECK] Verification des fichiers Federated Learning...")

    missing = []

    for filename in REQUIRED_FILES:
        if not (PROJECT_DIR / filename).exists():
            missing.append(filename)

    if missing:
        print("\n[ERREUR] Fichiers obligatoires manquants :")
        for filename in missing:
            print(f"  - {filename}")
        sys.exit(1)

    print("[OK] Fichiers obligatoires trouves.")

    for filename in OPTIONAL_FILES:
        if not (PROJECT_DIR / filename).exists():
            print(f"[WARN] Fichier optionnel absent : {filename}")


def stream_output(name: str, process: subprocess.Popen) -> None:
    try:
        if process.stdout is None:
            return

        for line in process.stdout:
            line = line.rstrip()
            if line:
                print(f"[{name}] {line}")

    except Exception as exc:
        print(f"[{name}] Erreur lecture sortie : {exc}")


def start_process(name: str, args: list[str]) -> subprocess.Popen:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    command = [sys.executable, "-u"] + args

    print(f"\n[START] {name}")
    print("Commande :", " ".join(command))

    process = subprocess.Popen(
        command,
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


def stop_all() -> None:
    print("\n[STOP] Arret des processus Federated Learning...")

    for name, process in processes:
        if process.poll() is None:
            print(f"[STOP] {name}")
            try:
                process.terminate()
            except Exception:
                pass

    time.sleep(2)

    for name, process in processes:
        if process.poll() is None:
            print(f"[KILL] {name}")
            try:
                process.kill()
            except Exception:
                pass

    print("[OK] Tous les processus sont arretes.")


def prepare_clients(csv_path: str) -> None:
    script_path = PROJECT_DIR / "fl_prepare_clients.py"

    if not script_path.exists():
        print("[WARN] fl_prepare_clients.py introuvable. Preparation ignoree.")
        return

    print("\n[PREPARE] Preparation des datasets clients ICU / ER / LAB...")

    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--csv",
            csv_path,
        ],
        cwd=str(PROJECT_DIR),
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if result.returncode != 0:
        print("[ERREUR] La preparation des clients a echoue.")
        sys.exit(result.returncode)

    print("[OK] Preparation des clients terminee.")


def wait(seconds: int, message: str) -> None:
    print(f"\n[WAIT] {message} ({seconds}s)")
    time.sleep(seconds)


def print_header(rounds: int, use_metrics_writer: bool, prepare: bool) -> None:
    print("=" * 90)
    print("FEDERATED LEARNING KAFKA PIPELINE - VERSION 2")
    print("=" * 90)
    print(f"Projet         : {PROJECT_DIR}")
    print(f"Python         : {sys.executable}")
    print(f"Clients        : {', '.join(CLIENTS)}")
    print(f"Rounds         : {rounds}")
    print(f"Preparation    : {'activee' if prepare else 'non'}")
    print(f"Metrics writer : {'active' if use_metrics_writer else 'non'}")
    print("=" * 90)

    print("\nOrdre de lancement :")
    print("  1. Verification dependances")
    print("  2. fl_metrics_writer.py")
    print("  3. fl_client.py --client-id ICU")
    print("  4. fl_client.py --client-id ER")
    print("  5. fl_client.py --client-id LAB")
    print("  6. fl_aggregator.py --rounds N")

    print("\nObjectif :")
    print("  - Les donnees ICU / ER / LAB restent locales.")
    print("  - Les clients entrainent un modele local.")
    print("  - Les parametres sont envoyes via Kafka.")
    print("  - L'agregateur applique FedAvg.")
    print("  - Les metriques peuvent etre stockees dans TimescaleDB.")
    print("=" * 90)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Federated Learning Kafka pipeline in one terminal."
    )

    parser.add_argument(
        "--rounds",
        type=int,
        default=NUM_ROUNDS,
        help="Nombre de rounds Federated Learning.",
    )

    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Preparer les datasets clients avant de lancer le pipeline.",
    )

    parser.add_argument(
        "--csv",
        type=str,
        default="data/hospital_energy_semisynthetic_3zones_from_PUHY_Aurora.csv",
        help="Chemin du dataset source.",
    )

    parser.add_argument(
        "--no-metrics-writer",
        action="store_true",
        help="Ne pas lancer fl_metrics_writer.py.",
    )

    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Installer automatiquement les dependances manquantes dans le venv courant.",
    )

    parser.add_argument(
        "--client-wait",
        type=int,
        default=8,
        help="Temps d'attente avant de lancer l'agregateur.",
    )

    args = parser.parse_args()

    use_metrics_writer = not args.no_metrics_writer

    print_header(
        rounds=args.rounds,
        use_metrics_writer=use_metrics_writer,
        prepare=args.prepare,
    )

    check_python_dependencies(
        use_metrics_writer=use_metrics_writer,
        install_deps=args.install_deps,
    )

    check_files()

    if args.prepare:
        prepare_clients(args.csv)

    try:
        metrics_writer_path = PROJECT_DIR / "fl_metrics_writer.py"

        if use_metrics_writer and metrics_writer_path.exists():
            start_process("FL_METRICS_WRITER", ["fl_metrics_writer.py"])
            wait(3, "Demarrage du metrics writer")
        else:
            print("\n[INFO] Metrics writer non lance.")

        for client_id in CLIENTS:
            start_process(
                f"CLIENT_{client_id}",
                ["fl_client.py", "--client-id", client_id],
            )
            time.sleep(2)

        wait(
            args.client_wait,
            "Attente pour que les clients ecoutent le topic fl_global_model",
        )

        aggregator = start_process(
            "AGGREGATOR",
            ["fl_aggregator.py", "--rounds", str(args.rounds)],
        )

        print("\n[RUNNING] Pipeline Federated Learning lance.")
        print("[INFO] Attente de la fin de l'agregateur...")

        return_code = aggregator.wait()

        print(f"\n[AGGREGATOR] Termine avec code : {return_code}")

        if return_code == 0:
            print("\n[SUCCESS] Federated Learning termine avec succes.")
            print("Fichiers attendus :")
            print("  models/federated_mlp_t60_global.pt")
            print("  models/federated_mlp_t60_metadata.json")
        else:
            print("\n[WARN] L'agregateur s'est termine avec une erreur.")

    except KeyboardInterrupt:
        print("\n[INTERRUPT] Interruption utilisateur.")

    finally:
        stop_all()


if __name__ == "__main__":
    main()