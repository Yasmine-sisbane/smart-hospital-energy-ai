
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from kafka import KafkaProducer



try:
    from config import KAFKA_BOOTSTRAP_SERVERS, PUHY_AURORA_TOPIC
except ImportError:
    KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
    PUHY_AURORA_TOPIC = "puhy_aurora_enriched_data"


CSV_PATH = Path(
    r"C:\Users\lenovo\OneDrive\Documents\hospital_kafka_project\data\hospital_energy_semisynthetic_3zones_from_PUHY_Aurora.csv"
)

TOPIC = PUHY_AURORA_TOPIC

DEFAULT_MAX_ROWS = 5000
DEFAULT_SLEEP_SECONDS = 0.02
DEFAULT_LOG_EVERY = 100


# ============================================================
# 2. Helpers
# ============================================================

def parse_max_rows(value):
    """
    Permet :
    --max-rows 5000
    --max-rows none
    --max-rows all
    """
    if value is None:
        return DEFAULT_MAX_ROWS

    value = str(value).strip().lower()

    if value in ["none", "all", "full"]:
        return None

    return int(value)


def clean_value(value):
    """
    Convertit les valeurs pandas/numpy en types JSON compatibles.
    Évite les erreurs avec NaN, Timestamp, int64, float64, etc.
    """
    if pd.isna(value):
        return None

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    return value


def row_to_event(row):
    """
    Transforme une ligne du CSV en événement Kafka.

    On garde :
    - les colonnes originales du CSV
    - target_observed_energy_next_15min si déjà présent
    - target_observed_energy_t60
    - target_observed_energy_t24h
    - target_anomaly_t60
    - target_anomaly_t24h
    """
    event = {}

    for col, value in row.items():
        event[col] = clean_value(value)

    timestamp = pd.to_datetime(row["timestamp"])

    event["event_time"] = timestamp.isoformat()
    event["ingestion_time"] = datetime.now().isoformat(timespec="seconds")

    event["event_type"] = "puhy_aurora_ml_event"
    event["source"] = "puhy_aurora_semisynthetic_replay"

    event.setdefault("hospital_id", "H1")
    event.setdefault("building_id", "B1")

    zone_name = event.get("zone_name", "UNKNOWN")

    if zone_name == "ICU":
        event.setdefault("floor_id", "F1")
        event.setdefault("zone_id", "Z1")
        event.setdefault("service_type", "Intensive Care Unit")
    elif zone_name == "ER":
        event.setdefault("floor_id", "F1")
        event.setdefault("zone_id", "Z2")
        event.setdefault("service_type", "Emergency")
    elif zone_name == "LAB":
        event.setdefault("floor_id", "F2")
        event.setdefault("zone_id", "Z3")
        event.setdefault("service_type", "Laboratory")
    else:
        event.setdefault("floor_id", "F0")
        event.setdefault("zone_id", "Z0")
        event.setdefault("service_type", "Unknown")

    return event


def should_log(sent_count, total_rows, log_every):
    return (
        sent_count == 1
        or sent_count == total_rows
        or sent_count % log_every == 0
    )


def create_kafka_producer():
    return KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        api_version=(0, 10, 1),
        value_serializer=lambda v: json.dumps(
            v,
            ensure_ascii=False,
            allow_nan=False
        ).encode("utf-8"),
    )


def add_multi_horizon_targets(df):
    """
    Ajoute les vraies valeurs futures pour évaluer les prédictions multi-horizon.

    Le CSV est en pas de 15 minutes :
    T+1h  = 4 lignes après
    T+24h = 96 lignes après
    """

    df = df.sort_values(["zone_name", "timestamp"]).copy()

    df["target_observed_energy_t60"] = (
        df.groupby("zone_name")["zone_energy_kwh"].shift(-4)
    )

    df["target_observed_energy_t24h"] = (
        df.groupby("zone_name")["zone_energy_kwh"].shift(-96)
    )

    if "anomaly_label" in df.columns:
        df["target_anomaly_t60"] = (
            df.groupby("zone_name")["anomaly_label"].shift(-4)
        )

        df["target_anomaly_t24h"] = (
            df.groupby("zone_name")["anomaly_label"].shift(-96)
        )
    else:
        df["target_anomaly_t60"] = None
        df["target_anomaly_t24h"] = None

    # On remet l'ordre temporel global pour simuler un vrai flux Kafka
    df = df.sort_values(["timestamp", "zone_name"]).copy()

    return df


# ============================================================
# 3. Main
# ============================================================

def main(max_rows, sleep_seconds, log_every):
    print("Producer PUHY/Aurora started.")
    print("CSV        :", CSV_PATH)
    print("Topic      :", TOPIC)
    print("Kafka      :", KAFKA_BOOTSTRAP_SERVERS)
    print("Max rows   :", max_rows if max_rows is not None else "ALL")
    print("Sleep      :", sleep_seconds)
    print("Log every  :", log_every)

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV introuvable : {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    print("Dataset chargé :", df.shape)

    required_columns = [
        "timestamp",
        "zone_name",
        "zone_energy_kwh",
    ]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Colonne obligatoire introuvable dans le CSV : {col}")

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Ajouter les cibles futures AVANT de limiter à max_rows
    # comme ça T+1h et T+24h restent corrects.
    df = add_multi_horizon_targets(df)

    if max_rows is not None:
        df = df.head(max_rows).copy()

    total_rows = len(df)

    print("Nombre de lignes à envoyer :", total_rows)

    if total_rows == 0:
        print("Aucune ligne à envoyer.")
        return

    producer = create_kafka_producer()

    sent_count = 0
    start_time = time.time()

    try:
        for _, row in df.iterrows():
            event = row_to_event(row)

            future = producer.send(TOPIC, value=event)
            metadata = future.get(timeout=10)

            sent_count += 1

            if should_log(sent_count, total_rows, log_every):
                print(
                    f"[PUHY/AURORA] Sent {sent_count}/{total_rows} | "
                    f"Topic={metadata.topic} | "
                    f"Partition={metadata.partition} | "
                    f"Offset={metadata.offset} | "
                    f"EventTime={event.get('event_time')} | "
                    f"Zone={event.get('zone_name')} | "
                    f"Energy={event.get('zone_energy_kwh')} kWh | "
                    f"T+1h={event.get('target_observed_energy_t60')} | "
                    f"T+24h={event.get('target_observed_energy_t24h')}"
                )

            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

        producer.flush()

    finally:
        producer.close()

    elapsed = time.time() - start_time
    rate = sent_count / elapsed if elapsed > 0 else 0

    print("")
    print("Streaming PUHY/Aurora terminé.")
    print(f"Messages envoyés : {sent_count}")
    print(f"Durée            : {elapsed:.2f} secondes")
    print(f"Débit moyen      : {rate:.2f} messages/seconde")


# ============================================================
# 4. CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Producer Kafka PUHY/Aurora pour hospital_enriched_data."
    )

    parser.add_argument(
        "--max-rows",
        default=str(DEFAULT_MAX_ROWS),
        help="Nombre de lignes à envoyer. Exemple: 5000, 300, all, none."
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=DEFAULT_SLEEP_SECONDS,
        help="Pause entre messages Kafka. Exemple: 0.02, 0.05, 0."
    )

    parser.add_argument(
        "--log-every",
        type=int,
        default=DEFAULT_LOG_EVERY,
        help="Afficher un log chaque N messages. Exemple: 100."
    )

    args = parser.parse_args()

    main(
        max_rows=parse_max_rows(args.max_rows),
        sleep_seconds=args.sleep,
        log_every=args.log_every,
    )