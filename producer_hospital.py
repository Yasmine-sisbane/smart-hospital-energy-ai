# =========================
# producer_hospital.py
# Producteur Kafka des données internes hospitalières
# Mode : historical replay streaming
# =========================

import json
import time
from datetime import datetime

import pandas as pd
from kafka import KafkaProducer

from config import (
    KAFKA_BOOTSTRAP_SERVERS,
    HOSPITAL_RAW_TOPIC,
    CSV_FILE_PATH,
    HOSPITAL_REPLAY_SLEEP_SECONDS,
    HOSPITAL_MAX_ROWS,
)


# Colonnes qui ne doivent PAS être envoyées dans hospital_raw_data
# Elles viennent plus tard des APIs ou de l'IA.
COLUMNS_TO_REMOVE = [
    # API météo / soleil / UV
    "outside_temp_c",
    "outside_humidity_pct",
    "outside_pressure_hpa",
    "wind_speed_mps",
    "solar_radiation_wm2",
    "direct_radiation_wm2",
    "diffuse_radiation_wm2",
    "uv_index",
    "sunshine_duration_min",

    # API qualité de l'air
    "aqi",
    "pm25",
    "pm10",
    "no2",
    "o3",

    # Mobilité / STM
    "stm_mobility_index",
    "stm_incident_flag",

    # Sorties IA
    "predicted_energy_kwh",
    "residual_energy_kwh",
    "anomaly_label",
    "anomaly_type",
    "anomaly_score",
]


producer = KafkaProducer(
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
    api_version=(0, 10, 1),
    value_serializer=lambda v: json.dumps(
        v,
        ensure_ascii=False,
        allow_nan=False
    ).encode("utf-8")
)


def load_csv(file_path):
    """
    Charge le CSV hospitalier.
    Le fichier utilise le séparateur ;
    """
    df = pd.read_csv(file_path, sep=";")
    df.columns = df.columns.str.strip()

    # Remplacer NaN par None pour produire du JSON propre
    df = df.astype(object).where(pd.notnull(df), None)

    if HOSPITAL_MAX_ROWS is not None:
        df = df.head(HOSPITAL_MAX_ROWS)

    print("Colonnes détectées dans le dataset :")
    print(df.columns.tolist())

    return df.to_dict(orient="records")


def clean_hospital_row(row):
    """
    Supprime les colonnes externes et IA.
    Le topic hospital_raw_data doit contenir seulement les données internes.
    """
    cleaned = row.copy()

    for col in COLUMNS_TO_REMOVE:
        cleaned.pop(col, None)

    return cleaned


def fix_numeric_units(event):
    """
    Corrige les valeurs qui ont perdu leur virgule après ouverture Excel.
    Exemple :
    2645 -> 26.45
    257  -> 25.7
    1552 -> 15.52
    """

    energy_columns = [
        "zone_energy_kwh",
    ]

    for col in energy_columns:
        value = event.get(col)

        try:
            value = float(value)
        except (TypeError, ValueError):
            continue

        if 100 <= value < 1000:
            event[col] = round(value / 10.0, 2)
        elif value >= 1000:
            event[col] = round(value / 100.0, 2)
        else:
            event[col] = round(value, 2)

    return event

def main():
    records = load_csv(CSV_FILE_PATH)

    print(f"{len(records)} lignes hospitalières chargées.")
    print(f"Streaming vers Kafka topic : {HOSPITAL_RAW_TOPIC}")
    print(f"Mode : historical replay streaming")
    print(f"Sleep : {HOSPITAL_REPLAY_SLEEP_SECONDS} seconde(s)")

    for index, row in enumerate(records, start=1):
        try:
            hospital_event = clean_hospital_row(row)
            hospital_event = fix_numeric_units(hospital_event)

            # Date originale de la mesure capteur
            hospital_event["event_time"] = hospital_event.get("timestamp")

            # Date réelle d'entrée dans Kafka
            hospital_event["ingestion_time"] = datetime.now().isoformat(timespec="seconds")

            # Métadonnées professionnelles
            hospital_event["event_type"] = "hospital_sensor_data"
            hospital_event["source"] = "hospital_sensor_csv_replay"
            hospital_event["stream_mode"] = "historical_replay"
            hospital_event["replay_mode"] = True

            future = producer.send(HOSPITAL_RAW_TOPIC, value=hospital_event)
            metadata = future.get(timeout=10)

            print(
                f"Sent {index}/{len(records)} | "
                f"Topic={metadata.topic} | "
                f"Partition={metadata.partition} | "
                f"Offset={metadata.offset} | "
                f"EventTime={hospital_event.get('event_time')} | "
                f"IngestionTime={hospital_event.get('ingestion_time')} | "
                f"Zone={hospital_event.get('zone_name')} | "
                f"Energy={hospital_event.get('zone_energy_kwh')} kWh"
            )

            time.sleep(HOSPITAL_REPLAY_SLEEP_SECONDS)

        except Exception as e:
            print(
                f"Erreur ligne {index} | "
                f"{row.get('timestamp')} | "
                f"{row.get('zone_name')} : {e}"
            )

    producer.flush()
    producer.close()

    print("Streaming hospitalier terminé.")


if __name__ == "__main__":
    main()