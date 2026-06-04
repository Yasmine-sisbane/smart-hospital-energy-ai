# =========================
# producer_air_quality.py
# Producteur Kafka qualité de l'air
# =========================

import json
import time
from datetime import datetime

import requests
from kafka import KafkaProducer

from config import (
    KAFKA_BOOTSTRAP_SERVERS,
    AIR_QUALITY_TOPIC,
    LATITUDE,
    LONGITUDE,
    TIMEZONE,
    AIR_QUALITY_REFRESH_SECONDS,
)


producer = KafkaProducer(
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
    api_version=(0, 10, 1),
    value_serializer=lambda v: json.dumps(
        v,
        ensure_ascii=False,
        allow_nan=False
    ).encode("utf-8")
)


def safe_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def get_air_quality_data():
    """
    API Open-Meteo Air Quality :
    - AQI
    - PM2.5
    - PM10
    - NO2
    - O3
    """

    url = (
        "https://air-quality-api.open-meteo.com/v1/air-quality"
        f"?latitude={LATITUDE}"
        f"&longitude={LONGITUDE}"
        f"&timezone={TIMEZONE}"
        "&current=us_aqi,pm10,pm2_5,nitrogen_dioxide,ozone"
    )

    response = requests.get(url, timeout=20)
    response.raise_for_status()

    data = response.json()
    current = data.get("current", {})

    air_quality_event = {
        "event_type": "air_quality_data",

        "air_quality_api_time": current.get("time"),
        "ingestion_time": datetime.now().isoformat(timespec="seconds"),

        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "timezone": TIMEZONE,

        "aqi": safe_float(current.get("us_aqi")),
        "pm25": safe_float(current.get("pm2_5")),
        "pm10": safe_float(current.get("pm10")),
        "no2": safe_float(current.get("nitrogen_dioxide")),
        "o3": safe_float(current.get("ozone")),

        "source": "open_meteo_air_quality_api",
    }

    return air_quality_event


def main():
    print("Air quality producer started.")
    print(f"Topic Kafka : {AIR_QUALITY_TOPIC}")
    print(f"Refresh every {AIR_QUALITY_REFRESH_SECONDS} seconds.")

    while True:
        try:
            air_quality_event = get_air_quality_data()

            future = producer.send(
                AIR_QUALITY_TOPIC,
                value=air_quality_event
            )

            metadata = future.get(timeout=10)

            print(
                f"Air quality sent | "
                f"Topic={metadata.topic} | "
                f"Partition={metadata.partition} | "
                f"Offset={metadata.offset} | "
                f"AQI={air_quality_event.get('aqi')} | "
                f"PM2.5={air_quality_event.get('pm25')} | "
                f"PM10={air_quality_event.get('pm10')} | "
                f"NO2={air_quality_event.get('no2')} | "
                f"O3={air_quality_event.get('o3')} | "
                f"API Time={air_quality_event.get('air_quality_api_time')}"
            )

            producer.flush()

        except Exception as e:
            print(f"Erreur producer_air_quality : {e}")

        time.sleep(AIR_QUALITY_REFRESH_SECONDS)


if __name__ == "__main__":
    main()