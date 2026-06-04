# =========================
# producer_weather.py
# Producteur Kafka météo + soleil + UV
# =========================

import json
import time
from datetime import datetime

import requests
from kafka import KafkaProducer

from config import (
    KAFKA_BOOTSTRAP_SERVERS,
    WEATHER_TOPIC,
    LATITUDE,
    LONGITUDE,
    TIMEZONE,
    WEATHER_REFRESH_SECONDS,
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


def get_hourly_value(hourly_data, variable_name, index):
    values = hourly_data.get(variable_name, [])

    if not values or index >= len(values):
        return None

    return safe_float(values[index])


def find_current_hour_index(hourly_times, current_time):
    """
    Trouve l'index horaire correspondant à l'heure actuelle retournée par l'API.
    """
    if not hourly_times:
        return 0

    if current_time in hourly_times:
        return hourly_times.index(current_time)

    if current_time:
        current_hour = current_time[:13] + ":00"
        if current_hour in hourly_times:
            return hourly_times.index(current_hour)

    return 0


def get_weather_data():
    """
    API Open-Meteo :
    - météo actuelle
    - rayonnement solaire horaire
    - indice UV journalier
    """

    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={LATITUDE}"
        f"&longitude={LONGITUDE}"
        f"&timezone={TIMEZONE}"
        "&current=temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m"
        "&hourly=shortwave_radiation,direct_radiation,diffuse_radiation"
        "&daily=uv_index_max,sunshine_duration"
        "&forecast_days=1"
        "&wind_speed_unit=ms"
    )

    response = requests.get(url, timeout=20)
    response.raise_for_status()

    data = response.json()

    current = data.get("current", {})
    hourly = data.get("hourly", {})
    daily = data.get("daily", {})

    current_time = current.get("time")
    hourly_times = hourly.get("time", [])
    hour_index = find_current_hour_index(hourly_times, current_time)

    sunshine_seconds = None
    if daily.get("sunshine_duration"):
        sunshine_seconds = safe_float(daily["sunshine_duration"][0])

    weather_event = {
        "event_type": "weather_data",

        "weather_api_time": current_time,
        "ingestion_time": datetime.now().isoformat(timespec="seconds"),

        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "timezone": TIMEZONE,

        "outside_temp_c": safe_float(current.get("temperature_2m")),
        "outside_humidity_pct": safe_float(current.get("relative_humidity_2m")),
        "outside_pressure_hpa": safe_float(current.get("pressure_msl")),
        "wind_speed_mps": safe_float(current.get("wind_speed_10m")),

        "solar_radiation_wm2": get_hourly_value(
            hourly,
            "shortwave_radiation",
            hour_index
        ),
        "direct_radiation_wm2": get_hourly_value(
            hourly,
            "direct_radiation",
            hour_index
        ),
        "diffuse_radiation_wm2": get_hourly_value(
            hourly,
            "diffuse_radiation",
            hour_index
        ),

        "uv_index": safe_float(daily.get("uv_index_max", [None])[0]),
        "sunshine_duration_min": (
            round(sunshine_seconds / 60.0, 2)
            if sunshine_seconds is not None
            else None
        ),

        "source": "open_meteo_api",
    }

    return weather_event


def main():
    print(f"Weather producer started.")
    print(f"Topic Kafka : {WEATHER_TOPIC}")
    print(f"Refresh every {WEATHER_REFRESH_SECONDS} seconds.")

    while True:
        try:
            weather_event = get_weather_data()

            future = producer.send(WEATHER_TOPIC, value=weather_event)
            metadata = future.get(timeout=10)

            print(
                f"Weather sent | "
                f"Topic={metadata.topic} | "
                f"Partition={metadata.partition} | "
                f"Offset={metadata.offset} | "
                f"Temp={weather_event.get('outside_temp_c')}°C | "
                f"Humidity={weather_event.get('outside_humidity_pct')}% | "
                f"Solar={weather_event.get('solar_radiation_wm2')} W/m² | "
                f"UV={weather_event.get('uv_index')} | "
                f"API Time={weather_event.get('weather_api_time')}"
            )

            producer.flush()

        except Exception as e:
            print(f"Erreur producer_weather : {e}")

        time.sleep(WEATHER_REFRESH_SECONDS)


if __name__ == "__main__":
    main()