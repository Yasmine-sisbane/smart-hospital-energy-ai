# =========================
# enrichment_service.py
# Service d'enrichissement temps réel
# =========================

import json
from datetime import datetime

from kafka import KafkaConsumer, KafkaProducer

from config import (
    KAFKA_BOOTSTRAP_SERVERS,
    HOSPITAL_RAW_TOPIC,
    WEATHER_TOPIC,
    AIR_QUALITY_TOPIC,
    HOSPITAL_ENRICHED_TOPIC,
)


consumer = KafkaConsumer(
    HOSPITAL_RAW_TOPIC,
    WEATHER_TOPIC,
    AIR_QUALITY_TOPIC,
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
    api_version=(0, 10, 1),
    group_id="hospital-enrichment-service",
    auto_offset_reset="earliest",
    enable_auto_commit=True,
    value_deserializer=lambda m: json.loads(m.decode("utf-8")),
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


latest_weather = {}
latest_air_quality = {}


def extract_weather_context(event):
    return {
        "weather_api_time": event.get("weather_api_time"),
        "outside_temp_c": event.get("outside_temp_c"),
        "outside_humidity_pct": event.get("outside_humidity_pct"),
        "outside_pressure_hpa": event.get("outside_pressure_hpa"),
        "wind_speed_mps": event.get("wind_speed_mps"),
        "solar_radiation_wm2": event.get("solar_radiation_wm2"),
        "direct_radiation_wm2": event.get("direct_radiation_wm2"),
        "diffuse_radiation_wm2": event.get("diffuse_radiation_wm2"),
        "uv_index": event.get("uv_index"),
        "sunshine_duration_min": event.get("sunshine_duration_min"),
        "weather_source": event.get("source"),
    }


def extract_air_quality_context(event):
    return {
        "air_quality_api_time": event.get("air_quality_api_time"),
        "aqi": event.get("aqi"),
        "pm25": event.get("pm25"),
        "pm10": event.get("pm10"),
        "no2": event.get("no2"),
        "o3": event.get("o3"),
        "air_quality_source": event.get("source"),
    }


def enrich_hospital_event(hospital_event):
    enriched_event = hospital_event.copy()

    # Ajouter dernière météo connue
    enriched_event.update(latest_weather)

    # Ajouter dernière qualité de l'air connue
    enriched_event.update(latest_air_quality)

    # Métadonnées d'enrichissement
    enriched_event["event_type"] = "hospital_enriched_data"
    enriched_event["enrichment_time"] = datetime.now().isoformat(timespec="seconds")
    enriched_event["has_weather_context"] = bool(latest_weather)
    enriched_event["has_air_quality_context"] = bool(latest_air_quality)

    return enriched_event


def main():
    global latest_weather, latest_air_quality

    print("Enrichment service started.")
    print(f"Input topics : {HOSPITAL_RAW_TOPIC}, {WEATHER_TOPIC}, {AIR_QUALITY_TOPIC}")
    print(f"Output topic : {HOSPITAL_ENRICHED_TOPIC}")

    for message in consumer:
        try:
            topic = message.topic
            event = message.value

            if topic == WEATHER_TOPIC:
                latest_weather = extract_weather_context(event)

                print(
                    f"Weather context updated | "
                    f"Temp={latest_weather.get('outside_temp_c')}°C | "
                    f"Solar={latest_weather.get('solar_radiation_wm2')} W/m² | "
                    f"UV={latest_weather.get('uv_index')}"
                )

            elif topic == AIR_QUALITY_TOPIC:
                latest_air_quality = extract_air_quality_context(event)

                print(
                    f"Air quality context updated | "
                    f"AQI={latest_air_quality.get('aqi')} | "
                    f"PM2.5={latest_air_quality.get('pm25')} | "
                    f"PM10={latest_air_quality.get('pm10')}"
                )

            elif topic == HOSPITAL_RAW_TOPIC:
                enriched_event = enrich_hospital_event(event)

                future = producer.send(
                    HOSPITAL_ENRICHED_TOPIC,
                    value=enriched_event
                )

                metadata = future.get(timeout=10)

                print(
                    f"Enriched sent | "
                    f"Topic={metadata.topic} | "
                    f"Partition={metadata.partition} | "
                    f"Offset={metadata.offset} | "
                    f"Zone={enriched_event.get('zone_name')} | "
                    f"Energy={enriched_event.get('zone_energy_kwh')} kWh | "
                    f"Weather={enriched_event.get('has_weather_context')} | "
                    f"Air={enriched_event.get('has_air_quality_context')}"
                )

                producer.flush()

        except Exception as e:
            print(f"Erreur enrichment_service : {e}")


if __name__ == "__main__":
    main()