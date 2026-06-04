# =========================
# analytics_service.py
# Analyse temps réel : métriques, anomalies et alertes
# =========================

import json
from datetime import datetime
from pathlib import Path
from collections import defaultdict, deque

from kafka import KafkaConsumer, KafkaProducer

from config import (
    KAFKA_BOOTSTRAP_SERVERS,
    HOSPITAL_ENRICHED_TOPIC,
    METRICS_TOPIC,
    ANOMALIES_TOPIC,
    ALERTS_TOPIC,
    HIGH_ENERGY_THRESHOLD,
    CRITICAL_ENERGY_THRESHOLD,
    MIN_INDOOR_TEMP,
    MAX_INDOOR_TEMP,
    MIN_HUMIDITY_PCT,
    MAX_HUMIDITY_PCT,
    BAD_AQI_THRESHOLD,
    HIGH_ENERGY_PER_OCCUPANT_THRESHOLD,
    HIGH_ANNUALIZED_KWH_M2,
    CRITICAL_ANNUALIZED_KWH_M2,
)

# =========================
# Kafka consumer / producer
# =========================

consumer = KafkaConsumer(
    HOSPITAL_ENRICHED_TOPIC,
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
    api_version=(0, 10, 1),
    group_id="hospital-analytics-service",
    auto_offset_reset="latest",
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
    ).encode("utf-8"),
)


# =========================
# Stockage local temporaire
# Plus tard, TimescaleDB remplacera cette partie
# =========================

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

PROCESSED_EVENTS_FILE = OUTPUT_DIR / "processed_events.jsonl"
METRICS_FILE = OUTPUT_DIR / "metrics.jsonl"
ANOMALIES_FILE = OUTPUT_DIR / "anomalies.jsonl"
ALERTS_FILE = OUTPUT_DIR / "alerts.jsonl"


# Fenêtre glissante par zone
# On garde les 10 dernières consommations pour calculer une moyenne mobile.
energy_history_by_zone = defaultdict(lambda: deque(maxlen=10))

ZONE_AREA_M2 = {
    "ICU": 2200,
    "ER": 2000,
    "LAB": 1300,
}


def safe_float(value, default=None):
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value, default=0):
    try:
        if value is None:
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def write_jsonl(file_path, event):
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def normalize_hospital_event(event):
    """
    Sécurise quelques valeurs numériques.
    Le producer_hospital.py corrige déjà l'énergie.
    Ici on protège aussi les températures si jamais elles arrivent sous forme 220 -> 22.0.
    """
    normalized = event.copy()

    scale_10_columns = [
        "temp_zone_c",
        "humidity_zone_pct",
        "setpoint_temp_c",
        "outside_temp_c",
        "outside_humidity_pct",
        "outside_pressure_hpa",
        "wind_speed_mps",
        "solar_radiation_wm2",
        "uv_index",
        "pm25",
        "pm10",
        "no2",
        "o3",
    ]

    for col in scale_10_columns:
        value = safe_float(normalized.get(col), None)

        if value is None:
            continue

        # Correction défensive :
        # temp 220 -> 22.0
        # humidity 534 -> 53.4
        # pressure 10115 -> 1011.5
        if col == "outside_pressure_hpa":
            if value > 2000:
                normalized[col] = round(value / 10.0, 2)
            else:
                normalized[col] = value
        elif value > 100:
            normalized[col] = round(value / 10.0, 2)
        else:
            normalized[col] = value

    return normalized


def compute_metrics(event):
    zone_name = event.get("zone_name", "Unknown")
    area_m2 = ZONE_AREA_M2.get(zone_name, 1000)
    event_time = event.get("event_time")
    ingestion_time = event.get("ingestion_time")

    energy_kwh = safe_float(event.get("zone_energy_kwh"), 0.0)
    energy_kwh_per_m2 = energy_kwh / area_m2 if area_m2 > 0 else None
    annualized_kwh_m2 = (
    energy_kwh_per_m2 * 35040
    if energy_kwh_per_m2 is not None
    else None
)
    temp_zone_c = safe_float(event.get("temp_zone_c"), None)
    occupancy_count = safe_int(event.get("occupancy_count"), 0)
    aqi = safe_float(event.get("aqi"), None)

    energy_history_by_zone[zone_name].append(energy_kwh)
    avg_energy_last_10 = sum(energy_history_by_zone[zone_name]) / len(
        energy_history_by_zone[zone_name]
    )

    if occupancy_count > 0:
        energy_per_occupant = energy_kwh / occupancy_count
    else:
        energy_per_occupant = None

    # Score de risque simple entre 0 et 1
    risk_score = 0.0

    if energy_kwh >= HIGH_ENERGY_THRESHOLD:
        risk_score += 0.35

    if energy_per_occupant is not None and energy_per_occupant >= HIGH_ENERGY_PER_OCCUPANT_THRESHOLD:
        risk_score += 0.25

    if temp_zone_c is not None and (temp_zone_c < MIN_INDOOR_TEMP or temp_zone_c > MAX_INDOOR_TEMP):
        risk_score += 0.20

    if aqi is not None and aqi >= BAD_AQI_THRESHOLD:
        risk_score += 0.20

    risk_score = min(round(risk_score, 2), 1.0)

    if risk_score >= 0.75:
        status = "critical"
    elif risk_score >= 0.45:
        status = "high"
    elif risk_score >= 0.20:
        status = "medium"
    else:
        status = "normal"

    metrics_event = {
        "event_type": "hospital_metrics",
        "event_time": event_time,
        "ingestion_time": ingestion_time,
        "analytics_time": datetime.now().isoformat(timespec="seconds"),

        "hospital_id": event.get("hospital_id"),
        "building_id": event.get("building_id"),
        "floor_id": event.get("floor_id"),
        "zone_id": event.get("zone_id"),
        "zone_name": zone_name,
        "service_type": event.get("service_type"),

        "zone_energy_kwh": round(energy_kwh, 2),
        "avg_energy_last_10_events": round(avg_energy_last_10, 2),
        "occupancy_count": occupancy_count,
        "energy_per_occupant": (
            round(energy_per_occupant, 2)
            if energy_per_occupant is not None
            else None
        ),

        "temp_zone_c": temp_zone_c,
        "outside_temp_c": event.get("outside_temp_c"),
        "aqi": aqi,
        "pm25": event.get("pm25"),
        "pm10": event.get("pm10"),

        "hvac_status": event.get("hvac_status"),
        "hvac_mode": event.get("hvac_mode"),

        "risk_score": risk_score,
        "status": status,
        
        
        
        
        
        
        
        "area_m2": area_m2,
"energy_kwh_per_m2": (
    round(energy_kwh_per_m2, 5)
    if energy_kwh_per_m2 is not None
    else None
),
"annualized_kwh_m2": (
    round(annualized_kwh_m2, 2)
    if annualized_kwh_m2 is not None
    else None
),
    }

    return metrics_event


def detect_anomalies(event, metrics):
    anomalies = []

    zone_name = metrics.get("zone_name")
    event_time = metrics.get("event_time")
    energy_kwh = safe_float(metrics.get("zone_energy_kwh"), 0.0)
    temp_zone_c = safe_float(metrics.get("temp_zone_c"), None)
    occupancy_count = safe_int(metrics.get("occupancy_count"), 0)
    energy_per_occupant = safe_float(metrics.get("energy_per_occupant"), None)
    aqi = safe_float(metrics.get("aqi"), None)

    hvac_status = str(event.get("hvac_status", "")).lower()
    critical_zone_flag = safe_int(event.get("critical_zone_flag"), 0)
    
    humidity_zone_pct = safe_float(event.get("humidity_zone_pct"), None)
    annualized_kwh_m2 = safe_float(metrics.get("annualized_kwh_m2"), None)

    base = {
        "event_time": event_time,
        "analytics_time": datetime.now().isoformat(timespec="seconds"),
        "hospital_id": event.get("hospital_id"),
        "building_id": event.get("building_id"),
        "floor_id": event.get("floor_id"),
        "zone_id": event.get("zone_id"),
        "zone_name": zone_name,
        "service_type": event.get("service_type"),
    }

    # 1. Pic énergétique
    if energy_kwh >= HIGH_ENERGY_THRESHOLD:
        severity = "critical" if energy_kwh >= CRITICAL_ENERGY_THRESHOLD else "high"

        anomalies.append({
            **base,
            "event_type": "hospital_anomaly",
            "anomaly_type": "energy_spike",
            "severity": severity,
            "value": energy_kwh,
            "threshold": HIGH_ENERGY_THRESHOLD,
            "message": f"Pic de consommation détecté dans {zone_name}: {energy_kwh} kWh",
        })

    # 2. Zone critique + consommation élevée
    if critical_zone_flag == 1 and energy_kwh >= HIGH_ENERGY_THRESHOLD:
        anomalies.append({
            **base,
            "event_type": "hospital_anomaly",
            "anomaly_type": "critical_zone_high_energy",
            "severity": "critical",
            "value": energy_kwh,
            "threshold": HIGH_ENERGY_THRESHOLD,
            "message": f"Surconsommation dans une zone critique: {zone_name}",
        })

    # 3. HVAC actif alors que la zone est vide
    if hvac_status in ["on", "active", "1", "true"] and occupancy_count == 0:
        anomalies.append({
            **base,
            "event_type": "hospital_anomaly",
            "anomaly_type": "hvac_on_empty_zone",
            "severity": "medium",
            "value": occupancy_count,
            "threshold": 0,
            "message": f"HVAC actif alors que la zone {zone_name} est vide",
        })

    # 4. Température intérieure hors plage
    if temp_zone_c is not None and (temp_zone_c < MIN_INDOOR_TEMP or temp_zone_c > MAX_INDOOR_TEMP):
        anomalies.append({
            **base,
            "event_type": "hospital_anomaly",
            "anomaly_type": "temperature_out_of_range",
            "severity": "high",
            "value": temp_zone_c,
            "min_threshold": MIN_INDOOR_TEMP,
            "max_threshold": MAX_INDOOR_TEMP,
            "message": f"Température intérieure anormale dans {zone_name}: {temp_zone_c} °C",
        })

    # 5. Qualité de l'air mauvaise
    if aqi is not None and aqi >= BAD_AQI_THRESHOLD:
        severity = "critical" if critical_zone_flag == 1 else "high"

        anomalies.append({
            **base,
            "event_type": "hospital_anomaly",
            "anomaly_type": "bad_air_quality",
            "severity": severity,
            "value": aqi,
            "threshold": BAD_AQI_THRESHOLD,
            "message": f"Qualité de l'air dégradée dans {zone_name}: AQI={aqi}",
        })

    # 6. Consommation par occupant élevée
    if energy_per_occupant is not None and energy_per_occupant >= HIGH_ENERGY_PER_OCCUPANT_THRESHOLD:
        anomalies.append({
            **base,
            "event_type": "hospital_anomaly",
            "anomaly_type": "high_energy_per_occupant",
            "severity": "medium",
            "value": energy_per_occupant,
            "threshold": HIGH_ENERGY_PER_OCCUPANT_THRESHOLD,
            "message": f"Consommation par occupant élevée dans {zone_name}: {energy_per_occupant} kWh/personne",
        })
        
        
      
        
        
    # 7. Intensité énergétique élevée par m²
    if annualized_kwh_m2 is not None and annualized_kwh_m2 >= HIGH_ANNUALIZED_KWH_M2:
        severity = (
            "critical"
            if annualized_kwh_m2 >= CRITICAL_ANNUALIZED_KWH_M2
            else "high"
        )

        anomalies.append({
            **base,
            "event_type": "hospital_anomaly",
            "anomaly_type": "high_energy_intensity",
            "severity": severity,
            "value": annualized_kwh_m2,
            "threshold": HIGH_ANNUALIZED_KWH_M2,
            "message": f"Intensité énergétique élevée dans {zone_name}: {annualized_kwh_m2} kWh/m²/an",
        })

    # 8. Humidité hors plage
    if humidity_zone_pct is not None and (
        humidity_zone_pct < MIN_HUMIDITY_PCT
        or humidity_zone_pct > MAX_HUMIDITY_PCT
    ):
        anomalies.append({
            **base,
            "event_type": "hospital_anomaly",
            "anomaly_type": "humidity_out_of_range",
            "severity": "medium",
            "value": humidity_zone_pct,
            "min_threshold": MIN_HUMIDITY_PCT,
            "max_threshold": MAX_HUMIDITY_PCT,
            "message": f"Humidité hors plage dans {zone_name}: {humidity_zone_pct}%",
        })

    return anomalies

def build_alert(anomaly):
    severity = anomaly.get("severity")

    if severity not in ["high", "critical"]:
        return None

    alert_level = "critical" if severity == "critical" else "high"

    return {
        "event_type": "hospital_alert",
        "event_time": anomaly.get("event_time"),
        "alert_time": datetime.now().isoformat(timespec="seconds"),
        "zone_name": anomaly.get("zone_name"),
        "service_type": anomaly.get("service_type"),
        "anomaly_type": anomaly.get("anomaly_type"),
        "alert_level": alert_level,
        "message": anomaly.get("message"),
        "value": anomaly.get("value"),
        "recommended_action": get_recommended_action(anomaly.get("anomaly_type")),
    }


def get_recommended_action(anomaly_type):
    actions = {
        "energy_spike": "Vérifier les équipements énergivores et l'état HVAC.",
        "critical_zone_high_energy": "Inspecter immédiatement la zone critique et les équipements médicaux.",
        "hvac_on_empty_zone": "Optimiser ou désactiver le HVAC dans la zone vide.",
        "temperature_out_of_range": "Vérifier les consignes HVAC et les capteurs de température.",
        "bad_air_quality": "Contrôler la ventilation et la qualité de l'air.",
        "high_energy_per_occupant": "Analyser la charge énergétique par rapport à l'occupation réelle.",
        "high_energy_intensity": "Comparer la consommation avec la surface de la zone et vérifier HVAC/équipements.",
         "humidity_out_of_range": "Vérifier le contrôle humidité, ventilation et conditions HVAC.",
    }
    

    return actions.get(anomaly_type, "Analyser l'événement et vérifier les capteurs.")


def send_to_kafka(topic, event):
    future = producer.send(topic, value=event)
    metadata = future.get(timeout=10)
    return metadata


def main():
    print("Analytics service started.")
    print(f"Input topic  : {HOSPITAL_ENRICHED_TOPIC}")
    print(f"Metrics topic: {METRICS_TOPIC}")
    print(f"Anomalies    : {ANOMALIES_TOPIC}")
    print(f"Alerts       : {ALERTS_TOPIC}")

    for message in consumer:
        try:
            raw_event = message.value
            event = normalize_hospital_event(raw_event)

            metrics = compute_metrics(event)
            anomalies = detect_anomalies(event, metrics)

            metrics_metadata = send_to_kafka(METRICS_TOPIC, metrics)
            write_jsonl(METRICS_FILE, metrics)
            write_jsonl(PROCESSED_EVENTS_FILE, event)

            alerts_count = 0

            for anomaly in anomalies:
                send_to_kafka(ANOMALIES_TOPIC, anomaly)
                write_jsonl(ANOMALIES_FILE, anomaly)

                alert = build_alert(anomaly)

                if alert is not None:
                    send_to_kafka(ALERTS_TOPIC, alert)
                    write_jsonl(ALERTS_FILE, alert)
                    alerts_count += 1

            producer.flush()

            print(
                f"Analytics processed | "
                f"Zone={metrics.get('zone_name')} | "
                f"Energy={metrics.get('zone_energy_kwh')} kWh | "
                f"Avg10={metrics.get('avg_energy_last_10_events')} | "
                f"Energy/Occupant={metrics.get('energy_per_occupant')} | "
                f"Risk={metrics.get('risk_score')} | "
                f"Status={metrics.get('status')} | "
                f"Anomalies={len(anomalies)} | "
                f"Alerts={alerts_count} | "
                f"MetricsOffset={metrics_metadata.offset}"
            )

        except Exception as e:
            print(f"Erreur analytics_service : {e}")


if __name__ == "__main__":
    main()