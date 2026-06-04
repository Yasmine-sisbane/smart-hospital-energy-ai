# =========================
# timescale_writer.py
# Écriture des topics Kafka vers TimescaleDB
# =========================

import json
import time
from datetime import datetime, timezone

import psycopg2
from psycopg2.extras import Json
from kafka import KafkaConsumer

from config import (
    KAFKA_BOOTSTRAP_SERVERS,
    HOSPITAL_ENRICHED_TOPIC,
    METRICS_TOPIC,
    ANOMALIES_TOPIC,
    ALERTS_TOPIC,
    TIMESCALE_HOST,
    TIMESCALE_PORT,
    TIMESCALE_DB,
    TIMESCALE_USER,
    TIMESCALE_PASSWORD,
)


TOPICS = [
    HOSPITAL_ENRICHED_TOPIC,
    METRICS_TOPIC,
    ANOMALIES_TOPIC,
    ALERTS_TOPIC,
]


def get_connection():
    while True:
        try:
            conn = psycopg2.connect(
                host=TIMESCALE_HOST,
                port=TIMESCALE_PORT,
                dbname=TIMESCALE_DB,
                user=TIMESCALE_USER,
                password=TIMESCALE_PASSWORD,
            )
            conn.autocommit = True
            return conn
        except Exception as e:
            print(f"Connexion TimescaleDB impossible, nouvel essai dans 3s : {e}")
            time.sleep(3)


def create_tables(conn):
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")

        cur.execute("""
            CREATE TABLE IF NOT EXISTS hospital_enriched_events (
                ts TIMESTAMPTZ NOT NULL,
                event_time TEXT,
                zone_name TEXT,
                service_type TEXT,
                zone_energy_kwh DOUBLE PRECISION,
                temp_zone_c DOUBLE PRECISION,
                occupancy_count INTEGER,
                payload JSONB
            );
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS hospital_metrics (
                ts TIMESTAMPTZ NOT NULL,
                event_time TEXT,
                zone_name TEXT,
                service_type TEXT,
                zone_energy_kwh DOUBLE PRECISION,
                avg_energy_last_10_events DOUBLE PRECISION,
                energy_per_occupant DOUBLE PRECISION,
                risk_score DOUBLE PRECISION,
                status TEXT,
                annualized_kwh_m2 DOUBLE PRECISION,
                payload JSONB
            );
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS hospital_anomalies (
                ts TIMESTAMPTZ NOT NULL,
                event_time TEXT,
                zone_name TEXT,
                service_type TEXT,
                anomaly_type TEXT,
                severity TEXT,
                value DOUBLE PRECISION,
                message TEXT,
                payload JSONB
            );
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS hospital_alerts (
                ts TIMESTAMPTZ NOT NULL,
                event_time TEXT,
                zone_name TEXT,
                service_type TEXT,
                anomaly_type TEXT,
                alert_level TEXT,
                message TEXT,
                recommended_action TEXT,
                payload JSONB
            );
        """)

        cur.execute("""
            SELECT create_hypertable(
                'hospital_enriched_events',
                'ts',
                if_not_exists => TRUE
            );
        """)

        cur.execute("""
            SELECT create_hypertable(
                'hospital_metrics',
                'ts',
                if_not_exists => TRUE
            );
        """)

        cur.execute("""
            SELECT create_hypertable(
                'hospital_anomalies',
                'ts',
                if_not_exists => TRUE
            );
        """)

        cur.execute("""
            SELECT create_hypertable(
                'hospital_alerts',
                'ts',
                if_not_exists => TRUE
            );
        """)


def safe_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value):
    try:
        if value is None:
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def now_utc():
    return datetime.now(timezone.utc)


def insert_enriched(conn, event):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO hospital_enriched_events (
                ts, event_time, zone_name, service_type,
                zone_energy_kwh, temp_zone_c, occupancy_count, payload
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
        """, (
            now_utc(),
            event.get("event_time"),
            event.get("zone_name"),
            event.get("service_type"),
            safe_float(event.get("zone_energy_kwh")),
            safe_float(event.get("temp_zone_c")),
            safe_int(event.get("occupancy_count")),
            Json(event),
        ))


def insert_metrics(conn, event):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO hospital_metrics (
                ts, event_time, zone_name, service_type,
                zone_energy_kwh, avg_energy_last_10_events,
                energy_per_occupant, risk_score, status,
                annualized_kwh_m2, payload
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
        """, (
            now_utc(),
            event.get("event_time"),
            event.get("zone_name"),
            event.get("service_type"),
            safe_float(event.get("zone_energy_kwh")),
            safe_float(event.get("avg_energy_last_10_events")),
            safe_float(event.get("energy_per_occupant")),
            safe_float(event.get("risk_score")),
            event.get("status"),
            safe_float(event.get("annualized_kwh_m2")),
            Json(event),
        ))


def insert_anomaly(conn, event):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO hospital_anomalies (
                ts, event_time, zone_name, service_type,
                anomaly_type, severity, value, message, payload
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
        """, (
            now_utc(),
            event.get("event_time"),
            event.get("zone_name"),
            event.get("service_type"),
            event.get("anomaly_type"),
            event.get("severity"),
            safe_float(event.get("value")),
            event.get("message"),
            Json(event),
        ))


def insert_alert(conn, event):
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO hospital_alerts (
                ts, event_time, zone_name, service_type,
                anomaly_type, alert_level, message,
                recommended_action, payload
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
        """, (
            now_utc(),
            event.get("event_time"),
            event.get("zone_name"),
            event.get("service_type"),
            event.get("anomaly_type"),
            event.get("alert_level"),
            event.get("message"),
            event.get("recommended_action"),
            Json(event),
        ))


def main():
    print("Timescale writer started.")
    print(f"Listening topics: {', '.join(TOPICS)}")

    conn = get_connection()
    create_tables(conn)

    consumer = KafkaConsumer(
        *TOPICS,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        api_version=(0, 10, 1),
        group_id="hospital-timescale-writer",
        auto_offset_reset="latest",
        enable_auto_commit=True,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    )

    for message in consumer:
        try:
            topic = message.topic
            event = message.value

            if topic == HOSPITAL_ENRICHED_TOPIC:
                insert_enriched(conn, event)
                table_name = "hospital_enriched_events"

            elif topic == METRICS_TOPIC:
                insert_metrics(conn, event)
                table_name = "hospital_metrics"

            elif topic == ANOMALIES_TOPIC:
                insert_anomaly(conn, event)
                table_name = "hospital_anomalies"

            elif topic == ALERTS_TOPIC:
                insert_alert(conn, event)
                table_name = "hospital_alerts"

            else:
                continue

            print(
                f"Timescale insert | "
                f"Topic={topic} | "
                f"Table={table_name} | "
                f"Zone={event.get('zone_name')} | "
                f"Offset={message.offset}"
            )

        except Exception as e:
            print(f"Erreur timescale_writer : {e}")


if __name__ == "__main__":
    main()