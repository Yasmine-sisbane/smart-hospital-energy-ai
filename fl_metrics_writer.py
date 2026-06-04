"""
Consommateur optionnel pour stocker les metriques Federated Learning dans TimescaleDB.
"""

from __future__ import annotations

import json
import time

import psycopg2
from kafka import KafkaConsumer

from fl_config import (
    DB_HOST,
    DB_NAME,
    DB_PASSWORD,
    DB_PORT,
    DB_USER,
    FL_TRAINING_METRICS_TOPIC,
    KAFKA_BOOTSTRAP_SERVERS,
)


def get_connection():
    while True:
        try:
            conn = psycopg2.connect(
                host=DB_HOST,
                port=DB_PORT,
                dbname=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD,
            )
            conn.autocommit = True
            return conn
        except Exception as e:
            print(f"Connexion DB impossible, retry 3s: {e}")
            time.sleep(3)


def create_table(conn):
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS federated_training_metrics (
                ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                round_number INTEGER,
                client_id TEXT,
                horizon TEXT,
                loss DOUBLE PRECISION,
                mae DOUBLE PRECISION,
                rmse DOUBLE PRECISION,
                r2 DOUBLE PRECISION,
                training_time_seconds DOUBLE PRECISION,
                n_samples INTEGER,
                payload JSONB
            );
            """
        )
        cur.execute(
            """
            SELECT create_hypertable(
                'federated_training_metrics',
                'ts',
                if_not_exists => TRUE
            );
            """
        )


def main():
    conn = get_connection()
    create_table(conn)

    consumer = KafkaConsumer(
        FL_TRAINING_METRICS_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        api_version=(0, 10, 1),
        group_id="fl-metrics-writer",
        auto_offset_reset="latest",
        enable_auto_commit=True,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    )

    print(f"[FL_METRICS_WRITER] Listening {FL_TRAINING_METRICS_TOPIC}")

    for message in consumer:
        event = message.value
        if event.get("event_type") != "fl_training_metrics":
            continue

        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO federated_training_metrics (
                    round_number, client_id, horizon, loss, mae, rmse, r2,
                    training_time_seconds, n_samples, payload
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb);
                """,
                (
                    event.get("round_number"),
                    event.get("client_id"),
                    event.get("horizon"),
                    event.get("loss"),
                    event.get("mae"),
                    event.get("rmse"),
                    event.get("r2"),
                    event.get("training_time_seconds"),
                    event.get("n_samples"),
                    json.dumps(event, ensure_ascii=False),
                ),
            )

        print(
            f"[FL_METRICS_WRITER] Insert | Round={event.get('round_number')} | "
            f"Client={event.get('client_id')} | MAE={event.get('mae')}"
        )


if __name__ == "__main__":
    main()
