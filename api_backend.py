

from __future__ import annotations

import os
from datetime import datetime
from decimal import Decimal
from typing import Any

import psycopg2
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool


# ============================================================
# Configuration TimescaleDB
# ============================================================

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "dbname": os.getenv("DB_NAME", "hospital_iot"),
    "user": os.getenv("DB_USER", "hospital_user"),
    "password": os.getenv("DB_PASSWORD", "hospital_password"),
}


# ============================================================
# FastAPI app
# ============================================================

app = FastAPI(
    title="Smart Hospital EnergyAI API",
    description="API FastAPI pour TimescaleDB, ML XGBoost et Federated Learning.",
    version="1.0.0",
)

# Autoriser React local
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
    "http://localhost:3000",
    "http://127.0.0.1:3000",

    "http://localhost:5173",
    "http://127.0.0.1:5173",

    "http://localhost:5174",
    "http://127.0.0.1:5174",

    "http://localhost:5175",
    "http://127.0.0.1:5175",
],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




try:
    db_pool = SimpleConnectionPool(
        minconn=1,
        maxconn=10,
        **DB_CONFIG,
    )
except Exception as exc:
    db_pool = None
    print(f"[API] Erreur connexion TimescaleDB au demarrage : {exc}")


def serialize_value(value: Any):
    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, Decimal):
        return float(value)

    return value


def serialize_row(row: dict):
    return {
        key: serialize_value(value)
        for key, value in dict(row).items()
    }


def fetch_all(sql: str, params: tuple = ()):
    if db_pool is None:
        raise HTTPException(
            status_code=500,
            detail="Connexion TimescaleDB indisponible.",
        )

    conn = db_pool.getconn()

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
            return [serialize_row(row) for row in rows]

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur SQL : {exc}",
        )

    finally:
        db_pool.putconn(conn)


def fetch_one(sql: str, params: tuple = ()):
    rows = fetch_all(sql, params)
    return rows[0] if rows else None




@app.get("/")
def root():
    return {
        "message": "Smart Hospital EnergyAI API is running",
        "docs": "/docs",
        "database": "TimescaleDB",
    }


@app.get("/api/health")
def health():
    result = fetch_one("SELECT NOW() AS db_time;")

    return {
        "status": "ok",
        "database": "connected",
        "db_time": result["db_time"] if result else None,
    }




@app.get("/api/kpis")
def get_kpis():
    sql = """
        SELECT
            (SELECT COUNT(*) FROM hospital_enriched_events) AS enriched_events,
            (SELECT COUNT(*) FROM hospital_metrics) AS metrics,
            (SELECT COUNT(*) FROM hospital_anomalies) AS anomalies,
            (SELECT COUNT(*) FROM hospital_alerts) AS alerts,
            (SELECT COUNT(*) FROM energy_predictions) AS energy_predictions,
            (SELECT COUNT(*) FROM federated_training_metrics) AS fl_training_metrics,
            (SELECT COUNT(*) FROM federated_model_comparison) AS fl_model_comparisons,
            (SELECT COUNT(*) FROM federated_personalized_models) AS fl_personalized_models;
    """

    return fetch_one(sql)


# ============================================================
# Données énergétiques
# ============================================================

@app.get("/api/energy/timeseries")
def get_energy_timeseries(
    zone: str | None = Query(default=None, description="ICU, ER ou LAB"),
    limit: int = Query(default=500, ge=1, le=5000),
):
    sql = """
        SELECT
            ts,
            event_time,
            zone_name,
            service_type,
            zone_energy_kwh,
            temp_zone_c,
            occupancy_count
        FROM hospital_enriched_events
        WHERE (%s IS NULL OR zone_name = %s)
        ORDER BY ts DESC
        LIMIT %s;
    """

    return fetch_all(sql, (zone, zone, limit))


@app.get("/api/energy/by-zone")
def get_energy_by_zone():
    sql = """
        SELECT
            zone_name,
            COUNT(*) AS points,
            ROUND(AVG(zone_energy_kwh)::numeric, 4) AS avg_energy_kwh,
            ROUND(MIN(zone_energy_kwh)::numeric, 4) AS min_energy_kwh,
            ROUND(MAX(zone_energy_kwh)::numeric, 4) AS max_energy_kwh
        FROM hospital_enriched_events
        WHERE zone_energy_kwh IS NOT NULL
        GROUP BY zone_name
        ORDER BY zone_name;
    """

    return fetch_all(sql)


# ============================================================
# Prédictions XGBoost
# ============================================================

@app.get("/api/predictions/latest")
def get_latest_predictions(
    zone: str | None = Query(default=None, description="ICU, ER ou LAB"),
    horizon: str | None = Query(default=None, description="T+15min, T+1h ou T+24h"),
    limit: int = Query(default=100, ge=1, le=1000),
):
    sql = """
        SELECT
            event_time,
            target_time,
            prediction_time,
            zone_name,
            horizon,
            energy_current,
            predicted_energy_kwh,
            actual_energy_kwh,
            residual_energy_kwh,
            abs_error_kwh,
            anomaly_from_prediction,
            anomaly_severity,
            recommended_action,
            model_name
        FROM energy_predictions
        WHERE (%s IS NULL OR zone_name = %s)
          AND (%s IS NULL OR horizon = %s)
        ORDER BY target_time DESC, event_time DESC, prediction_time DESC
        LIMIT %s;
    """

    return fetch_all(sql, (zone, zone, horizon, horizon, limit))


@app.get("/api/predictions/summary")
def get_prediction_summary():
    sql = """
        SELECT
            horizon,
            COUNT(*) AS total_predictions,
            ROUND(AVG(abs_error_kwh)::numeric, 4) AS avg_abs_error_kwh,
            ROUND(AVG(predicted_energy_kwh)::numeric, 4) AS avg_predicted_energy_kwh,
            ROUND(AVG(actual_energy_kwh)::numeric, 4) AS avg_actual_energy_kwh,
            SUM(CASE WHEN anomaly_from_prediction THEN 1 ELSE 0 END) AS predicted_anomalies
        FROM energy_predictions
        GROUP BY horizon
        ORDER BY horizon;
    """

    return fetch_all(sql)


@app.get("/api/predictions/by-zone")
def get_prediction_by_zone():
    sql = """
        SELECT
            zone_name,
            horizon,
            COUNT(*) AS total_predictions,
            ROUND(AVG(abs_error_kwh)::numeric, 4) AS avg_abs_error_kwh,
            ROUND(AVG(predicted_energy_kwh)::numeric, 4) AS avg_predicted_energy_kwh,
            ROUND(AVG(actual_energy_kwh)::numeric, 4) AS avg_actual_energy_kwh
        FROM energy_predictions
        GROUP BY zone_name, horizon
        ORDER BY zone_name, horizon;
    """

    return fetch_all(sql)


# ============================================================
# Anomalies et alertes
# ============================================================

@app.get("/api/anomalies/latest")
def get_latest_anomalies(
    limit: int = Query(default=100, ge=1, le=1000),
):
    sql = """
        SELECT
            ts,
            event_time,
            zone_name,
            service_type,
            anomaly_type,
            severity,
            value,
            message
        FROM hospital_anomalies
        ORDER BY ts DESC
        LIMIT %s;
    """

    return fetch_all(sql, (limit,))


@app.get("/api/alerts/latest")
def get_latest_alerts(
    limit: int = Query(default=100, ge=1, le=1000),
):
    sql = """
        SELECT
            ts,
            event_time,
            zone_name,
            service_type,
            anomaly_type,
            alert_level,
            message,
            recommended_action
        FROM hospital_alerts
        ORDER BY ts DESC
        LIMIT %s;
    """

    return fetch_all(sql, (limit,))


@app.get("/api/anomalies/summary")
def get_anomalies_summary():
    sql = """
        SELECT
            severity,
            anomaly_type,
            COUNT(*) AS total
        FROM hospital_anomalies
        GROUP BY severity, anomaly_type
        ORDER BY total DESC;
    """

    return fetch_all(sql)


# ============================================================
# Federated Learning
# ============================================================

@app.get("/api/federated/comparison")
def get_federated_comparison():
    sql = """
        WITH latest_run AS (
            SELECT run_id
            FROM federated_model_comparison
            ORDER BY ts DESC
            LIMIT 1
        )
        SELECT
            approach,
            data_centralized,
            mae,
            rmse,
            r2,
            interpretation
        FROM federated_model_comparison
        WHERE run_id = (SELECT run_id FROM latest_run)
        ORDER BY
            CASE
                WHEN approach = 'Centralized MLP' THEN 1
                WHEN approach = 'FedAvg global model' THEN 2
                WHEN approach = 'Personalized Federated Learning' THEN 3
                ELSE 4
            END;
    """

    return fetch_all(sql)


@app.get("/api/federated/personalized-models")
def get_federated_personalized_models():
    sql = """
        WITH latest_run AS (
            SELECT run_id
            FROM federated_personalized_models
            ORDER BY ts DESC
            LIMIT 1
        )
        SELECT
            client_id,
            horizon,
            mae,
            rmse,
            r2,
            model_path,
            metadata_path,
            data_centralized
        FROM federated_personalized_models
        WHERE run_id = (SELECT run_id FROM latest_run)
        ORDER BY
            CASE
                WHEN client_id = 'ICU' THEN 1
                WHEN client_id = 'ER' THEN 2
                WHEN client_id = 'LAB' THEN 3
                WHEN client_id = 'AVERAGE' THEN 4
                ELSE 5
            END;
    """

    return fetch_all(sql)


@app.get("/api/federated/round-metrics")
def get_federated_round_metrics(
    client_id: str | None = Query(default=None, description="ICU, ER ou LAB"),
    limit: int = Query(default=300, ge=1, le=5000),
):
    sql = """
        SELECT
            ts,
            round_number,
            client_id,
            horizon,
            loss,
            mae,
            rmse,
            r2,
            training_time_seconds,
            n_samples
        FROM federated_training_metrics
        WHERE (%s IS NULL OR client_id = %s)
        ORDER BY ts DESC
        LIMIT %s;
    """

    return fetch_all(sql, (client_id, client_id, limit))


@app.get("/api/federated/round-summary")
def get_federated_round_summary():
    sql = """
        SELECT
            round_number,
            ROUND(AVG(mae)::numeric, 4) AS avg_mae,
            ROUND(AVG(rmse)::numeric, 4) AS avg_rmse,
            ROUND(AVG(r2)::numeric, 4) AS avg_r2,
            COUNT(*) AS total_client_updates
        FROM federated_training_metrics
        WHERE round_number IS NOT NULL
        GROUP BY round_number
        ORDER BY round_number;
    """

    return fetch_all(sql)

# ============================================================
# Frontend clean separation layer
# These endpoints keep React away from mixed public.* tables.
# Realtime = Kafka replay hospital_* tables
# ML = energy_predictions
# Federated = federated_* tables
# ============================================================


def table_exists(table_name: str) -> bool:
    row = fetch_one(
        """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_name = %s
        ) AS exists;
        """,
        (table_name,),
    )
    return bool(row and row.get("exists"))


def safe_fetch_all(sql: str, params: tuple = ()):  # never crashes the dashboard
    try:
        return fetch_all(sql, params)
    except HTTPException as exc:
        return [{"error": exc.detail}]


def safe_fetch_one(sql: str, params: tuple = ()):  # never crashes the dashboard
    rows = safe_fetch_all(sql, params)
    if rows and "error" not in rows[0]:
        return rows[0]
    return {}


@app.get("/api/system/catalog")
def get_system_catalog():
    """Tables grouped by module so the frontend does not mix pipelines."""
    sql = """
        SELECT
            relname AS table_name,
            CASE
                WHEN relname LIKE 'hospital_%' THEN 'realtime_kafka'
                WHEN relname LIKE 'energy_predictions%' THEN 'machine_learning'
                WHEN relname LIKE 'federated_%' THEN 'federated_learning'
                ELSE 'other'
            END AS module,
            COALESCE(n_live_tup, 0) AS estimated_rows,
            pg_size_pretty(pg_total_relation_size(pg_statio_user_tables.relid)) AS total_size
        FROM pg_catalog.pg_statio_user_tables
        LEFT JOIN pg_catalog.pg_stat_user_tables
            ON pg_statio_user_tables.relid = pg_stat_user_tables.relid
        ORDER BY module, table_name;
    """
    return fetch_all(sql)


@app.get("/api/system/date-ranges")
def get_date_ranges():
    """Montre la différence entre inserted_at et original_event_time par pipeline."""
    rows = []

    rows.extend(safe_fetch_all("""
        SELECT
            'realtime_metrics' AS table_name,
            MIN(inserted_at) AS min_inserted_at,
            MAX(inserted_at) AS max_inserted_at,
            MIN(original_event_ts) AS min_original_event_time,
            MAX(original_event_ts) AS max_original_event_time,
            COUNT(*) AS rows
        FROM frontend.realtime_metrics;
    """))

    rows.extend(safe_fetch_all("""
        SELECT
            'realtime_enriched_events' AS table_name,
            MIN(inserted_at) AS min_inserted_at,
            MAX(inserted_at) AS max_inserted_at,
            MIN(original_event_ts) AS min_original_event_time,
            MAX(original_event_ts) AS max_original_event_time,
            COUNT(*) AS rows
        FROM frontend.realtime_enriched_events;
    """))

    rows.extend(safe_fetch_all("""
        SELECT
            'realtime_anomalies' AS table_name,
            MIN(inserted_at) AS min_inserted_at,
            MAX(inserted_at) AS max_inserted_at,
            MIN(original_event_ts) AS min_original_event_time,
            MAX(original_event_ts) AS max_original_event_time,
            COUNT(*) AS rows
        FROM frontend.realtime_anomalies;
    """))

    rows.extend(safe_fetch_all("""
        SELECT
            'realtime_alerts' AS table_name,
            MIN(inserted_at) AS min_inserted_at,
            MAX(inserted_at) AS max_inserted_at,
            MIN(original_event_ts) AS min_original_event_time,
            MAX(original_event_ts) AS max_original_event_time,
            COUNT(*) AS rows
        FROM frontend.realtime_alerts;
    """))

    if table_exists("energy_predictions"):
        rows.extend(safe_fetch_all("""
            SELECT
                'energy_predictions' AS table_name,
                MIN(prediction_time) AS min_inserted_at,
                MAX(prediction_time) AS max_inserted_at,
                MIN(event_time) AS min_original_event_time,
                MAX(event_time) AS max_original_event_time,
                COUNT(*) AS rows
            FROM energy_predictions;
        """))

    if table_exists("federated_training_metrics"):
        rows.extend(safe_fetch_all("""
            SELECT
                'federated_training_metrics' AS table_name,
                MIN(ts) AS min_inserted_at,
                MAX(ts) AS max_inserted_at,
                NULL::timestamp AS min_original_event_time,
                NULL::timestamp AS max_original_event_time,
                COUNT(*) AS rows
            FROM federated_training_metrics;
        """))

    return rows


@app.get("/api/realtime/kpis")
def get_realtime_kpis():
    sql = """
        SELECT
            (SELECT COUNT(*) FROM frontend.realtime_enriched_events) AS enriched_events,
            (SELECT COUNT(*) FROM frontend.realtime_metrics) AS metrics,
            (SELECT COUNT(*) FROM frontend.realtime_anomalies) AS anomalies,
            (SELECT COUNT(*) FROM frontend.realtime_alerts) AS alerts,
            (SELECT COUNT(*) FROM frontend.realtime_anomalies WHERE severity = 'critical') AS critical_anomalies,
            (SELECT ROUND(AVG(zone_energy_kwh)::numeric, 3) FROM frontend.realtime_metrics) AS avg_energy_kwh,
            (SELECT ROUND(MAX(zone_energy_kwh)::numeric, 3) FROM frontend.realtime_metrics) AS max_energy_kwh,
            (SELECT MAX(inserted_at) FROM frontend.realtime_metrics) AS latest_inserted_at,
            (SELECT MAX(original_event_ts) FROM frontend.realtime_metrics) AS latest_original_event_time;
    """
    result = safe_fetch_one(sql)
    result["pipeline"] = "kafka_realtime_replay"
    result["source_dataset"] = "hospital_sensor_raw_180days_51840"
    return result


@app.get("/api/realtime/metrics")
def get_realtime_metrics_clean(
    zone: str | None = Query(default=None, description="ICU, ER ou LAB"),
    limit: int = Query(default=300, ge=1, le=5000),
):
    sql = """
        SELECT
            inserted_at,
            original_event_time,
            original_event_ts,
            pipeline,
            source_dataset,
            zone_name,
            service_type,
            zone_energy_kwh,
            avg_energy_last_10_events,
            energy_per_occupant,
            risk_score,
            status,
            annualized_kwh_m2,
            payload
        FROM frontend.realtime_metrics
        WHERE (%s IS NULL OR zone_name = %s)
        ORDER BY original_event_ts DESC, inserted_at DESC
        LIMIT %s;
    """
    return fetch_all(sql, (zone, zone, limit))


@app.get("/api/realtime/enriched-events")
def get_realtime_enriched_events_clean(
    zone: str | None = Query(default=None, description="ICU, ER ou LAB"),
    limit: int = Query(default=300, ge=1, le=5000),
):
    sql = """
        SELECT
            inserted_at,
            original_event_time,
            original_event_ts,
            pipeline,
            source_dataset,
            zone_name,
            service_type,
            zone_energy_kwh,
            temp_zone_c,
            occupancy_count,
            payload
        FROM frontend.realtime_enriched_events
        WHERE (%s IS NULL OR zone_name = %s)
        ORDER BY original_event_ts DESC, inserted_at DESC
        LIMIT %s;
    """
    return fetch_all(sql, (zone, zone, limit))


@app.get("/api/realtime/by-zone")
def get_realtime_by_zone():
    sql = """
        SELECT
            zone_name,
            COUNT(*) AS points,
            ROUND(AVG(zone_energy_kwh)::numeric, 3) AS avg_energy_kwh,
            ROUND(MIN(zone_energy_kwh)::numeric, 3) AS min_energy_kwh,
            ROUND(MAX(zone_energy_kwh)::numeric, 3) AS max_energy_kwh,
            MAX(inserted_at) AS latest_inserted_at,
            MAX(original_event_ts) AS latest_original_event_time
        FROM frontend.realtime_metrics
        WHERE zone_energy_kwh IS NOT NULL
        GROUP BY zone_name
        ORDER BY zone_name;
    """
    return fetch_all(sql)


@app.get("/api/realtime/anomalies")
def get_realtime_anomalies_clean(
    limit: int = Query(default=100, ge=1, le=1000),
    dedupe: bool = Query(default=False, description="True = une dernière anomalie par zone/type/severity"),
):
    if dedupe:
        sql = """
            SELECT *
            FROM (
                SELECT DISTINCT ON (zone_name, anomaly_type, severity)
                    inserted_at,
                    original_event_time,
                    original_event_ts,
                    pipeline,
                    source_dataset,
                    zone_name,
                    service_type,
                    anomaly_type,
                    severity,
                    value,
                    message,
                    payload
                FROM frontend.realtime_anomalies
                ORDER BY zone_name, anomaly_type, severity, original_event_ts DESC, inserted_at DESC
            ) AS latest_unique_anomalies
            ORDER BY original_event_ts DESC, inserted_at DESC
            LIMIT %s;
        """
    else:
        sql = """
            SELECT
                inserted_at,
                original_event_time,
                original_event_ts,
                pipeline,
                source_dataset,
                zone_name,
                service_type,
                anomaly_type,
                severity,
                value,
                message,
                payload
            FROM frontend.realtime_anomalies
            ORDER BY original_event_ts DESC, inserted_at DESC
            LIMIT %s;
        """

    return fetch_all(sql, (limit,))


@app.get("/api/realtime/alerts")
def get_realtime_alerts_clean(limit: int = Query(default=100, ge=1, le=1000)):
    sql = """
        SELECT
            inserted_at,
            original_event_time,
            original_event_ts,
            pipeline,
            source_dataset,
            zone_name,
            service_type,
            anomaly_type,
            alert_level,
            message,
            recommended_action,
            payload
        FROM frontend.realtime_alerts
        ORDER BY original_event_ts DESC, inserted_at DESC
        LIMIT %s;
    """
    return fetch_all(sql, (limit,))


@app.get("/api/realtime/stream")
def get_realtime_stream(limit: int = Query(default=50, ge=1, le=500)):
    sql = """
        SELECT * FROM (
            SELECT
                inserted_at,
                original_event_time,
                original_event_ts,
                'hospital_enriched_data' AS topic,
                CONCAT('Capteur ', COALESCE(zone_name, 'Unknown')) AS source,
                'Donnée enrichie' AS event_type,
                CONCAT(ROUND(zone_energy_kwh::numeric, 2), ' kWh') AS value,
                'normal' AS status
            FROM frontend.realtime_enriched_events

            UNION ALL

            SELECT
                inserted_at,
                original_event_time,
                original_event_ts,
                'hospital_metrics' AS topic,
                CONCAT('Analytics ', COALESCE(zone_name, 'Unknown')) AS source,
                'Métriques' AS event_type,
                CONCAT('Risk=', COALESCE(risk_score::text, 'n/a')) AS value,
                COALESCE(status, 'normal') AS status
            FROM frontend.realtime_metrics

            UNION ALL

            SELECT
                inserted_at,
                original_event_time,
                original_event_ts,
                'hospital_anomalies' AS topic,
                CONCAT('Anomalie ', COALESCE(zone_name, 'Unknown')) AS source,
                COALESCE(anomaly_type, 'Anomalie') AS event_type,
                COALESCE(value::text, 'n/a') AS value,
                COALESCE(severity, 'unknown') AS status
            FROM frontend.realtime_anomalies

            UNION ALL

            SELECT
                inserted_at,
                original_event_time,
                original_event_ts,
                'hospital_alerts' AS topic,
                CONCAT('Alerte ', COALESCE(zone_name, 'Unknown')) AS source,
                COALESCE(anomaly_type, 'Alerte') AS event_type,
                COALESCE(alert_level, 'n/a') AS value,
                COALESCE(alert_level, 'unknown') AS status
            FROM frontend.realtime_alerts
        ) AS stream_events
        ORDER BY original_event_ts DESC, inserted_at DESC
        LIMIT %s;
    """
    return fetch_all(sql, (limit,))


@app.get("/api/ml/predictions")
def get_ml_predictions_clean(
    zone: str | None = Query(default=None, description="ICU, ER ou LAB"),
    horizon: str | None = Query(default=None, description="T+15min, T+1h ou T+24h"),
    limit: int = Query(default=300, ge=1, le=5000),
):
    sql = """
        SELECT
            prediction_time AS inserted_at,
            event_time AS original_event_time,
            target_time,
            'machine_learning' AS pipeline,
            'PUHY_Aurora' AS source_dataset,
            zone_name,
            horizon,
            energy_current,
            predicted_energy_kwh,
            actual_energy_kwh,
            residual_energy_kwh,
            abs_error_kwh,
            anomaly_from_prediction,
            anomaly_severity,
            recommended_action,
            model_name
        FROM energy_predictions
        WHERE (%s IS NULL OR zone_name = %s)
          AND (%s IS NULL OR horizon = %s)
        ORDER BY target_time DESC, event_time DESC, prediction_time DESC
        LIMIT %s;
    """
    return fetch_all(sql, (zone, zone, horizon, horizon, limit))


@app.get("/api/ml/summary-clean")
def get_ml_summary_clean():
    sql = """
        SELECT
            'machine_learning' AS pipeline,
            'PUHY_Aurora' AS source_dataset,
            horizon,
            COUNT(*) AS total_predictions,
            ROUND(AVG(abs_error_kwh)::numeric, 4) AS avg_abs_error_kwh,
            ROUND(AVG(predicted_energy_kwh)::numeric, 4) AS avg_predicted_energy_kwh,
            ROUND(AVG(actual_energy_kwh)::numeric, 4) AS avg_actual_energy_kwh,
            ROUND(AVG(residual_energy_kwh)::numeric, 4) AS avg_residual_energy_kwh,
            SUM(CASE WHEN anomaly_from_prediction THEN 1 ELSE 0 END) AS predicted_anomalies
        FROM energy_predictions
        GROUP BY horizon
        ORDER BY horizon;
    """
    return fetch_all(sql)


@app.get("/api/ml/by-zone-clean")
def get_ml_by_zone_clean():
    sql = """
        SELECT
            zone_name,
            horizon,
            COUNT(*) AS total_predictions,
            ROUND(AVG(abs_error_kwh)::numeric, 4) AS avg_abs_error_kwh,
            ROUND(AVG(predicted_energy_kwh)::numeric, 4) AS avg_predicted_energy_kwh,
            ROUND(AVG(actual_energy_kwh)::numeric, 4) AS avg_actual_energy_kwh
        FROM energy_predictions
        GROUP BY zone_name, horizon
        ORDER BY zone_name, horizon;
    """
    return fetch_all(sql)


@app.get("/api/federated/overview")
def get_federated_overview():
    sql = """
        SELECT
            'federated_learning' AS pipeline,
            'PUHY_Aurora' AS source_dataset,
            COUNT(*) AS total_training_rows,
            COUNT(DISTINCT client_id) AS clients,
            MAX(round_number) AS latest_round,
            ROUND(AVG(mae)::numeric, 4) AS avg_mae,
            ROUND(AVG(rmse)::numeric, 4) AS avg_rmse,
            ROUND(AVG(r2)::numeric, 4) AS avg_r2,
            MAX(ts) AS latest_inserted_at
        FROM federated_training_metrics;
    """
    return safe_fetch_one(sql)


@app.get("/api/federated/training")
def get_federated_training_clean(
    client_id: str | None = Query(default=None, description="ICU, ER ou LAB"),
    limit: int = Query(default=300, ge=1, le=5000),
):
    sql = """
        SELECT
            ts AS inserted_at,
            'federated_learning' AS pipeline,
            'PUHY_Aurora' AS source_dataset,
            round_number,
            client_id,
            horizon,
            loss,
            mae,
            rmse,
            r2,
            training_time_seconds,
            n_samples
        FROM federated_training_metrics
        WHERE (%s IS NULL OR client_id = %s)
        ORDER BY ts DESC
        LIMIT %s;
    """
    return fetch_all(sql, (client_id, client_id, limit))
