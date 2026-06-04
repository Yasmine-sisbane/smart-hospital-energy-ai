# =========================
# ml_prediction_service.py
# Service ML temps réel : prédiction énergétique multi-horizon avec XGBoost
# Horizons : T+15min, T+1h, T+24h
# Kafka + XGBoost + PostgreSQL/TimescaleDB
# =========================

import json
import os
from datetime import datetime, timedelta
import time

import joblib
import numpy as np
import pandas as pd
import psycopg2

from psycopg2.extras import execute_values
from kafka import KafkaConsumer


# ============================================================
# 1. Configuration Kafka
# ============================================================

try:
    from config import KAFKA_BOOTSTRAP_SERVERS, PUHY_AURORA_TOPIC
except ImportError:
    KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
    PUHY_AURORA_TOPIC = "puhy_aurora_enriched_data"


# Nouveau group id pour éviter les anciens offsets Kafka
CONSUMER_GROUP_ID = "ml-puhy-aurora-prediction-service-v1"


# ============================================================
# 2. Configuration PostgreSQL / TimescaleDB
# ============================================================

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "hospital_iot")
DB_USER = os.getenv("DB_USER", "hospital_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "hospital_password")


# ============================================================
# 3. Configuration modèles multi-horizon
# ============================================================

MODEL_CONFIGS = {
    "T+15min": {
        "minutes": 15,
        "model_path": "models/xgb_energy_model_t15.joblib",
        "columns_path": "models/xgb_model_columns_t15.joblib",
        "actual_col": "target_observed_energy_next_15min",
        "model_name": "XGBoost_T15_PUHY_Aurora",
        "thresholds": {
            "warning": 3.951,
            "high": 6.861,
            "critical": 9.178,
        },
    },
    "T+1h": {
        "minutes": 60,
        "model_path": "models/xgb_energy_model_t60.joblib",
        "columns_path": "models/xgb_model_columns_t60.joblib",
        "thresholds_path": "models/xgb_anomaly_thresholds_t60.joblib",
        "actual_col": "target_observed_energy_t60",
        "model_name": "XGBoost_T60_PUHY_Aurora",
    },
    "T+24h": {
        "minutes": 1440,
        "model_path": "models/xgb_energy_model_t24h.joblib",
        "columns_path": "models/xgb_model_columns_t24h.joblib",
        "thresholds_path": "models/xgb_anomaly_thresholds_t24h.joblib",
        "actual_col": "target_observed_energy_t24h",
        "model_name": "XGBoost_T24H_PUHY_Aurora",
    },
}


def load_models():
    loaded = {}

    for horizon, cfg in MODEL_CONFIGS.items():
        print(f"Loading model for {horizon}...")

        model = joblib.load(cfg["model_path"])
        columns = joblib.load(cfg["columns_path"])

        if "thresholds" in cfg:
            thresholds = cfg["thresholds"]
        else:
            thresholds = joblib.load(cfg["thresholds_path"])

        loaded[horizon] = {
            "minutes": cfg["minutes"],
            "model": model,
            "columns": columns,
            "thresholds": thresholds,
            "actual_col": cfg["actual_col"],
            "model_name": cfg["model_name"],
        }

    return loaded


MODELS = load_models()


# ============================================================
# 4. Paramètres service
# ============================================================

LOG_EVERY = 100
INSERT_COUNT = 0

BATCH_KAFKA_MESSAGES = 100
BATCH_PREDICTIONS = 300
POLL_TIMEOUT_MS = 1000


ZONE_AREA_M2 = {
    "ICU": 2200,
    "ER": 2000,
    "LAB": 1300,
}


# ============================================================
# 5. Helpers
# ============================================================

def safe_float(value, default=None):
    try:
        if value is None:
            return default

        if isinstance(value, str) and value.strip() == "":
            return default

        value = float(value)

        if np.isnan(value) or np.isinf(value):
            return default

        return value
    except (TypeError, ValueError):
        return default


def parse_event_time(event):
    raw_time = (
        event.get("event_time")
        or event.get("timestamp")
        or event.get("time")
    )

    if raw_time is None:
        return datetime.now()

    return pd.to_datetime(raw_time).to_pydatetime()


def classify_prediction_anomaly(abs_error, thresholds):
    if abs_error is None:
        return False, "pending"

    critical = safe_float(thresholds.get("critical"), None)
    high = safe_float(thresholds.get("high"), None)
    warning = safe_float(thresholds.get("warning"), None)

    if critical is not None and abs_error >= critical:
        return True, "critical"

    if high is not None and abs_error >= high:
        return True, "high"

    if warning is not None and abs_error >= warning:
        return True, "warning"

    return False, "normal"


def get_recommended_action(severity, horizon):
    actions = {
        "critical": f"Anomalie critique prévue pour {horizon}. Inspecter immédiatement HVAC, équipements et capteurs.",
        "high": f"Écart important prévu pour {horizon}. Analyser rapidement la consommation et l’occupation.",
        "warning": f"Écart modéré prévu pour {horizon}. Surveiller la zone et vérifier si l’écart persiste.",
        "normal": "Aucune action nécessaire.",
        "pending": "En attente de la valeur réelle future pour évaluer l’erreur.",
    }

    return actions.get(severity, "Analyser l'événement énergétique.")


def should_log(insert_count, severity):
    return (
        insert_count == 1
        or insert_count % LOG_EVERY == 0
        or severity in ["warning", "high", "critical"]
    )


# ============================================================
# 6. Préparation des features
# ============================================================

def build_model_features(event, model_columns):
    event_time = parse_event_time(event)
    zone_name = event.get("zone_name", "Unknown")

    raw = dict(event)

    raw["energy_current"] = safe_float(
        event.get("energy_current"),
        safe_float(event.get("zone_energy_kwh"), 0.0)
    )

    raw["hour"] = event_time.hour
    raw["day_of_week_num"] = event_time.weekday()
    raw["is_weekend"] = 1 if event_time.weekday() >= 5 else 0
    raw["month"] = event_time.month

    raw["hour_sin"] = np.sin(2 * np.pi * raw["hour"] / 24)
    raw["hour_cos"] = np.cos(2 * np.pi * raw["hour"] / 24)

    raw["day_sin"] = np.sin(2 * np.pi * raw["day_of_week_num"] / 7)
    raw["day_cos"] = np.cos(2 * np.pi * raw["day_of_week_num"] / 7)

    if raw.get("zone_area_m2") is None:
        raw["zone_area_m2"] = ZONE_AREA_M2.get(zone_name, 1000)

    if raw.get("occupancy_ratio") is None:
        occupancy_count = safe_float(raw.get("occupancy_count"), 0.0)
        zone_area_m2 = safe_float(raw.get("zone_area_m2"), 1000.0)

        raw["occupancy_ratio"] = (
            occupancy_count / zone_area_m2
            if zone_area_m2 > 0
            else 0.0
        )

    forbidden = {
        "timestamp",
        "event_time",
        "ingestion_time",
        "prediction_time",
        "source_basis",

        "target_normal_energy_next_15min",
        "target_observed_energy_next_15min",
        "target_anomaly_next_15min",

        "target_normal_energy_t60",
        "target_observed_energy_t60",
        "target_anomaly_t60",

        "target_normal_energy_t24h",
        "target_observed_energy_t24h",
        "target_anomaly_t24h",

        "anomaly_label",
        "anomaly_type",
        "anomaly_severity",

        "normal_zone_energy_kwh",
        "zone_energy_kwh",
    }

    for col in forbidden:
        raw.pop(col, None)

    row = {col: 0 for col in model_columns}

    categorical_cols = [
        "zone_name",
        "zone_type",
        "service_type",
        "season",
        "shift_type",
        "hvac_status",
        "hvac_mode",
        "day_of_week",
    ]

    for col in model_columns:
        if col in raw and col not in categorical_cols:
            value = safe_float(raw.get(col), 0.0)
            row[col] = value if value is not None else 0.0

    for cat_col in categorical_cols:
        cat_value = raw.get(cat_col)

        if cat_value is None:
            continue

        dummy_col = f"{cat_col}_{cat_value}"

        if dummy_col in row:
            row[dummy_col] = 1

    X = pd.DataFrame([row])
    X = X.reindex(columns=model_columns, fill_value=0)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    return X


# ============================================================
# 7. PostgreSQL / TimescaleDB
# ============================================================

def get_db_connection():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )


def ensure_prediction_table(conn):
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS energy_predictions (
                id BIGSERIAL,

                event_time TIMESTAMP NOT NULL,
                target_time TIMESTAMP NOT NULL,
                prediction_time TIMESTAMP NOT NULL DEFAULT NOW(),

                hospital_id TEXT,
                building_id TEXT,
                floor_id TEXT,
                zone_id TEXT,
                zone_name TEXT,

                horizon TEXT NOT NULL,
                energy_current DOUBLE PRECISION,
                predicted_energy_kwh DOUBLE PRECISION,

                actual_energy_kwh DOUBLE PRECISION,
                residual_energy_kwh DOUBLE PRECISION,
                abs_error_kwh DOUBLE PRECISION,

                anomaly_from_prediction BOOLEAN DEFAULT FALSE,
                anomaly_severity TEXT,
                recommended_action TEXT,

                model_name TEXT,
                created_at TIMESTAMP DEFAULT NOW()
            );
            """
        )

        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_energy_predictions_event_time
            ON energy_predictions (event_time DESC);
            """
        )

        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_energy_predictions_zone_time
            ON energy_predictions (zone_name, event_time DESC);
            """
        )

        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_energy_predictions_severity
            ON energy_predictions (anomaly_severity);
            """
        )

        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_energy_predictions_horizon
            ON energy_predictions (horizon);
            """
        )

        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_energy_predictions_horizon_time
            ON energy_predictions (horizon, event_time DESC);
            """
        )

        try:
            cur.execute(
                """
                SELECT create_hypertable(
                    'energy_predictions',
                    'event_time',
                    if_not_exists => TRUE
                );
                """
            )
        except Exception:
            conn.rollback()

            with conn.cursor() as cur2:
                cur2.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_energy_predictions_event_time
                    ON energy_predictions (event_time DESC);
                    """
                )

                cur2.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_energy_predictions_zone_time
                    ON energy_predictions (zone_name, event_time DESC);
                    """
                )

                cur2.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_energy_predictions_severity
                    ON energy_predictions (anomaly_severity);
                    """
                )

                cur2.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_energy_predictions_horizon
                    ON energy_predictions (horizon);
                    """
                )

                cur2.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_energy_predictions_horizon_time
                    ON energy_predictions (horizon, event_time DESC);
                    """
                )

    conn.commit()

def insert_predictions_batch(conn, prediction_events):
    """
    Insère plusieurs prédictions en une seule requête SQL + un seul commit.

    Pour chaque message Kafka :
    - T+15min
    - T+1h
    - T+24h

    Donc 3 lignes insérées ensemble.
    """

    if not prediction_events:
        return

    rows = []

    for prediction_event in prediction_events:
        rows.append((
            prediction_event["event_time"],
            prediction_event["target_time"],
            prediction_event["prediction_time"],

            prediction_event.get("hospital_id"),
            prediction_event.get("building_id"),
            prediction_event.get("floor_id"),
            prediction_event.get("zone_id"),
            prediction_event.get("zone_name"),

            prediction_event.get("horizon"),
            prediction_event.get("energy_current"),
            prediction_event.get("predicted_energy_kwh"),

            prediction_event.get("actual_energy_kwh"),
            prediction_event.get("residual_energy_kwh"),
            prediction_event.get("abs_error_kwh"),

            prediction_event.get("anomaly_from_prediction"),
            prediction_event.get("anomaly_severity"),
            prediction_event.get("recommended_action"),

            prediction_event.get("model_name"),
        ))

    sql = """
    INSERT INTO energy_predictions (
        event_time,
        target_time,
        prediction_time,
        hospital_id,
        building_id,
        floor_id,
        zone_id,
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
    )
    VALUES %s
    ON CONFLICT (event_time, target_time, zone_name, horizon)
    DO NOTHING
"""

    with conn.cursor() as cur:
        execute_values(cur, sql, rows)

    conn.commit()


# ============================================================
# 8. Kafka Consumer
# ============================================================

def create_consumer():
    return KafkaConsumer(
        PUHY_AURORA_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        api_version=(0, 10, 1),
        group_id=CONSUMER_GROUP_ID,
        auto_offset_reset="latest",
        enable_auto_commit=True,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        max_poll_records=BATCH_KAFKA_MESSAGES,
    )

# ============================================================
# 9. Prédiction multi-horizon
# ============================================================

def build_prediction_event(event, horizon, cfg, event_time, prediction_time):
    target_time = event_time + timedelta(minutes=cfg["minutes"])

    X = build_model_features(event, cfg["columns"])
    predicted_energy = float(cfg["model"].predict(X)[0])

    energy_current = safe_float(
        event.get("energy_current"),
        safe_float(event.get("zone_energy_kwh"), None)
    )

    actual_energy = safe_float(event.get(cfg["actual_col"]), None)

    if actual_energy is not None:
        residual = actual_energy - predicted_energy
        abs_error = abs(residual)
        anomaly_flag, anomaly_severity = classify_prediction_anomaly(
            abs_error,
            cfg["thresholds"]
        )
    else:
        residual = None
        abs_error = None
        anomaly_flag = False
        anomaly_severity = "pending"

    prediction_event = {
        "event_time": event_time,
        "target_time": target_time,
        "prediction_time": prediction_time,

        "hospital_id": event.get("hospital_id"),
        "building_id": event.get("building_id"),
        "floor_id": event.get("floor_id"),
        "zone_id": event.get("zone_id"),
        "zone_name": event.get("zone_name"),

        "horizon": horizon,
        "energy_current": energy_current,
        "predicted_energy_kwh": round(predicted_energy, 4),

        "actual_energy_kwh": (
            round(actual_energy, 4)
            if actual_energy is not None
            else None
        ),
        "residual_energy_kwh": (
            round(residual, 4)
            if residual is not None
            else None
        ),
        "abs_error_kwh": (
            round(abs_error, 4)
            if abs_error is not None
            else None
        ),

        "anomaly_from_prediction": anomaly_flag,
        "anomaly_severity": anomaly_severity,
        "recommended_action": get_recommended_action(anomaly_severity, horizon),

        "model_name": cfg["model_name"],
    }

    return prediction_event


def flush_prediction_buffer(conn, prediction_buffer):
    global INSERT_COUNT

    if not prediction_buffer:
        return

    batch_size = len(prediction_buffer)
    start = time.time()

    insert_predictions_batch(conn, prediction_buffer)

    INSERT_COUNT += batch_size

    elapsed = time.time() - start
    last_event = prediction_buffer[-1]

    print(
        f"ML batch inserted | "
        f"Batch={batch_size} predictions | "
        f"Total={INSERT_COUNT} | "
        f"LastHorizon={last_event.get('horizon')} | "
        f"LastZone={last_event.get('zone_name')} | "
        f"LastSeverity={last_event.get('anomaly_severity')} | "
        f"DBTime={elapsed:.3f}s"
    )

    prediction_buffer.clear()

# ============================================================
# 10. Main Loop
# ============================================================

# ============================================================
# 10. Main Loop
# ============================================================

def main():
    print("ML Multi-Horizon Prediction Service started.")
    print("Input topic :", PUHY_AURORA_TOPIC)
    print("Consumer group :", CONSUMER_GROUP_ID)
    print("Horizons :", ", ".join(MODELS.keys()))
    print("Batch Kafka messages :", BATCH_KAFKA_MESSAGES)
    print("Batch predictions :", BATCH_PREDICTIONS)

    for horizon, cfg in MODELS.items():
        print("")
        print(f"[{horizon}]")
        print("Model      :", MODEL_CONFIGS[horizon]["model_path"])
        print("Columns    :", MODEL_CONFIGS[horizon]["columns_path"])
        print("Actual col :", cfg["actual_col"])
        print("Thresholds :")
        print("  WARNING  :", cfg["thresholds"].get("warning"))
        print("  HIGH     :", cfg["thresholds"].get("high"))
        print("  CRITICAL :", cfg["thresholds"].get("critical"))

    conn = get_db_connection()
    ensure_prediction_table(conn)

    consumer = create_consumer()

    print("")
    print("Database connected.")
    print("Waiting for Kafka messages...")

    prediction_buffer = []

    while True:
        try:
            records = consumer.poll(
                timeout_ms=POLL_TIMEOUT_MS,
                max_records=BATCH_KAFKA_MESSAGES
            )

            # Si aucun nouveau message, on vide quand même le buffer restant
            if not records:
                flush_prediction_buffer(conn, prediction_buffer)
                continue

            for _, messages in records.items():
                for message in messages:
                    event = message.value

                    event_time = parse_event_time(event)
                    prediction_time = datetime.now()

                    for horizon, cfg in MODELS.items():
                        prediction_event = build_prediction_event(
                            event=event,
                            horizon=horizon,
                            cfg=cfg,
                            event_time=event_time,
                            prediction_time=prediction_time,
                        )

                        prediction_buffer.append(prediction_event)

                    if len(prediction_buffer) >= BATCH_PREDICTIONS:
                        flush_prediction_buffer(conn, prediction_buffer)

        except psycopg2.Error as db_error:
            print(f"Erreur DB ml_prediction_service : {db_error}")

            try:
                conn.rollback()
            except Exception:
                pass

            prediction_buffer.clear()

        except KeyboardInterrupt:
            print("Arrêt demandé. Vidage du buffer restant...")
            flush_prediction_buffer(conn, prediction_buffer)
            break

        except Exception as e:
            print(f"Erreur ml_prediction_service : {e}")


if __name__ == "__main__":
    main()