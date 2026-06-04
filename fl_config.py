"""
Configuration Federated Learning via Kafka
Projet: Smart Hospital Energy Prediction

Objectif:
- Clients FL: ICU, ER, LAB
- Donnees locales: un CSV par zone
- Communication: Kafka uniquement pour les parametres du modele et les metriques
- Aggregation: FedAvg
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------
# Kafka
# ---------------------------------------------------------------------
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

FL_GLOBAL_MODEL_TOPIC = os.getenv("FL_GLOBAL_MODEL_TOPIC", "fl_global_model")
FL_CLIENT_UPDATES_TOPIC = os.getenv("FL_CLIENT_UPDATES_TOPIC", "fl_client_updates")
FL_TRAINING_METRICS_TOPIC = os.getenv("FL_TRAINING_METRICS_TOPIC", "fl_training_metrics")

# ---------------------------------------------------------------------
# Projet / donnees
# ---------------------------------------------------------------------
PROJECT_DIR = Path(os.getenv("PROJECT_DIR", ".")).resolve()
DATA_DIR = Path(os.getenv("FL_DATA_DIR", PROJECT_DIR / "data" / "fl_clients"))
MODEL_DIR = Path(os.getenv("FL_MODEL_DIR", PROJECT_DIR / "models"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Scaler global commun pour tous les clients FL
# Utilise en V3.1 pour eviter une normalisation differente entre ICU, ER et LAB
GLOBAL_SCALER_PATH = MODEL_DIR / "fl_global_scaler_t60.json"

# Chemin du dataset source PUHY/Aurora.
# A adapter si ton CSV est ailleurs.
CSV_PATH = Path(
    os.getenv(
        "CSV_PATH",
        PROJECT_DIR / "data" / "hospital_energy_semisynthetic_3zones_from_PUHY_Aurora.csv"
    )
)

CLIENTS = ["ICU", "ER", "LAB"]

# ---------------------------------------------------------------------
# ML federated model
# ---------------------------------------------------------------------
HORIZON = os.getenv("FL_HORIZON", "T+1h")
TARGET_COL = os.getenv("FL_TARGET_COL", "target_observed_energy_t60")
HORIZON_STEPS = int(os.getenv("FL_HORIZON_STEPS", "4"))  # 4 x 15min = 1h

# Features numeriques simples et robustes.
# Tu peux en ajouter apres validation.
FEATURE_COLUMNS = [
    "zone_energy_kwh",
    "temp_zone_c",
    "humidity_zone_pct",
    "occupancy_count",
    "outside_temp_c",
    "outside_humidity_pct",
    "solar_radiation_wm2",
    "aqi",
    "pm25",
    "pm10",
    "hour",
    "day_of_week_num",
    "is_weekend",
    "month",
]

# Parametres d'entrainement
NUM_ROUNDS = int(os.getenv("FL_NUM_ROUNDS", "5"))
LOCAL_EPOCHS = int(os.getenv("FL_LOCAL_EPOCHS", "3"))
BATCH_SIZE = int(os.getenv("FL_BATCH_SIZE", "64"))
LEARNING_RATE = float(os.getenv("FL_LEARNING_RATE", "0.001"))
HIDDEN_DIM = int(os.getenv("FL_HIDDEN_DIM", "64"))

GLOBAL_MODEL_PATH = MODEL_DIR / "federated_mlp_t60_global.pt"
GLOBAL_METADATA_PATH = MODEL_DIR / "federated_mlp_t60_metadata.json"

# ---------------------------------------------------------------------
# TimescaleDB optionnel pour stocker les metriques FL
# ---------------------------------------------------------------------
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "hospital_iot")
DB_USER = os.getenv("DB_USER", "hospital_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "hospital_password")