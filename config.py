# =========================
# config.py
# Configuration globale du projet
# =========================

KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"

# =========================
# Topics Kafka
# =========================

# Données internes hôpital
HOSPITAL_RAW_TOPIC = "hospital_raw_data"

# Données API météo
WEATHER_TOPIC = "weather_data"

# Données API qualité de l'air
AIR_QUALITY_TOPIC = "air_quality_data"

# Données enrichies après fusion
HOSPITAL_ENRICHED_TOPIC = "hospital_enriched_data"

# Donnees ML PUHY/Aurora separees du pipeline Kafka hospitalier
PUHY_AURORA_TOPIC = "puhy_aurora_enriched_data"

# Topics analytics
METRICS_TOPIC = "hospital_metrics"
ANOMALIES_TOPIC = "hospital_anomalies"
ALERTS_TOPIC = "hospital_alerts"

# Topic IA plus tard
PREDICTIONS_TOPIC = "hospital_predictions"

# Compatibilité avec ancien code
TOPIC_NAME = HOSPITAL_RAW_TOPIC

# =========================
# Dataset brut capteurs hôpital
# =========================

CSV_FILE_PATH = r"C:\Users\lenovo\OneDrive\Documents\hospital_kafka_project\data\hospital_sensor_raw_180days_51840rows.csv"

# =========================
# Localisation API : Montréal
# =========================

LATITUDE = 45.5019
LONGITUDE = -73.5674
TIMEZONE = "America/Toronto"

# =========================
# Fréquences APIs
# =========================

WEATHER_REFRESH_SECONDS = 30
AIR_QUALITY_REFRESH_SECONDS = 60

# =========================
# Simulation temps réel / replay historique
# =========================

HOSPITAL_REPLAY_SLEEP_SECONDS = 0.5
HOSPITAL_MAX_ROWS = 300

# =========================
# Seuils analytics temps réel
# =========================

# Énergie brute par événement de 15 minutes
HIGH_ENERGY_THRESHOLD = 35.0
CRITICAL_ENERGY_THRESHOLD = 40.0

# Ancien seuil simple, gardé pour compatibilité
ENERGY_ALERT_THRESHOLD = HIGH_ENERGY_THRESHOLD
ENERGY_RESIDUAL_THRESHOLD = 2.0

# Température intérieure recommandée pour zones critiques
MIN_INDOOR_TEMP = 20.0
MAX_INDOOR_TEMP = 26.0

# Humidité recommandée
MIN_HUMIDITY_PCT = 40.0
MAX_HUMIDITY_PCT = 60.0

# Qualité de l'air
BAD_AQI_THRESHOLD = 100.0

# Consommation par occupant
HIGH_ENERGY_PER_OCCUPANT_THRESHOLD = 3.0

# Intensité énergétique annualisée
# ICU réaliste : environ 250 à 500 kWh/m²/an
HIGH_ANNUALIZED_KWH_M2 = 650.0
CRITICAL_ANNUALIZED_KWH_M2 = 800.0

# =========================
# Fichiers de sortie
# =========================

PROCESSED_EVENTS_FILE = "outputs/processed_events.jsonl"
ANOMALIES_FILE = "outputs/anomalies.jsonl"
PREDICTIONS_FILE = "outputs/predictions.jsonl"
ALERTS_FILE = "outputs/alerts.jsonl"

# =========================
# Modèle ML
# =========================

MODEL_PATH = "models/best_energy_model.joblib"
MODEL_COLUMNS_PATH = "models/model_columns.json"
MODEL_METRICS_PATH = "models/model_metrics.json"




# =========================
# TimescaleDB
# =========================

TIMESCALE_HOST = "localhost"
TIMESCALE_PORT = 5432
TIMESCALE_DB = "hospital_iot"
TIMESCALE_USER = "hospital_user"
TIMESCALE_PASSWORD = "hospital_password"