# =========================
# train_xgboost_t24h.py
# Modèle XGBoost pour prédiction énergétique T+24h
# Dataset PUHY/Aurora semi-synthétique 3 zones
# =========================

import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from xgboost import XGBRegressor


# ============================================================
# 1. Configuration 
# ============================================================

CSV_PATH = r"C:\Users\lenovo\OneDrive\Documents\hospital_kafka_project\data\hospital_energy_semisynthetic_3zones_from_PUHY_Aurora.csv"

OUTPUT_DIR = Path("models")
OUTPUT_DIR.mkdir(exist_ok=True)

RESULTS_DIR = Path("outputs_ml")
RESULTS_DIR.mkdir(exist_ok=True)

HORIZON_NAME = "T+24h"
HORIZON_KEY = "t24h"
HORIZON_STEPS = 96
HORIZON_MINUTES = 1440

TARGET_NORMAL_COL = "target_normal_energy_t24h"
TARGET_OBSERVED_COL = "target_observed_energy_t24h"
TARGET_ANOMALY_COL = "target_anomaly_t24h"

MODEL_PATH = OUTPUT_DIR / "xgb_energy_model_t24h.joblib"
COLUMNS_PATH = OUTPUT_DIR / "xgb_model_columns_t24h.joblib"
THRESHOLDS_PATH = OUTPUT_DIR / "xgb_anomaly_thresholds_t24h.joblib"

METRICS_PATH = RESULTS_DIR / "xgb_metrics_t24h.csv"
TEST_RESULTS_PATH = RESULTS_DIR / "xgb_test_predictions_t24h.csv"
THRESHOLD_REPORT_PATH = RESULTS_DIR / "xgb_threshold_report_t24h.csv"


# ============================================================
# 2. Fonctions utilitaires
# ============================================================

def regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


def build_features(dataframe):
    cols_to_drop = [
        # Temps brut
        "timestamp",

        # Texte explicatif
        "source_basis",

        # Cibles originales T+15 si elles existent déjà dans le CSV
        "target_normal_energy_next_15min",
        "target_observed_energy_next_15min",
        "target_anomaly_next_15min",

        # Cibles T+24h
        TARGET_NORMAL_COL,
        TARGET_OBSERVED_COL,
        TARGET_ANOMALY_COL,

        # Labels anomalies
        "anomaly_label",
        "anomaly_type",
        "anomaly_severity",

        # Fuite de cible normale directe
        "normal_zone_energy_kwh",

        # On remplace zone_energy_kwh par energy_current
        "zone_energy_kwh",
    ]

    cols_to_drop = [c for c in cols_to_drop if c in dataframe.columns]

    X = dataframe.drop(columns=cols_to_drop, errors="ignore").copy()

    X = pd.get_dummies(X, drop_first=True)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    return X


def evaluate_thresholds(train_abs_error, test_abs_error_observed, y_test_anomaly):
    percentiles = [
        99.0, 99.3, 99.5, 99.7, 99.8,
        99.9, 99.95, 99.97, 99.99, 100.0
    ]

    rows = []

    best_f1 = -1
    best_row = None

    print("\n===== Recherche seuils anomalies T+24h =====")

    for p in percentiles:
        threshold = float(np.percentile(train_abs_error, p))

        y_pred_anomaly = (test_abs_error_observed >= threshold).astype(int)

        precision = precision_score(y_test_anomaly, y_pred_anomaly, zero_division=0)
        recall = recall_score(y_test_anomaly, y_pred_anomaly, zero_division=0)
        f1 = f1_score(y_test_anomaly, y_pred_anomaly, zero_division=0)
        cm = confusion_matrix(y_test_anomaly, y_pred_anomaly)

        row = {
            "percentile": p,
            "threshold_kwh": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tn": int(cm[0, 0]) if cm.shape == (2, 2) else None,
            "fp": int(cm[0, 1]) if cm.shape == (2, 2) else None,
            "fn": int(cm[1, 0]) if cm.shape == (2, 2) else None,
            "tp": int(cm[1, 1]) if cm.shape == (2, 2) else None,
        }

        rows.append(row)

        print(
            f"Percentile={p} | "
            f"Threshold={threshold:.3f} kWh | "
            f"Precision={precision:.3f} | "
            f"Recall={recall:.3f} | "
            f"F1={f1:.3f}"
        )
        print(cm)

        if f1 > best_f1:
            best_f1 = f1
            best_row = row

    threshold_report = pd.DataFrame(rows)

    thresholds = {
        "warning": float(np.percentile(train_abs_error, 99.9)),
        "high": float(np.percentile(train_abs_error, 99.99)),
        "critical": float(np.percentile(train_abs_error, 100.0)),
        "best_f1_threshold": float(best_row["threshold_kwh"]),
        "best_f1_percentile": float(best_row["percentile"]),
        "best_f1": float(best_row["f1"]),
    }

    return thresholds, threshold_report


def classify_with_thresholds(abs_error, thresholds):
    if abs_error >= thresholds["critical"]:
        return "critical"
    if abs_error >= thresholds["high"]:
        return "high"
    if abs_error >= thresholds["warning"]:
        return "warning"
    return "normal"


# ============================================================
# 3. Chargement dataset
# ============================================================

df = pd.read_csv(CSV_PATH)

print("Dataset chargé :", df.shape)
print("Colonnes :", len(df.columns))

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values(["zone_name", "timestamp"]).copy()

df["energy_current"] = df["zone_energy_kwh"]


# ============================================================
# 4. Création cible T+24h
# ============================================================

df[TARGET_NORMAL_COL] = (
    df.groupby("zone_name")["normal_zone_energy_kwh"].shift(-HORIZON_STEPS)
)

df[TARGET_OBSERVED_COL] = (
    df.groupby("zone_name")["zone_energy_kwh"].shift(-HORIZON_STEPS)
)

df[TARGET_ANOMALY_COL] = (
    df.groupby("zone_name")["anomaly_label"].shift(-HORIZON_STEPS)
)

required_cols = [
    TARGET_NORMAL_COL,
    TARGET_OBSERVED_COL,
    TARGET_ANOMALY_COL,
    "energy_lag_1",
    "energy_lag_4",
    "energy_lag_96",
    "energy_rolling_4",
    "energy_rolling_96",
]

df = df.dropna(subset=required_cols).copy()
df[TARGET_ANOMALY_COL] = df[TARGET_ANOMALY_COL].astype(int)

print("Après création cible T+24h :", df.shape)


# ============================================================
# 5. Split temporel train/test
# ============================================================

unique_times = np.sort(df["timestamp"].unique())
split_index = int(len(unique_times) * 0.8)
split_time = unique_times[split_index]

train_df = df[df["timestamp"] < split_time].copy()
test_df = df[df["timestamp"] >= split_time].copy()

train_normal_df = train_df[train_df["anomaly_label"] == 0].copy()

print("\nSplit temporel")
print("Train :", train_df["timestamp"].min(), "→", train_df["timestamp"].max(), train_df.shape)
print("Test  :", test_df["timestamp"].min(), "→", test_df["timestamp"].max(), test_df.shape)
print("Train normal uniquement :", train_normal_df.shape)


# ============================================================
# 6. Features
# ============================================================

X_train = build_features(train_normal_df)
y_train = train_normal_df[TARGET_NORMAL_COL]

X_test = build_features(test_df)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

y_test_normal = test_df[TARGET_NORMAL_COL]
y_test_observed = test_df[TARGET_OBSERVED_COL]
y_test_anomaly = test_df[TARGET_ANOMALY_COL]

print("\nX_train :", X_train.shape)
print("X_test  :", X_test.shape)


# ============================================================
# 7. Baselines
# ============================================================

print("\n===== Baselines T+24h =====")

metrics_rows = []

for baseline_col in ["energy_current", "energy_rolling_4", "energy_lag_4", "energy_lag_96"]:
    if baseline_col in X_test.columns:
        pred_base = X_test[baseline_col]

        mae, rmse, r2 = regression_metrics(y_test_normal, pred_base)

        metrics_rows.append({
            "horizon": HORIZON_NAME,
            "model": f"Baseline_{baseline_col}",
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
        })

        print(
            f"Baseline {baseline_col} | "
            f"MAE={mae:.3f} | RMSE={rmse:.3f} | R2={r2:.3f}"
        )


# ============================================================
# 8. Modèle XGBoost
# ============================================================

print("\n===== Entraînement XGBoost T+24h =====")

xgb_model = XGBRegressor(
    n_estimators=900,
    max_depth=5,
    learning_rate=0.025,
    subsample=0.85,
    colsample_bytree=0.85,
    min_child_weight=3,
    reg_lambda=2.0,
    reg_alpha=0.1,
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1,
)

xgb_model.fit(X_train, y_train)

y_pred_normal = xgb_model.predict(X_test)

mae, rmse, r2 = regression_metrics(y_test_normal, y_pred_normal)

print(
    f"XGBoost {HORIZON_NAME} | "
    f"MAE={mae:.3f} | RMSE={rmse:.3f} | R2={r2:.3f}"
)


# ============================================================
# 9. Anomalies par erreur de prédiction
# ============================================================

train_pred = xgb_model.predict(X_train)
train_abs_error = np.abs(y_train - train_pred)

test_residual_observed = y_test_observed - y_pred_normal
test_abs_error_observed = np.abs(test_residual_observed)

thresholds, threshold_report = evaluate_thresholds(
    train_abs_error=train_abs_error,
    test_abs_error_observed=test_abs_error_observed,
    y_test_anomaly=y_test_anomaly,
)

print("\n===== Seuils retenus T+24h =====")
print("warning :", round(thresholds["warning"], 3))
print("high    :", round(thresholds["high"], 3))
print("critical:", round(thresholds["critical"], 3))
print("best F1 threshold :", round(thresholds["best_f1_threshold"], 3))
print("best F1 percentile :", thresholds["best_f1_percentile"])
print("best F1 :", round(thresholds["best_f1"], 3))

y_pred_anomaly = (test_abs_error_observed >= thresholds["warning"]).astype(int)

precision = precision_score(y_test_anomaly, y_pred_anomaly, zero_division=0)
recall = recall_score(y_test_anomaly, y_pred_anomaly, zero_division=0)
f1 = f1_score(y_test_anomaly, y_pred_anomaly, zero_division=0)
cm = confusion_matrix(y_test_anomaly, y_pred_anomaly)

print("\n===== Détection anomalies T+24h avec seuil warning =====")
print("Precision :", round(precision, 3))
print("Recall    :", round(recall, 3))
print("F1        :", round(f1, 3))
print("Confusion matrix :")
print(cm)


# ============================================================
# 10. Importance features
# ============================================================

feature_importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": xgb_model.feature_importances_,
}).sort_values("importance", ascending=False)

print("\nTop 15 features importantes T+24h :")
print(feature_importance.head(15).to_string(index=False))


# ============================================================
# 11. Sauvegardes
# ============================================================

joblib.dump(xgb_model, MODEL_PATH)
joblib.dump(X_train.columns.tolist(), COLUMNS_PATH)
joblib.dump(thresholds, THRESHOLDS_PATH)

metrics_rows.append({
    "horizon": HORIZON_NAME,
    "model": "XGBoost_T24H",
    "MAE": mae,
    "RMSE": rmse,
    "R2": r2,
    "Precision_warning_threshold": precision,
    "Recall_warning_threshold": recall,
    "F1_warning_threshold": f1,
    "warning_threshold": thresholds["warning"],
    "high_threshold": thresholds["high"],
    "critical_threshold": thresholds["critical"],
    "best_f1": thresholds["best_f1"],
    "best_f1_threshold": thresholds["best_f1_threshold"],
    "best_f1_percentile": thresholds["best_f1_percentile"],
})

metrics_df = pd.DataFrame(metrics_rows)
metrics_df.to_csv(METRICS_PATH, index=False)

threshold_report.to_csv(THRESHOLD_REPORT_PATH, index=False)

test_results = test_df[[
    "timestamp",
    "zone_name",
    "zone_type",
    "energy_current",
    "zone_energy_kwh",
    "normal_zone_energy_kwh",
    "anomaly_label",
    "anomaly_type",
    TARGET_NORMAL_COL,
    TARGET_OBSERVED_COL,
    TARGET_ANOMALY_COL,
]].copy()

test_results["prediction_normal_energy_t24h"] = y_pred_normal
test_results["residual_observed_t24h"] = test_residual_observed
test_results["abs_error_observed_t24h"] = test_abs_error_observed
test_results["predicted_anomaly_t24h"] = y_pred_anomaly
test_results["predicted_severity_t24h"] = [
    classify_with_thresholds(v, thresholds)
    for v in test_abs_error_observed
]

test_results.to_csv(TEST_RESULTS_PATH, index=False)

print("\nFichiers sauvegardés T+24h :")
print("-", MODEL_PATH)
print("-", COLUMNS_PATH)
print("-", THRESHOLDS_PATH)
print("-", METRICS_PATH)
print("-", TEST_RESULTS_PATH)
print("-", THRESHOLD_REPORT_PATH)

print("\nEntraînement XGBoost T+24h terminé avec succès.")