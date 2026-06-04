# =========================
# train_xgboost_energy_model.py
# Modèle avancé XGBoost pour prédiction énergétique T+15 minutes
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

TARGET_COL = "target_normal_energy_next_15min"
OBSERVED_TARGET_COL = "target_observed_energy_next_15min"

MODEL_PATH = OUTPUT_DIR / "xgb_energy_model_t15.joblib"
COLUMNS_PATH = OUTPUT_DIR / "xgb_model_columns_t15.joblib"

# Seuil principal automatiquement choisi selon F1-score
THRESHOLD_PATH = OUTPUT_DIR / "xgb_anomaly_threshold_t15.joblib"

# Seuils multi-niveaux pour dashboard / Kafka ML service
THRESHOLDS_LEVELS_PATH = OUTPUT_DIR / "xgb_anomaly_thresholds_levels_t15.joblib"

METRICS_PATH = OUTPUT_DIR / "xgb_training_metrics_t15.csv"
TEST_RESULTS_PATH = OUTPUT_DIR / "xgb_test_predictions_t15.csv"
THRESHOLD_SEARCH_PATH = OUTPUT_DIR / "xgb_threshold_search_t15.csv"


# ============================================================
# 2. Charger le dataset
# ============================================================

df = pd.read_csv(CSV_PATH)

print("Dataset chargé :", df.shape)
print("Colonnes :", len(df.columns))

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values(["timestamp", "zone_name"]).copy()


# ============================================================
# 3. Nettoyage de base
# ============================================================

required_cols = [
    TARGET_COL,
    OBSERVED_TARGET_COL,
    "energy_lag_1",
    "energy_lag_4",
    "energy_lag_96",
    "energy_rolling_4",
    "energy_rolling_96",
]

df = df.dropna(subset=required_cols).copy()

# Énergie actuelle connue à l'instant T.
# C'est autorisé car on prédit T+15.
df["energy_current"] = df["zone_energy_kwh"]

# Anomalie suivante : utile pour évaluer la détection d'anomalies à T+15.
df["target_anomaly_next_15min"] = (
    df.groupby("zone_name")["anomaly_label"].shift(-1)
)

df["target_anomaly_next_15min"] = (
    df["target_anomaly_next_15min"]
    .fillna(0)
    .astype(int)
)

print("Après nettoyage :", df.shape)


# ============================================================
# 4. Split temporel train/test
# ============================================================
# On coupe selon le temps, pas au hasard.

unique_times = np.sort(df["timestamp"].unique())
split_index = int(len(unique_times) * 0.8)
split_time = unique_times[split_index]

train_df = df[df["timestamp"] < split_time].copy()
test_df = df[df["timestamp"] >= split_time].copy()

print("\nSplit temporel")
print(
    "Train :",
    train_df["timestamp"].min(),
    "→",
    train_df["timestamp"].max(),
    train_df.shape,
)
print(
    "Test  :",
    test_df["timestamp"].min(),
    "→",
    test_df["timestamp"].max(),
    test_df.shape,
)


# ============================================================
# 5. Entraînement uniquement sur les lignes normales
# ============================================================
# Objectif : apprendre la consommation normale.
# Les anomalies ne doivent pas guider le modèle.

train_normal_df = train_df[train_df["anomaly_label"] == 0].copy()

print("\nTrain normal uniquement :", train_normal_df.shape)


# ============================================================
# 6. Colonnes interdites / leakage
# ============================================================

cols_to_drop = [
    # Temps brut
    "timestamp",

    # Texte explicatif inutile
    "source_basis",

    # Valeurs cibles / futures
    "target_normal_energy_next_15min",
    "target_observed_energy_next_15min",
    "target_anomaly_next_15min",

    # Labels d'anomalies
    "anomaly_label",
    "anomaly_type",
    "anomaly_severity",

    # Cette colonne donne trop directement la consommation normale.
    # On garde energy_current au lieu de normal_zone_energy_kwh.
    "normal_zone_energy_kwh",

    # La valeur actuelle brute est copiée dans energy_current
    # pour avoir un nom plus clair côté Kafka.
    "zone_energy_kwh",
]

cols_to_drop = [c for c in cols_to_drop if c in df.columns]


def build_features(dataframe):
    X = dataframe.drop(columns=cols_to_drop, errors="ignore").copy()

    # Encodage des colonnes texte
    X = pd.get_dummies(X, drop_first=True)

    # Nettoyage numérique
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    return X


X_train = build_features(train_normal_df)
y_train = train_normal_df[TARGET_COL]

X_test = build_features(test_df)
y_test_normal = test_df[TARGET_COL]
y_test_observed = test_df[OBSERVED_TARGET_COL]
y_test_anomaly = test_df["target_anomaly_next_15min"]

# Aligner les colonnes test sur train
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

print("\nX_train :", X_train.shape)
print("X_test  :", X_test.shape)


# ============================================================
# 7. Baselines
# ============================================================

def regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return mae, rmse, r2


print("\n===== Baselines =====")

baseline_results = []

for baseline_col in [
    "energy_current",
    "energy_rolling_4",
    "energy_lag_4",
    "energy_lag_96",
]:
    if baseline_col in X_test.columns:
        pred_base = X_test[baseline_col]

        mae, rmse, r2 = regression_metrics(y_test_normal, pred_base)

        baseline_results.append({
            "model": f"Baseline_{baseline_col}",
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
        })

        print(
            f"Baseline {baseline_col} | "
            f"MAE: {mae:.3f}, RMSE: {rmse:.3f}, R2: {r2:.3f}"
        )


# ============================================================
# 8. Modèle XGBoost
# ============================================================

print("\n===== Entraînement XGBoost =====")

xgb_model = XGBRegressor(
    n_estimators=700,
    max_depth=5,
    learning_rate=0.03,
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

xgb_mae, xgb_rmse, xgb_r2 = regression_metrics(
    y_test_normal,
    y_pred_normal,
)

print(
    f"XGBoost | "
    f"MAE: {xgb_mae:.3f}, RMSE: {xgb_rmse:.3f}, R2: {xgb_r2:.3f}"
)


# ============================================================
# 9. Détection d'anomalies par erreur de prédiction
# ============================================================
# Le modèle prédit la consommation normale à T+15.
# On compare ensuite la consommation observée avec cette prédiction.
# Si l'erreur est grande, on considère cela comme une anomalie.

train_pred = xgb_model.predict(X_train)
train_residual = y_train - train_pred
train_abs_error = np.abs(train_residual)

# Erreur observée sur le test :
# consommation observée à T+15 - consommation normale prédite à T+15
test_residual_observed = y_test_observed - y_pred_normal
test_abs_error_observed = np.abs(test_residual_observed)


# ------------------------------------------------------------
# Seuils multi-niveaux pour le futur dashboard
# ------------------------------------------------------------
# Ces seuils permettront d'afficher :
# normal / warning / high / critical / extreme

threshold_levels = {
    "warning": float(np.percentile(train_abs_error, 99.7)),
    "high": float(np.percentile(train_abs_error, 99.9)),
    "critical": float(np.percentile(train_abs_error, 99.99)),
    "extreme": float(np.percentile(train_abs_error, 100.0)),
}

print("\n===== Seuils multi-niveaux =====")
for level, value in threshold_levels.items():
    print(f"{level.upper()} threshold: {value:.3f} kWh")


# ------------------------------------------------------------
# Recherche automatique du meilleur seuil selon F1-score
# ------------------------------------------------------------
# Important :
# On ne met pas 100 ici.
# Percentile 100 = erreur maximale du train normal.
# C'est utile comme seuil "extreme", mais trop strict comme seuil général.

percentiles = [
    99.0,
    99.1,
    99.2,
    99.3,
    99.4,
    99.5,
    99.6,
    99.7,
    99.8,
    99.85,
    99.9,
    99.93,
    99.95,
    99.97,
    99.99,
]

best_f1 = -1
best_threshold = None
best_percentile = None
best_precision = None
best_recall = None
best_cm = None
best_y_pred_anomaly = None

threshold_search_rows = []

print("\n===== Recherche meilleur seuil anomalie =====")

for p in percentiles:
    threshold = float(np.percentile(train_abs_error, p))

    y_pred_tmp = (test_abs_error_observed > threshold).astype(int)

    precision_tmp = precision_score(
        y_test_anomaly,
        y_pred_tmp,
        zero_division=0,
    )
    recall_tmp = recall_score(
        y_test_anomaly,
        y_pred_tmp,
        zero_division=0,
    )
    f1_tmp = f1_score(
        y_test_anomaly,
        y_pred_tmp,
        zero_division=0,
    )
    cm_tmp = confusion_matrix(y_test_anomaly, y_pred_tmp)

    tn, fp, fn, tp = cm_tmp.ravel()

    threshold_search_rows.append({
        "percentile": p,
        "threshold_kwh": threshold,
        "precision": precision_tmp,
        "recall": recall_tmp,
        "f1": f1_tmp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    })

    print(
        f"Percentile={p} | "
        f"Threshold={threshold:.3f} kWh | "
        f"Precision={precision_tmp:.3f} | "
        f"Recall={recall_tmp:.3f} | "
        f"F1={f1_tmp:.3f}"
    )
    print(cm_tmp)

    if f1_tmp > best_f1:
        best_f1 = f1_tmp
        best_threshold = threshold
        best_percentile = p
        best_precision = precision_tmp
        best_recall = recall_tmp
        best_cm = cm_tmp
        best_y_pred_anomaly = y_pred_tmp


# Seuil final retenu selon F1-score
anomaly_threshold = best_threshold
y_pred_anomaly = best_y_pred_anomaly

precision = best_precision
recall = best_recall
f1 = best_f1
cm = best_cm

print("\n===== Meilleur seuil retenu =====")
print("Percentile :", best_percentile)
print("Threshold  :", round(anomaly_threshold, 3), "kWh")
print("Precision  :", round(precision, 3))
print("Recall     :", round(recall, 3))
print("F1-score   :", round(f1, 3))
print("Confusion matrix :")
print(cm)

print("\n===== Détection anomalies par prédiction =====")
print("Precision :", round(precision, 3))
print("Recall    :", round(recall, 3))
print("F1-score  :", round(f1, 3))
print("Confusion matrix :")
print(cm)


# ============================================================
# 10. Importance des features
# ============================================================

feature_importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": xgb_model.feature_importances_,
}).sort_values("importance", ascending=False)

print("\nTop 15 features importantes :")
print(feature_importance.head(15).to_string(index=False))


# ============================================================
# 11. Sauvegardes
# ============================================================

joblib.dump(xgb_model, MODEL_PATH)
joblib.dump(X_train.columns.tolist(), COLUMNS_PATH)
joblib.dump(anomaly_threshold, THRESHOLD_PATH)
joblib.dump(threshold_levels, THRESHOLDS_LEVELS_PATH)

metrics_rows = []

metrics_rows.extend(baseline_results)

metrics_rows.append({
    "model": "XGBoost_T15_NormalEnergy",
    "MAE": xgb_mae,
    "RMSE": xgb_rmse,
    "R2": xgb_r2,
})

metrics_rows.append({
    "model": "XGBoost_AnomalyDetection_BestThreshold",
    "MAE": np.nan,
    "RMSE": np.nan,
    "R2": np.nan,
    "Precision": precision,
    "Recall": recall,
    "F1": f1,
    "BestPercentile": best_percentile,
    "AnomalyThresholdKwh": anomaly_threshold,
})

for level, threshold_value in threshold_levels.items():
    metrics_rows.append({
        "model": f"ThresholdLevel_{level}",
        "MAE": np.nan,
        "RMSE": np.nan,
        "R2": np.nan,
        "Precision": np.nan,
        "Recall": np.nan,
        "F1": np.nan,
        "BestPercentile": np.nan,
        "AnomalyThresholdKwh": threshold_value,
    })

metrics_df = pd.DataFrame(metrics_rows)
metrics_df.to_csv(METRICS_PATH, index=False)

threshold_search_df = pd.DataFrame(threshold_search_rows)
threshold_search_df.to_csv(THRESHOLD_SEARCH_PATH, index=False)

test_results = test_df[[
    "timestamp",
    "zone_name",
    "zone_type",
    "zone_energy_kwh",
    "normal_zone_energy_kwh",
    "anomaly_label",
    "anomaly_type",
    TARGET_COL,
    OBSERVED_TARGET_COL,
]].copy()

test_results["prediction_normal_energy_next_15min"] = y_pred_normal
test_results["residual_observed_next_15min"] = test_residual_observed
test_results["abs_error_observed_next_15min"] = test_abs_error_observed
test_results["predicted_anomaly_next_15min"] = y_pred_anomaly

# Classification multi-niveaux pour analyse offline
def classify_error(abs_error):
    if abs_error >= threshold_levels["extreme"]:
        return "extreme"
    elif abs_error >= threshold_levels["critical"]:
        return "critical"
    elif abs_error >= threshold_levels["high"]:
        return "high"
    elif abs_error >= threshold_levels["warning"]:
        return "warning"
    else:
        return "normal"


test_results["prediction_anomaly_level"] = test_results[
    "abs_error_observed_next_15min"
].apply(classify_error)

test_results.to_csv(TEST_RESULTS_PATH, index=False)

print("\nFichiers sauvegardés :")
print("-", MODEL_PATH)
print("-", COLUMNS_PATH)
print("-", THRESHOLD_PATH)
print("-", THRESHOLDS_LEVELS_PATH)
print("-", METRICS_PATH)
print("-", THRESHOLD_SEARCH_PATH)
print("-", TEST_RESULTS_PATH)

print("\nEntraînement XGBoost terminé avec succès.")