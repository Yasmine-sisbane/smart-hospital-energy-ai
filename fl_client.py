"""
Client Federated Learning via Kafka.

V3.3 :
Chaque client:
1. Lit uniquement ses donnees locales.
2. Applique un scaler global commun aux features X.
3. Normalise aussi la cible y avec un scaler global.
4. Attend un modele global depuis le topic fl_global_model.
5. Entraine localement le modele.
6. Sauvegarde son dernier modele local personnalise.
7. Envoie uniquement les poids du modele vers fl_client_updates.
8. Envoie ses metriques locales vers fl_training_metrics.

Important :
- Le modele apprend sur y normalise.
- Les metriques sont calculees apres denormalisation des predictions.
- Le modele personnalise sauvegarde correspond au dernier modele local du client.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from kafka import KafkaConsumer, KafkaProducer
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fl_config import (
    BATCH_SIZE,
    CLIENTS,
    DATA_DIR,
    FEATURE_COLUMNS,
    FL_CLIENT_UPDATES_TOPIC,
    FL_GLOBAL_MODEL_TOPIC,
    FL_TRAINING_METRICS_TOPIC,
    GLOBAL_SCALER_PATH,
    HIDDEN_DIM,
    KAFKA_BOOTSTRAP_SERVERS,
    LEARNING_RATE,
    LOCAL_EPOCHS,
    MODEL_DIR,
    TARGET_COL,
)

from fl_model import (
    base64_to_state_dict,
    build_model,
    regression_metrics,
    state_dict_to_base64,
)


def load_global_scaler():
    if not GLOBAL_SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Scaler global introuvable : {GLOBAL_SCALER_PATH}\n"
            "Lance d'abord : python fl_build_global_scaler.py"
        )

    with open(GLOBAL_SCALER_PATH, "r", encoding="utf-8") as f:
        scaler = json.load(f)

    x_mean = np.array(scaler["x_mean"], dtype=np.float32)
    x_std = np.array(scaler["x_std"], dtype=np.float32)
    x_std[x_std == 0] = 1.0

    target_mean = float(scaler["target_mean"])
    target_std = float(scaler["target_std"])

    if target_std == 0:
        target_std = 1.0

    return x_mean, x_std, target_mean, target_std


def normalize_x(X, x_mean, x_std):
    return (X - x_mean) / x_std


def normalize_y(y, target_mean, target_std):
    return (y - target_mean) / target_std


def denormalize_y(y_norm, target_mean, target_std):
    return (y_norm * target_std) + target_mean


def load_local_data(path: Path):
    df = pd.read_csv(path)
    df = df.dropna(subset=[TARGET_COL]).copy()

    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COL]).copy()

    X = df[FEATURE_COLUMNS].astype(float).values
    y = df[TARGET_COL].astype(float).values

    split_idx = int(len(df) * 0.8)

    X_train_raw = X[:split_idx]
    X_test_raw = X[split_idx:]

    y_train_raw = y[:split_idx]
    y_test_raw = y[split_idx:]

    x_mean, x_std, target_mean, target_std = load_global_scaler()

    X_train = normalize_x(X_train_raw, x_mean, x_std)
    X_test = normalize_x(X_test_raw, x_mean, x_std)

    y_train_norm = normalize_y(y_train_raw, target_mean, target_std)

    return (
        X_train,
        y_train_norm,
        X_test,
        y_test_raw,
        target_mean,
        target_std,
    )


def train_one_round(model, X_train, y_train_norm, epochs: int, batch_size: int, lr: float):
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train_norm, dtype=torch.float32),
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    last_loss = 0.0

    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()

            pred_norm = model(xb)
            loss = criterion(pred_norm, yb)

            loss.backward()
            optimizer.step()

            last_loss = float(loss.item())

    return last_loss


def evaluate(model, X_test, y_test_raw, target_mean, target_std):
    model.eval()

    with torch.no_grad():
        pred_norm = model(torch.tensor(X_test, dtype=torch.float32)).numpy()

    pred_raw = denormalize_y(
        pred_norm,
        target_mean,
        target_std,
    )

    pred_raw = np.asarray(pred_raw).reshape(-1)
    y_test_raw = np.asarray(y_test_raw).reshape(-1)

    return regression_metrics(y_test_raw, pred_raw)


def save_personalized_model(model, client_id: str, round_number: int, metrics: dict) -> None:
    """
    Sauvegarde le dernier modele local du client.
    Ce modele represente la version personnalisee apres entrainement local.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / f"fl_personalized_{client_id}_t60.pt"
    metadata_path = MODEL_DIR / f"fl_personalized_{client_id}_t60_metadata.json"

    torch.save(model.state_dict(), model_path)

    metadata = {
        "event_type": "fl_personalized_model",
        "client_id": client_id,
        "round_number": int(round_number),
        "model_path": str(model_path),
        "scaler_path": str(GLOBAL_SCALER_PATH),
        "metrics": {
            "mae": float(metrics["mae"]),
            "rmse": float(metrics["rmse"]),
            "r2": float(metrics["r2"]),
        },
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[CLIENT {client_id}] Modele personnalise sauvegarde : {model_path}")
def create_producer():
    return KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        api_version=(0, 10, 1),
        value_serializer=lambda v: json.dumps(
            v,
            ensure_ascii=False,
        ).encode("utf-8"),
    )


def create_global_model_consumer(client_id: str):
    return KafkaConsumer(
        FL_GLOBAL_MODEL_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        api_version=(0, 10, 1),
        group_id=f"fl-client-{client_id.lower()}-global-model",
        auto_offset_reset="latest",
        enable_auto_commit=True,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    )


def run_client(client_id: str, data_path: Path):
    if client_id not in CLIENTS:
        raise ValueError(f"client_id invalide: {client_id}. Choix: {CLIENTS}")

    if not data_path.exists():
        raise FileNotFoundError(f"Fichier local introuvable: {data_path}")

    (
        X_train,
        y_train_norm,
        X_test,
        y_test_raw,
        target_mean,
        target_std,
    ) = load_local_data(data_path)

    input_dim = X_train.shape[1]

    producer = create_producer()
    consumer = create_global_model_consumer(client_id)

    print(f"[CLIENT {client_id}] Demarre")
    print(f"[CLIENT {client_id}] Donnees train={len(X_train)} test={len(X_test)}")
    print(f"[CLIENT {client_id}] Scaler global charge : {GLOBAL_SCALER_PATH}")
    print(
        f"[CLIENT {client_id}] Target scaler | "
        f"mean={target_mean:.4f} std={target_std:.4f}"
    )
    print(f"[CLIENT {client_id}] En attente du modele global Kafka topic={FL_GLOBAL_MODEL_TOPIC}")

    for message in consumer:
        event = message.value

        if event.get("event_type") != "fl_global_model":
            continue

        round_number = int(event["round_number"])
        input_dim_msg = int(event["input_dim"])

        if input_dim_msg != input_dim:
            raise ValueError(
                f"input_dim incompatible: serveur={input_dim_msg}, client={input_dim}"
            )

        model = build_model(
            input_dim=input_dim,
            hidden_dim=HIDDEN_DIM,
        )

        model.load_state_dict(
            base64_to_state_dict(event["model_state_b64"])
        )

        start = time.time()

        loss = train_one_round(
            model=model,
            X_train=X_train,
            y_train_norm=y_train_norm,
            epochs=LOCAL_EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE,
        )

        training_time = time.time() - start

        metrics = evaluate(
            model=model,
            X_test=X_test,
            y_test_raw=y_test_raw,
            target_mean=target_mean,
            target_std=target_std,
        )

        # V3.3 : sauvegarde du modele local personnalise du client.
        save_personalized_model(
            model=model,
            client_id=client_id,
            round_number=round_number,
            metrics=metrics,
        )

        update_event = {
            "event_type": "fl_client_update",
            "round_number": round_number,
            "client_id": client_id,
            "n_samples": int(len(X_train)),
            "input_dim": input_dim,
            "model_state_b64": state_dict_to_base64(model.state_dict()),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }

        metrics_event = {
            "event_type": "fl_training_metrics",
            "round_number": round_number,
            "client_id": client_id,
            "horizon": "T+1h",
            "loss": float(loss),
            "mae": float(metrics["mae"]),
            "rmse": float(metrics["rmse"]),
            "r2": float(metrics["r2"]),
            "training_time_seconds": float(training_time),
            "n_samples": int(len(X_train)),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }

        producer.send(
            FL_CLIENT_UPDATES_TOPIC,
            value=update_event,
        ).get(timeout=20)

        producer.send(
            FL_TRAINING_METRICS_TOPIC,
            value=metrics_event,
        ).get(timeout=20)

        producer.flush()

        print(
            f"[CLIENT {client_id}] Round={round_number} | "
            f"LossNorm={loss:.4f} "
            f"MAE={metrics['mae']:.4f} "
            f"RMSE={metrics['rmse']:.4f} "
            f"R2={metrics['r2']:.4f} | "
            f"modele personnalise sauvegarde | "
            f"envoye vers {FL_CLIENT_UPDATES_TOPIC}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Client Federated Learning ICU/ER/LAB via Kafka."
    )

    parser.add_argument(
        "--client-id",
        required=True,
        choices=CLIENTS,
    )

    parser.add_argument(
        "--data",
        default=None,
        help="Chemin du CSV local du client.",
    )

    args = parser.parse_args()

    default_path = DATA_DIR / f"client_{args.client_id.lower()}.csv"

    run_client(
        args.client_id,
        Path(args.data) if args.data else default_path,
    )