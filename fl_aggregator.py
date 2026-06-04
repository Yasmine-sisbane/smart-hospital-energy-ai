"""
Serveur / Aggregateur Federated Learning via Kafka.

Role:
1. Publier un modele global initial dans fl_global_model.
2. Attendre les updates des clients ICU, ER, LAB dans fl_client_updates.
3. Agreger les poids avec FedAvg.
4. Publier le nouveau modele global.
5. Repeter pendant NUM_ROUNDS.

Les donnees brutes ne sont jamais recues par ce serveur.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime

import torch
from kafka import KafkaConsumer, KafkaProducer

from fl_config import (
    CLIENTS,
    FEATURE_COLUMNS,
    FL_CLIENT_UPDATES_TOPIC,
    FL_GLOBAL_MODEL_TOPIC,
    GLOBAL_METADATA_PATH,
    GLOBAL_MODEL_PATH,
    HIDDEN_DIM,
    KAFKA_BOOTSTRAP_SERVERS,
    NUM_ROUNDS,
)
from fl_model import base64_to_state_dict, build_model, fedavg, state_dict_to_base64


def create_producer():
    return KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        api_version=(0, 10, 1),
        value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
    )


def create_update_consumer():
    return KafkaConsumer(
        FL_CLIENT_UPDATES_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        api_version=(0, 10, 1),
        group_id="fl-aggregator-updates",
        auto_offset_reset="latest",
        enable_auto_commit=True,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    )


def publish_global_model(producer, model, round_number: int, input_dim: int):
    event = {
        "event_type": "fl_global_model",
        "round_number": round_number,
        "input_dim": input_dim,
        "hidden_dim": HIDDEN_DIM,
        "model_state_b64": state_dict_to_base64(model.state_dict()),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    metadata = producer.send(FL_GLOBAL_MODEL_TOPIC, value=event).get(timeout=20)
    producer.flush()
    print(
        f"[AGGREGATOR] Modele global publie | "
        f"Round={round_number} | Topic={metadata.topic} | Offset={metadata.offset}"
    )


def collect_round_updates(consumer, round_number: int, expected_clients: list[str]):
    received = {}
    print(f"[AGGREGATOR] Attente updates round={round_number} clients={expected_clients}")

    for message in consumer:
        event = message.value

        if event.get("event_type") != "fl_client_update":
            continue
        if int(event.get("round_number")) != round_number:
            continue

        client_id = event.get("client_id")
        if client_id not in expected_clients:
            continue

        state_dict = base64_to_state_dict(event["model_state_b64"])
        n_samples = int(event["n_samples"])
        received[client_id] = (state_dict, n_samples)

        print(
            f"[AGGREGATOR] Update recu | Round={round_number} | "
            f"Client={client_id} | Samples={n_samples} | "
            f"{len(received)}/{len(expected_clients)}"
        )

        if len(received) == len(expected_clients):
            return [received[c] for c in expected_clients]


def save_global_model(model, round_number: int, input_dim: int):
    torch.save(model.state_dict(), GLOBAL_MODEL_PATH)
    metadata = {
        "model_type": "PyTorch MLP Federated FedAvg",
        "round_number": round_number,
        "input_dim": input_dim,
        "hidden_dim": HIDDEN_DIM,
        "feature_columns": FEATURE_COLUMNS,
        "clients": CLIENTS,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    GLOBAL_METADATA_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[AGGREGATOR] Modele sauvegarde: {GLOBAL_MODEL_PATH}")
    print(f"[AGGREGATOR] Metadata sauvegardee: {GLOBAL_METADATA_PATH}")


def run_aggregator(num_rounds: int):
    input_dim = len(FEATURE_COLUMNS)
    model = build_model(input_dim=input_dim, hidden_dim=HIDDEN_DIM)

    producer = create_producer()
    consumer = create_update_consumer()

    print("[AGGREGATOR] Demarre")
    print("[AGGREGATOR] Clients:", CLIENTS)
    print("[AGGREGATOR] Features:", FEATURE_COLUMNS)
    print("[AGGREGATOR] Rounds:", num_rounds)

    for round_number in range(1, num_rounds + 1):
        publish_global_model(producer, model, round_number, input_dim)

        updates = collect_round_updates(
            consumer=consumer,
            round_number=round_number,
            expected_clients=CLIENTS,
        )

        global_state = fedavg(updates)
        model.load_state_dict(global_state)

        print(f"[AGGREGATOR] FedAvg termine pour round={round_number}")

    save_global_model(model, num_rounds, input_dim)
    print("[AGGREGATOR] Federated Learning termine.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregateur FL FedAvg via Kafka.")
    parser.add_argument("--rounds", type=int, default=NUM_ROUNDS)
    args = parser.parse_args()

    run_aggregator(num_rounds=args.rounds)
