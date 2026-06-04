"""
Modele PyTorch + helpers FedAvg pour Federated Learning via Kafka.
"""

from __future__ import annotations

import base64
import io
import json
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn


class EnergyMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def state_dict_to_base64(state_dict: Dict[str, torch.Tensor]) -> str:
    """Serialize un state_dict PyTorch en base64 pour Kafka JSON."""
    buffer = io.BytesIO()
    cpu_state = {k: v.detach().cpu() for k, v in state_dict.items()}
    torch.save(cpu_state, buffer)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def base64_to_state_dict(payload: str) -> Dict[str, torch.Tensor]:
    raw = base64.b64decode(payload.encode("utf-8"))
    buffer = io.BytesIO(raw)
    return torch.load(buffer, map_location="cpu")


def build_model(input_dim: int, hidden_dim: int = 64) -> EnergyMLP:
    return EnergyMLP(input_dim=input_dim, hidden_dim=hidden_dim)


def fedavg(updates: List[Tuple[Dict[str, torch.Tensor], int]]) -> Dict[str, torch.Tensor]:
    """
    Moyenne ponderee des poids locaux.
    updates = [(state_dict_client, n_samples_client), ...]
    """
    if not updates:
        raise ValueError("Aucun update client recu pour FedAvg.")

    total_samples = sum(n for _, n in updates)
    if total_samples <= 0:
        raise ValueError("Nombre total d'echantillons invalide.")

    global_state = {}
    keys = updates[0][0].keys()

    for key in keys:
        weighted_sum = None
        for state_dict, n_samples in updates:
            tensor = state_dict[key].float() * (n_samples / total_samples)
            weighted_sum = tensor if weighted_sum is None else weighted_sum + tensor
        global_state[key] = weighted_sum

    return global_state


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {"mae": mae, "rmse": rmse, "r2": r2}


def to_jsonable_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    return {k: float(v) for k, v in metrics.items()}
