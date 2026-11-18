"""Limpieza de plays.csv. Funciones puras: no mutan la entrada."""
from __future__ import annotations

import numpy as np
import pandas as pd

UNKNOWN = "Unknown"


def train_median_defenders_in_box(plays: pd.DataFrame, train_idx: np.ndarray) -> float:
    """Mediana de defendersInBox no nulo, solo de las filas de entrenamiento.

    `train_idx` son posiciones (iloc) dentro de `plays`.
    """
    values = plays["defendersInBox"].iloc[np.asarray(train_idx, dtype=int)].dropna()
    return float(values.median()) if len(values) else float("nan")


def clean_plays(plays: pd.DataFrame, train_median_box: float) -> pd.DataFrame:
    """down=0 y quarter=5 se conservan sin cambio y se etiquetan.

    Nulos de dropBackType y offenseFormation -> "Unknown".
    Nulos de defendersInBox -> train_median_box.
    """
    out = plays.copy()
    out["is_two_point"] = out["down"] == 0
    out["is_overtime"] = out["quarter"] == 5
    for col in ("dropBackType", "offenseFormation"):
        out[col] = out[col].fillna(UNKNOWN)
    out["defendersInBox"] = out["defendersInBox"].fillna(train_median_box)
    return out
