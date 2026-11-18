"""Regresión lineal OLS con numpy y pandas únicamente."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def split_indices(n: int, seed: int, train_frac: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    """Partición determinista. Devuelve (train, heldout), ambos ordenados, disjuntos y completos."""
    if n < 0 or not 0.0 < train_frac < 1.0:
        raise ValueError("n debe ser >= 0 y train_frac estar en (0, 1)")
    perm = np.random.default_rng(seed).permutation(n)
    n_train = int(np.floor(n * train_frac + 1e-9))
    return np.sort(perm[:n_train]), np.sort(perm[n_train:])


@dataclass(frozen=True)
class Encoder:
    """Matriz de diseño: [intercepto, numéricas..., one-hot por categórica...].

    El vocabulario se fija con el entrenamiento; un valor fuera de él es un error.
    """

    numeric: tuple[str, ...]
    vocab: dict[str, tuple[str, ...]]

    @property
    def feature_names(self) -> list[str]:
        names = ["intercept", *self.numeric]
        for col, cats in self.vocab.items():
            names += [f"{col}={c}" for c in cats]
        return names

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        n = len(df)
        blocks = [np.ones((n, 1))]
        if self.numeric:
            num = df[list(self.numeric)].to_numpy(dtype=float)
            if not np.isfinite(num).all():
                bad = [c for c in self.numeric if not np.isfinite(df[c].to_numpy(dtype=float)).all()]
                raise ValueError(f"valor no numérico o nulo en: {', '.join(bad)}")
            blocks.append(num)
        for col, cats in self.vocab.items():
            codes = pd.Categorical(df[col], categories=list(cats)).codes
            if (codes < 0).any():
                raise ValueError(f"valor no visto en el entrenamiento en: {col}")
            onehot = np.zeros((n, len(cats)))
            onehot[np.arange(n), codes] = 1.0
            blocks.append(onehot)
        return np.hstack(blocks)


def fit_encoder(df: pd.DataFrame, numeric: list[str], categorical: list[str]) -> Encoder:
    """Construye el Encoder con las categorías (ordenadas) presentes en `df` de entrenamiento."""
    vocab = {c: tuple(sorted(df[c].dropna().astype(str).unique())) for c in categorical}
    return Encoder(numeric=tuple(numeric), vocab=vocab)


def fit_ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Coeficientes por mínimos cuadrados (numpy.linalg.lstsq, solución de norma mínima)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2 or y.ndim != 1 or X.shape[0] != y.shape[0] or X.shape[0] == 0:
        raise ValueError("X debe ser (n, p) e y (n,) con n >= 1")
    if not (np.isfinite(X).all() and np.isfinite(y).all()):
        raise ValueError("X e y no pueden contener NaN ni infinitos")
    return np.linalg.lstsq(X, y, rcond=None)[0]


def predict_ols(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return np.asarray(X, dtype=float) @ beta


def metrics(y_true, y_pred) -> dict[str, float]:
    """R² = 1 - SS_res/SS_tot y MAE. R² es NaN si la varianza de y_true es 0."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError("y_true e y_pred deben tener la misma forma y no estar vacías")
    resid = y_true - y_pred
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"r2": r2, "mae": float(np.mean(np.abs(resid)))}
