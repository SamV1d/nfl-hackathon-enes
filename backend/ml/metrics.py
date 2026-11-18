"""Métricas de clasificación probabilística y regresión (solo numpy)."""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def _arr(a) -> np.ndarray:
    return np.asarray(a, dtype=float)


def brier(y, p) -> float:
    y, p = _arr(y), _arr(p)
    return float(np.mean((p - y) ** 2))


def log_loss(y, p) -> float:
    y, p = _arr(y), np.clip(_arr(p), EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Rangos promedio (1..n) con empates."""
    order = np.argsort(a, kind="mergesort")
    sorted_a = a[order]
    ranks = np.empty(len(a), dtype=float)
    i = 0
    n = len(a)
    while i < n:
        j = i
        while j + 1 < n and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def roc_auc(y, score) -> float:
    """AUC por estadístico de Mann-Whitney. NaN si falta una de las clases."""
    y, s = _arr(y), _arr(score)
    n_pos = float((y == 1).sum())
    n_neg = float((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata(s)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def pr_auc(y, score) -> float:
    """Average precision (área bajo la curva precision-recall, paso a paso)."""
    y, s = _arr(y), _arr(score)
    n_pos = float((y == 1).sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    ys, ss = y[order], s[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    # Un punto por cada valor distinto de score (último índice de cada grupo).
    last = np.r_[np.where(np.diff(ss) != 0)[0], len(ss) - 1]
    precision = tp[last] / (tp[last] + fp[last])
    recall = tp[last] / n_pos
    prev_recall = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - prev_recall) * precision))


def precision_recall_at(y, p, threshold: float) -> dict[str, float]:
    y, p = _arr(y), _arr(p)
    pred = p >= threshold
    tp = float(np.sum(pred & (y == 1)))
    fp = float(np.sum(pred & (y == 0)))
    fn = float(np.sum(~pred & (y == 1)))
    precision = tp / (tp + fp) if tp + fp > 0 else float("nan")
    recall = tp / (tp + fn) if tp + fn > 0 else float("nan")
    return {"threshold": float(threshold), "precision": precision, "recall": recall}


def best_f1_threshold(y, p) -> float:
    """Umbral que maximiza F1. Debe elegirse con datos de validación, no de prueba."""
    y, p = _arr(y), _arr(p)
    best_t, best_f = 0.5, -1.0
    for t in np.unique(np.quantile(p, np.linspace(0.02, 0.98, 49))):
        m = precision_recall_at(y, p, t)
        if np.isnan(m["precision"]) or np.isnan(m["recall"]) or m["precision"] + m["recall"] == 0:
            continue
        f1 = 2 * m["precision"] * m["recall"] / (m["precision"] + m["recall"])
        if f1 > best_f:
            best_t, best_f = float(t), f1
    return best_t


def calibration_curve(y, p, n_bins: int = 10, min_count: int = 1) -> list[dict]:
    """Bins por cuantiles de probabilidad. Devuelve media predicha, tasa observada y n."""
    y, p = _arr(y), _arr(p)
    if len(p) == 0:
        return []
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        return [{"mean_predicted": float(p.mean()), "observed_rate": float(y.mean()), "n": int(len(p))}]
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    out = []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() >= min_count:
            out.append({"mean_predicted": float(p[m].mean()), "observed_rate": float(y[m].mean()),
                        "n": int(m.sum())})
    return out


def expected_calibration_error(y, p, n_bins: int = 10) -> float:
    curve = calibration_curve(y, p, n_bins)
    n = sum(c["n"] for c in curve)
    if n == 0:
        return float("nan")
    return float(sum(c["n"] * abs(c["mean_predicted"] - c["observed_rate"]) for c in curve) / n)


def classification_report(y, p, threshold: float) -> dict:
    return {
        "brier": brier(y, p),
        "log_loss": log_loss(y, p),
        "roc_auc": roc_auc(y, p),
        "pr_auc": pr_auc(y, p),
        "ece": expected_calibration_error(y, p),
        **precision_recall_at(y, p, threshold),
        "base_rate": float(_arr(y).mean()),
        "n": int(len(_arr(y))),
    }


def mae(y, yhat) -> float:
    return float(np.mean(np.abs(_arr(y) - _arr(yhat))))


def rmse(y, yhat) -> float:
    return float(np.sqrt(np.mean((_arr(y) - _arr(yhat)) ** 2)))


def r2_score(y, yhat) -> float:
    """R² respecto a la media de `y` evaluada. NaN si la varianza es cero.
    Con un modelo base entrenado en otro periodo puede ser negativo."""
    y, yhat = _arr(y), _arr(yhat)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot == 0:
        return float("nan")
    return float(1.0 - np.sum((y - yhat) ** 2) / ss_tot)


def regression_report(y, yhat) -> dict:
    return {"mae": mae(y, yhat), "rmse": rmse(y, yhat), "r2": r2_score(y, yhat),
            "n": int(len(_arr(y)))}
