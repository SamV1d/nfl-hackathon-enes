"""Auditoría y selección de modelos con validación temporal (por semana).

Protocolo:
* entrenamiento: semanas 1-5; validación: semana 6 (elige modelo, hiperparámetros y umbral);
  prueba: semanas 7-8 (se evalúa una sola vez con el modelo elegido).
* Todo modelo se compara con un modelo base (tasa o media de entrenamiento).
* La selección usa solo validación. Ante empate técnico gana el modelo más simple.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np
import pandas as pd

from . import metrics as M
from .estimators import GradientBoosting, LogisticRegression, MeanBaseline, RandomForest, RidgeRegression
from .features import (CATEGORICAL, NUMERIC, FeatureEncoder, build_ml_frame, temporal_split,
                       used_features_by_timing)

# Criterios (decisiones de política del producto, documentadas en docs/ML.md).
TIE_TOLERANCE_LOGLOSS = 0.002
TIE_TOLERANCE_RMSE = 0.02
MIN_AUC_LOWER_BOUND = 0.65   # discriminación mínima para llamar "validated" a un clasificador
MIN_ECE = 0.05
MIN_R2_VALIDATED = 0.10
N_BOOT = 300
SEED = 7

# Orden = simplicidad creciente (para desempates).
CLASSIFIERS: dict[str, list[tuple[str, Callable]]] = {
    "baseline": [("tasa base", lambda: MeanBaseline())],
    "logistic": [(f"l2={l}", (lambda l=l: LogisticRegression(l2=l))) for l in (1.0, 10.0, 100.0)],
    "random_forest": [(f"depth={d}", (lambda d=d: RandomForest(n_trees=80, max_depth=d, min_leaf=40)))
                      for d in (4, 6)],
    "gradient_boosting": [(f"depth={d},trees={t}", (lambda d=d, t=t: GradientBoosting(
        "logistic", n_trees=t, max_depth=d, learning_rate=0.05))) for d in (1, 2) for t in (100, 250)],
}
REGRESSORS: dict[str, list[tuple[str, Callable]]] = {
    "baseline": [("media", lambda: MeanBaseline())],
    "ridge": [(f"alpha={a}", (lambda a=a: RidgeRegression(alpha=a))) for a in (10.0, 100.0, 1000.0)],
    "random_forest": [(f"depth={d}", (lambda d=d: RandomForest(n_trees=80, max_depth=d, min_leaf=60)))
                      for d in (3, 5)],
    "gradient_boosting": [(f"depth={d},trees={t}", (lambda d=d, t=t: GradientBoosting(
        "squared", n_trees=t, max_depth=d, learning_rate=0.03, min_leaf=60))) for d in (1, 2) for t in (100, 250)],
}


def _clean(o):
    """Convierte NaN/inf y tipos numpy a JSON válido."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


def _bootstrap(y, p, p_base, rng, n_boot=N_BOOT) -> dict:
    """IC 95% por remuestreo de jugadas: AUC y mejora de Brier frente al modelo base."""
    n = len(y)
    aucs, skills = [], []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        a = M.roc_auc(y[i], p[i])
        if math.isfinite(a):
            aucs.append(a)
        b_base = M.brier(y[i], p_base[i])
        skills.append(1.0 - M.brier(y[i], p[i]) / b_base if b_base > 0 else float("nan"))
    q = lambda v: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
    return {"auc_ci95": q(aucs), "brier_skill_ci95": q(skills)}


def _fit_predict(factory, Xtr, ytr, Xs: dict[str, np.ndarray]):
    model = factory().fit(Xtr, ytr)
    return model, {k: np.asarray(model.predict(v), dtype=float) for k, v in Xs.items()}


def classification_audit(frame: pd.DataFrame, parts: dict, target: str = "pressure",
                         numeric=None, categorical=None) -> dict:
    """Compara clasificadores y devuelve resultados de validación, prueba y veredicto."""
    numeric = list(NUMERIC if numeric is None else numeric)
    categorical = list(CATEGORICAL if categorical is None else categorical)
    tr, va, te = (frame.iloc[parts[k]] for k in ("train", "valid", "test"))
    enc = FeatureEncoder(numeric, categorical).fit(tr)
    X = {"train": enc.transform(tr), "valid": enc.transform(va), "test": enc.transform(te)}
    y = {k: v[target].to_numpy(dtype=float) for k, v in (("train", tr), ("valid", va), ("test", te))}

    candidates = []
    for family, configs in CLASSIFIERS.items():
        for cfg_name, factory in configs:
            model, preds = _fit_predict(factory, X["train"], y["train"], {"valid": X["valid"], "test": X["test"]})
            candidates.append({
                "family": family, "config": cfg_name, "factory": factory, "model": model,
                "valid_log_loss": M.log_loss(y["valid"], preds["valid"]),
                "valid_brier": M.brier(y["valid"], preds["valid"]),
                "valid_auc": M.roc_auc(y["valid"], preds["valid"]),
                "preds": preds,
            })
    base = next(c for c in candidates if c["family"] == "baseline")
    # mejor configuración por familia (validación), luego elección entre familias con desempate por simplicidad
    best_by_family = {}
    for c in candidates:
        cur = best_by_family.get(c["family"])
        if cur is None or c["valid_log_loss"] < cur["valid_log_loss"]:
            best_by_family[c["family"]] = c
    order = list(CLASSIFIERS)
    min_ll = min(c["valid_log_loss"] for c in best_by_family.values())
    near = [c for c in best_by_family.values() if c["valid_log_loss"] <= min_ll + TIE_TOLERANCE_LOGLOSS]
    champion = sorted(near, key=lambda c: order.index(c["family"]))[0]

    thr = M.best_f1_threshold(y["valid"], champion["preds"]["valid"])
    test_p, test_base = champion["preds"]["test"], base["preds"]["test"]
    test_rep = M.classification_report(y["test"], test_p, thr)
    base_rep = M.classification_report(y["test"], test_base, thr)
    ci = _bootstrap(y["test"], test_p, test_base, np.random.default_rng(SEED))
    skill = 1.0 - test_rep["brier"] / base_rep["brier"]

    reasons = []
    if champion["family"] == "baseline":
        reasons.append("Ningún modelo supera a la tasa base en validación.")
    if ci["auc_ci95"][0] < MIN_AUC_LOWER_BOUND:
        reasons.append(f"Baja capacidad de discriminación: el límite inferior del IC95% del ROC-AUC en "
                       f"prueba ({ci['auc_ci95'][0]:.3f}) es menor que {MIN_AUC_LOWER_BOUND}.")
    if ci["brier_skill_ci95"][0] <= 0:
        reasons.append("La mejora de Brier frente a la tasa base no es distinguible de cero en prueba.")
    if test_rep["ece"] > MIN_ECE:
        reasons.append(f"Error de calibración esperado ({test_rep['ece']:.3f}) mayor que {MIN_ECE}.")
    if not reasons:
        status = "validated"
    elif ci["brier_skill_ci95"][0] > 0 and champion["family"] != "baseline":
        status = "limited"   # mejor que la tasa base, pero con poca capacidad de discriminar o mal calibrado
    else:
        status = "not_reliable"

    n_pos = int(y["train"].sum())
    return _clean({
        "target": target,
        "features": {"numeric": numeric, "categorical": categorical},
        "n": {k: int(len(v)) for k, v in y.items()},
        "class_balance": {"train_positive_rate": float(y["train"].mean()),
                          "valid_positive_rate": float(y["valid"].mean()),
                          "test_positive_rate": float(y["test"].mean()),
                          "train_positives": n_pos},
        "candidates": [{k: c[k] for k in ("family", "config", "valid_log_loss", "valid_brier", "valid_auc")}
                       for c in candidates],
        "champion": {"family": champion["family"], "config": champion["config"]},
        "threshold_from_validation": thr,
        "test": test_rep, "test_baseline": base_rep, "brier_skill_vs_baseline": skill,
        "confidence_intervals": ci,
        "calibration_curve": M.calibration_curve(y["test"], test_p, 10),
        "status": status, "reasons": reasons,
    })


def regression_audit(frame: pd.DataFrame, parts: dict, target: str, numeric=None, categorical=None) -> dict:
    numeric = list(NUMERIC if numeric is None else numeric)
    categorical = list(CATEGORICAL if categorical is None else categorical)
    ok = frame[target].notna().to_numpy()
    p = {k: v[ok[v]] for k, v in parts.items()}
    tr, va, te = (frame.iloc[p[k]] for k in ("train", "valid", "test"))
    enc = FeatureEncoder(numeric, categorical).fit(tr)
    X = {"train": enc.transform(tr), "valid": enc.transform(va), "test": enc.transform(te)}
    y = {k: v[target].to_numpy(dtype=float) for k, v in (("train", tr), ("valid", va), ("test", te))}
    candidates = []
    for family, configs in REGRESSORS.items():
        for cfg_name, factory in configs:
            model, preds = _fit_predict(factory, X["train"], y["train"], {"valid": X["valid"], "test": X["test"]})
            candidates.append({"family": family, "config": cfg_name, "preds": preds,
                               "valid_rmse": M.rmse(y["valid"], preds["valid"]),
                               "valid_mae": M.mae(y["valid"], preds["valid"])})
    base = next(c for c in candidates if c["family"] == "baseline")
    best = {}
    for c in candidates:
        if c["family"] not in best or c["valid_rmse"] < best[c["family"]]["valid_rmse"]:
            best[c["family"]] = c
    order = list(REGRESSORS)
    min_r = min(c["valid_rmse"] for c in best.values())
    near = [c for c in best.values() if c["valid_rmse"] <= min_r + TIE_TOLERANCE_RMSE]
    champion = sorted(near, key=lambda c: order.index(c["family"]))[0]

    test_rep = M.regression_report(y["test"], champion["preds"]["test"])
    base_rep = M.regression_report(y["test"], base["preds"]["test"])
    skill = 1.0 - test_rep["rmse"] ** 2 / base_rep["rmse"] ** 2  # mejora de MSE frente a la media de entrenamiento
    rng = np.random.default_rng(SEED)
    n = len(y["test"])
    skills = []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        mb = np.mean((y["test"][i] - base["preds"]["test"][i]) ** 2)
        skills.append(1.0 - np.mean((y["test"][i] - champion["preds"]["test"][i]) ** 2) / mb)
    ci = [float(np.percentile(skills, 2.5)), float(np.percentile(skills, 97.5))]
    reasons = []
    if champion["family"] == "baseline":
        reasons.append("Ningún modelo supera a la media de entrenamiento en validación.")
    if ci[0] <= 0:
        reasons.append("La mejora de MSE frente a la media de entrenamiento no es distinguible de cero en prueba.")
    if test_rep["r2"] is None or not math.isfinite(test_rep["r2"]) or test_rep["r2"] < MIN_R2_VALIDATED:
        reasons.append(f"R² en prueba ({test_rep['r2']:.3f}) menor que {MIN_R2_VALIDATED}: explica poca variación.")
    status = "validated" if not reasons else ("limited" if ci[0] > 0 else "not_reliable")
    return _clean({
        "target": target, "n": {k: int(len(v)) for k, v in y.items()},
        "target_std_test": float(np.std(y["test"])),
        "candidates": [{k: c[k] for k in ("family", "config", "valid_rmse", "valid_mae")} for c in candidates],
        "champion": {"family": champion["family"], "config": champion["config"]},
        "test": test_rep, "test_baseline": base_rep, "mse_skill_vs_baseline": skill,
        "mse_skill_ci95": ci, "status": status, "reasons": reasons,
    })


def run_audit(dataset: pd.DataFrame, plays: pd.DataFrame, games: pd.DataFrame) -> dict:
    """Auditoría completa. Es determinista (semillas fijas)."""
    frame = build_ml_frame(dataset, plays, games)
    parts = temporal_split(frame)
    context_only = [c for c in NUMERIC if c != "passRushers"]
    return _clean({
        "protocol": {
            "split": "temporal por semana", "train_weeks": [1, 2, 3, 4, 5], "valid_weeks": [6],
            "test_weeks": [7, 8],
            "n_plays": {k: int(len(v)) for k, v in parts.items()},
        },
        "feature_timing": used_features_by_timing(),
        "pressure_with_defensive_choice": classification_audit(frame, parts, "pressure"),
        "pressure_pre_snap_only": classification_audit(
            frame, parts, "pressure", numeric=[c for c in context_only], categorical=["offenseFormation"]),
        "sack": classification_audit(frame, parts, "sack"),
        "yards_net": regression_audit(frame, parts, "netYards"),
        "yards_on_attempts": regression_audit(frame, parts, "attemptYards"),
    })
