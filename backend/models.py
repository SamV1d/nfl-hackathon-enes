"""Pressure_Model y Yards_Model: dos regresiones lineales encadenadas (numpy y pandas)."""
from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .linreg import Encoder, fit_encoder, fit_ols, metrics, predict_ols

MIN_LABELED = 100
MAX_DOWN, MAX_YARDS_TO_GO, MAX_PLAYERS = 4, 99, 11

PRESSURE_NUMERIC = ("down", "yardsToGo", "passRushers", "defendersInBox")
YARDS_NUMERIC = ("pressureProb", "down", "yardsToGo")
CATEGORICAL = ("offenseFormation", "expectedCoverage")


class ModelUnavailable(Exception):
    """El ajuste falló o no hay datos suficientes."""


class InvalidSituation(ValueError):
    """Situación inválida; `fields` nombra cada campo inválido."""

    def __init__(self, fields: list[str]):
        self.fields = fields
        super().__init__("campos inválidos: " + ", ".join(fields))


@dataclass(frozen=True)
class Situation:
    down: int
    yards_to_go: int
    offense_formation: str
    expected_coverage: str
    pass_rushers: int | None = None
    defenders_in_box: int | None = None


def _num(v) -> bool:
    return isinstance(v, numbers.Real) and not isinstance(v, bool) and math.isfinite(v)


def validate_situation(s: Situation, vocab: dict[str, Sequence[str]], require_call: bool = True) -> list[str]:
    """Nombres de los campos inválidos (lista vacía si es válida).

    `require_call` exige pass_rushers y defenders_in_box (features del Pressure_Model).
    """
    bad: list[str] = []
    if not (isinstance(s.down, numbers.Integral) and not isinstance(s.down, bool)
            and 0 <= s.down <= MAX_DOWN):
        bad.append("down")
    if not (isinstance(s.yards_to_go, numbers.Integral) and not isinstance(s.yards_to_go, bool)
            and 0 <= s.yards_to_go <= MAX_YARDS_TO_GO):
        bad.append("yards_to_go")
    if not (isinstance(s.offense_formation, str) and s.offense_formation in vocab["offenseFormation"]):
        bad.append("offense_formation")
    if not (isinstance(s.expected_coverage, str) and s.expected_coverage in vocab["expectedCoverage"]):
        bad.append("expected_coverage")
    for name in ("pass_rushers", "defenders_in_box"):
        v = getattr(s, name)
        if v is None:
            if require_call:
                bad.append(name)
        elif not (_num(v) and 0 <= v <= MAX_PLAYERS):
            bad.append(name)
    return bad


def situation_frame(items: Sequence[Situation]) -> pd.DataFrame:
    """Columnas con los nombres del dataset, una fila por situación."""
    return pd.DataFrame({
        "down": [i.down for i in items],
        "yardsToGo": [i.yards_to_go for i in items],
        "offenseFormation": [i.offense_formation for i in items],
        "expectedCoverage": [i.expected_coverage for i in items],
        "passRushers": [np.nan if i.pass_rushers is None else i.pass_rushers for i in items],
        "defendersInBox": [np.nan if i.defenders_in_box is None else i.defenders_in_box for i in items],
    })


def _vocab_lists(enc: Encoder) -> dict[str, list[str]]:
    return {k: list(v) for k, v in enc.vocab.items()}


@dataclass(frozen=True)
class PressureModel:
    encoder: Encoder
    beta: np.ndarray
    _metrics: dict

    @property
    def vocab(self) -> dict[str, list[str]]:
        return _vocab_lists(self.encoder)

    def predict_frame(self, df: pd.DataFrame) -> np.ndarray:
        """Probabilidades recortadas a [0, 1] (sin validar; el llamador valida)."""
        return np.clip(predict_ols(self.encoder.transform(df), self.beta), 0.0, 1.0)

    def predict(self, s: Situation) -> float:
        bad = validate_situation(s, self.vocab, require_call=True)
        if bad:
            raise InvalidSituation(bad)
        return float(self.predict_frame(situation_frame([s]))[0])

    def metrics(self) -> dict:
        return dict(self._metrics)


@dataclass(frozen=True)
class YardsModel:
    encoder: Encoder
    beta: np.ndarray
    pressure_model: PressureModel
    _metrics: dict

    @property
    def vocab(self) -> dict[str, list[str]]:
        return _vocab_lists(self.encoder)

    @property
    def pressure_coef(self) -> float:
        return float(self.beta[1 + YARDS_NUMERIC.index("pressureProb")])

    def predict_raw_frame(self, df: pd.DataFrame, pressure: np.ndarray) -> np.ndarray:
        x = df.copy()
        x["pressureProb"] = np.asarray(pressure, dtype=float)
        return predict_ols(self.encoder.transform(x), self.beta)

    def predict_raw(self, s: Situation, pressure: float) -> float:
        """Yardas esperadas sin redondear, dada la probabilidad de presión."""
        bad = validate_situation(s, self.vocab, require_call=False)
        if bad:
            raise InvalidSituation(bad)
        return float(self.predict_raw_frame(situation_frame([s]), np.array([pressure]))[0])

    def predict(self, s: Situation) -> float:
        """Yardas esperadas (2 decimales) usando la salida del Pressure_Model."""
        return round(self.predict_raw(s, self.pressure_model.predict(s)), 2)

    def metrics(self) -> dict:
        return dict(self._metrics)


@dataclass(frozen=True)
class FitResult:
    pressure_model: PressureModel | None
    yards_model: YardsModel | None
    errors: dict[str, str]  # "pressure" / "yards" -> motivo


def _readonly(a: np.ndarray) -> np.ndarray:
    a = np.array(a, dtype=float)
    a.setflags(write=False)
    return a


def _usable(df: pd.DataFrame, numeric: Sequence[str], cats: Sequence[str]) -> np.ndarray:
    ok = np.ones(len(df), dtype=bool)
    for c in numeric:
        ok &= np.isfinite(pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float))
    for c in cats:
        ok &= df[c].notna().to_numpy()
    return ok


def _seen(df: pd.DataFrame, enc: Encoder) -> np.ndarray:
    ok = np.ones(len(df), dtype=bool)
    for c, cats in enc.vocab.items():
        ok &= df[c].isin(cats).to_numpy()
    return ok


def _split_counts(train_idx, heldout_idx, n_train_used, n_ho_scored) -> dict:
    return {"n_train": int(len(train_idx)), "n_heldout": int(len(heldout_idx)),
            "n_train_used": int(n_train_used), "n_heldout_scored": int(n_ho_scored)}


def fit_pressure_model(dataset: pd.DataFrame, train_idx: np.ndarray, heldout_idx: np.ndarray) -> PressureModel:
    """Ajusta el Pressure_Model con las filas train_idx. Lanza ModelUnavailable si no se puede."""
    try:
        ok = _usable(dataset, [*PRESSURE_NUMERIC, "pressure"], CATEGORICAL)
        if int(ok.sum()) < MIN_LABELED:
            raise ModelUnavailable(f"menos de {MIN_LABELED} jugadas etiquetadas ({int(ok.sum())})")
        tr = dataset.iloc[train_idx][ok[train_idx]]
        if tr.empty:
            raise ModelUnavailable("sin filas de entrenamiento")
        enc = fit_encoder(tr, list(PRESSURE_NUMERIC), list(CATEGORICAL))
        beta = _readonly(fit_ols(enc.transform(tr), tr["pressure"].to_numpy(dtype=float)))

        ho = dataset.iloc[heldout_idx][ok[heldout_idx]]
        ho = ho[_seen(ho, enc)]
        if ho.empty:
            raise ModelUnavailable("sin filas reservadas evaluables")
        model = PressureModel(enc, beta, {})
        m = metrics(ho["pressure"].to_numpy(dtype=float), model.predict_frame(ho))
        info = {"r2": m["r2"], "mae": m["mae"], **_split_counts(train_idx, heldout_idx, len(tr), len(ho))}
        return PressureModel(enc, beta, info)
    except ModelUnavailable:
        raise
    except (ValueError, KeyError, np.linalg.LinAlgError) as exc:
        raise ModelUnavailable(f"fallo de ajuste: {type(exc).__name__}") from None


def fit_yards_model(dataset: pd.DataFrame, train_idx: np.ndarray, heldout_idx: np.ndarray,
                    pressure_model: PressureModel) -> YardsModel:
    """Ajusta el Yards_Model sobre la presión predicha (recortada) por el Pressure_Model."""
    try:
        ok = _usable(dataset, [*PRESSURE_NUMERIC, "yardsGained"], CATEGORICAL)
        tr = dataset.iloc[train_idx][ok[train_idx]]
        if tr.empty:
            raise ModelUnavailable("sin filas de entrenamiento")
        ptr = pressure_model.predict_frame(tr)
        x = tr.assign(pressureProb=ptr)
        enc = fit_encoder(x, list(YARDS_NUMERIC), list(CATEGORICAL))
        beta = _readonly(fit_ols(enc.transform(x), tr["yardsGained"].to_numpy(dtype=float)))

        ho = dataset.iloc[heldout_idx][ok[heldout_idx]]
        ho = ho[_seen(ho, enc) & _seen(ho, pressure_model.encoder)]
        if ho.empty:
            raise ModelUnavailable("sin filas reservadas evaluables")
        model = YardsModel(enc, beta, pressure_model, {})
        pred = model.predict_raw_frame(ho, pressure_model.predict_frame(ho))
        m = metrics(ho["yardsGained"].to_numpy(dtype=float), pred)
        info = {"r2": m["r2"], "mae_yards": m["mae"],
                **_split_counts(train_idx, heldout_idx, len(tr), len(ho))}
        return YardsModel(enc, beta, pressure_model, info)
    except ModelUnavailable:
        raise
    except (ValueError, KeyError, np.linalg.LinAlgError) as exc:
        raise ModelUnavailable(f"fallo de ajuste: {type(exc).__name__}") from None


def fit_models(dataset: pd.DataFrame, train_idx: np.ndarray, heldout_idx: np.ndarray) -> FitResult:
    """Ajusta ambos modelos con la misma partición. Los fallos se registran, no se propagan."""
    errors: dict[str, str] = {}
    pm = ym = None
    try:
        pm = fit_pressure_model(dataset, train_idx, heldout_idx)
    except ModelUnavailable as exc:
        errors["pressure"] = str(exc)
    if pm is not None:
        try:
            ym = fit_yards_model(dataset, train_idx, heldout_idx, pm)
        except ModelUnavailable as exc:
            errors["yards"] = str(exc)
    else:
        errors["yards"] = "Pressure_Model no disponible"
    return FitResult(pm, ym, errors)
