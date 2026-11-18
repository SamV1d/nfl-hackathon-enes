"""Recomendador: evalúa cada Candidate_Call observada y elige la Best_Call."""
from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from .models import InvalidSituation, Situation, validate_situation

LOW_CONFIDENCE_BELOW = 30


class PredictionUnavailable(Exception):
    """No se pudieron producir predicciones (modelos ausentes o error numérico)."""


@dataclass(frozen=True, order=True)
class CandidateCall:
    coverage: str
    pass_rushers: int
    defenders_in_box: float


@dataclass(frozen=True)
class ScoredCall:
    call: CandidateCall
    pressure_probability: float
    expected_yards: float


@dataclass(frozen=True)
class Recommendation:
    best_call: ScoredCall
    ranked: list[ScoredCall]
    typical_call: CandidateCall | None
    yards_delta_vs_typical: float | None
    two_point_conversion: bool
    low_confidence: bool
    n_matches: int
    recommendation_id: str
    situation: Situation

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _num(v: float) -> int | float:
    f = float(v)
    return int(f) if f.is_integer() else f


def candidate_calls(train: pd.DataFrame) -> list[CandidateCall]:
    """Combinaciones distintas (cobertura, rushers, defensores en la caja) vistas en `train`."""
    cols = ["actualCoverage", "passRushers", "defendersInBox"]
    distinct = train[cols].dropna().drop_duplicates()
    calls = {CandidateCall(str(c), int(r), _num(b)) for c, r, b in distinct.itertuples(index=False)}
    return sorted(calls)


def rank_calls(scored: list[ScoredCall]) -> list[ScoredCall]:
    """Mayor probabilidad de presión primero; empate: menos yardas esperadas; luego orden estable."""
    return sorted(scored, key=lambda x: (-x.pressure_probability, x.expected_yards, x.call))


def score_calls(ctx, s: Situation) -> list[ScoredCall]:
    """Probabilidad y yardas (sin redondear) de cada candidata para la situación `s`.

    La cobertura de la candidata alimenta solo al Pressure_Model; el Yards_Model usa la
    cobertura esperada de la situación.
    """
    pm, ym = ctx.pressure_model, ctx.yards_model
    if pm is None or ym is None:
        raise PredictionUnavailable("modelo no disponible")
    cands = ctx.candidates
    if not cands:
        raise PredictionUnavailable("sin llamadas candidatas")
    try:
        frame = pd.DataFrame({
            "down": s.down,
            "yardsToGo": s.yards_to_go,
            "offenseFormation": s.offense_formation,
            "expectedCoverage": [c.coverage for c in cands],
            "passRushers": [c.pass_rushers for c in cands],
            "defendersInBox": [c.defenders_in_box for c in cands],
        })
        p = pm.predict_frame(frame)
        y = ym.predict_raw_frame(frame.assign(expectedCoverage=s.expected_coverage), p)
    except (ValueError, KeyError, np.linalg.LinAlgError) as exc:
        raise PredictionUnavailable(type(exc).__name__) from None
    if not (np.isfinite(p).all() and np.isfinite(y).all()):
        raise PredictionUnavailable("predicción no finita")
    return [ScoredCall(c, float(pi), float(yi)) for c, pi, yi in zip(cands, p, y)]


def typical_call(train: pd.DataFrame, s: Situation) -> tuple[CandidateCall | None, int]:
    """Llamada más frecuente entre las jugadas de entrenamiento con la misma situación.

    Devuelve (llamada o None, número de jugadas parecidas). Empate: la llamada menor.
    """
    m = train[
        (train["down"] == s.down)
        & (train["yardsToGo"] == s.yards_to_go)
        & (train["offenseFormation"] == s.offense_formation)
        & (train["expectedCoverage"] == s.expected_coverage)
    ]
    n = len(m)
    calls = m[["actualCoverage", "passRushers", "defendersInBox"]].dropna()
    if calls.empty:
        return None, n
    counts = calls.value_counts()
    top = counts[counts == counts.max()].index
    best = min(CandidateCall(str(c), int(r), _num(b)) for c, r, b in top)
    return best, n


def recommend(ctx, s: Situation) -> Recommendation:
    """`ctx` expone pressure_model, yards_model, candidates, dataset y train_idx (p. ej. AppState).

    Lanza InvalidSituation (campos inválidos) o PredictionUnavailable. Sin Best_Call en ambos casos.
    """
    if ctx.pressure_model is None or ctx.yards_model is None:
        raise PredictionUnavailable("modelo no disponible")
    bad = validate_situation(replace(s, pass_rushers=None, defenders_in_box=None),
                             ctx.pressure_model.vocab, require_call=False)
    if bad:
        raise InvalidSituation(bad)

    ranked_raw = rank_calls(score_calls(ctx, s))
    ranked = [ScoredCall(r.call, r.pressure_probability, round(r.expected_yards, 1)) for r in ranked_raw]
    best = ranked[0]

    train = ctx.dataset.iloc[ctx.train_idx]
    typical, n_matches = typical_call(train, s)
    delta = None
    if typical is not None:
        t = next((r for r in ranked if r.call == typical), None)
        if t is not None:
            delta = round(best.expected_yards - t.expected_yards, 1)

    return Recommendation(
        best_call=best,
        ranked=ranked,
        typical_call=typical,
        yards_delta_vs_typical=delta,
        two_point_conversion=s.down == 0,
        low_confidence=n_matches < LOW_CONFIDENCE_BELOW,
        n_matches=n_matches,
        recommendation_id=uuid.uuid4().hex,
        situation=s,
    )
