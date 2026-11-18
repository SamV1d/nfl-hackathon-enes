"""Propiedades 12, 13 y 14: recomendador."""
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from backend.linreg import split_indices
from backend.models import InvalidSituation, Situation, fit_models
from backend.recommender import (CandidateCall, PredictionUnavailable, candidate_calls, rank_calls,
                                 recommend, ScoredCall)

from .fixtures import model_dataset


def make_ctx(seed: int, concentrated: bool):
    """`concentrated` reduce las combinaciones de situación para que haya 30 o más coincidencias."""
    rng = np.random.default_rng(seed)
    ds = model_dataset(rng, 500)
    if concentrated:
        ds["down"] = rng.choice([0, 1], len(ds))
        ds["yardsToGo"] = rng.choice([5, 10], len(ds))
        ds["offenseFormation"] = rng.choice(["SHOTGUN", "EMPTY"], len(ds))
        ds["expectedCoverage"] = ds["actualCoverage"] = rng.choice(["Cover-1", "Cover-3"], len(ds))
    tr, ho = split_indices(len(ds), seed)
    fit = fit_models(ds, tr, ho)
    ctx = SimpleNamespace(pressure_model=fit.pressure_model, yards_model=fit.yards_model, dataset=ds,
                          train_idx=tr, candidates=candidate_calls(ds.iloc[tr]))
    return ctx, rng


def pick_situation(ctx, rng) -> Situation:
    """Situación tomada de una fila de entrenamiento (con coincidencias) o aleatoria."""
    row = ctx.dataset.iloc[int(rng.choice(ctx.train_idx))]
    return Situation(int(row.down), int(row.yardsToGo), row.offenseFormation, row.expectedCoverage)


def matching(ctx, s):
    tr = ctx.dataset.iloc[ctx.train_idx]
    return [r for r in tr.itertuples()
            if (r.down, r.yardsToGo, r.offenseFormation, r.expectedCoverage)
            == (s.down, s.yards_to_go, s.offense_formation, s.expected_coverage)]


@pytest.mark.parametrize("concentrated", [False, True])
def test_property_12_ranking_order_and_best_call(concentrated):
    """Feature: coach-pass-rush-assistant, Property 12: Ranking order and Best_Call"""
    for seed in range(50):
        ctx, rng = make_ctx(seed, concentrated)
        rec = recommend(ctx, pick_situation(ctx, rng))
        assert sorted(r.call for r in rec.ranked) == sorted(ctx.candidates)
        assert len({r.call for r in rec.ranked}) == len(rec.ranked)
        keys = [(-r.pressure_probability, r.expected_yards) for r in rec.ranked]
        assert keys == sorted(keys)
        assert rec.best_call == rec.ranked[0]


def test_rank_calls_breaks_ties_by_lower_yards():
    a, b, c = (CandidateCall("Cover-1", 4, 6), CandidateCall("Cover-1", 5, 6), CandidateCall("Cover-3", 4, 6))
    ranked = rank_calls([ScoredCall(a, 0.5, 7.0), ScoredCall(b, 0.5, 3.0), ScoredCall(c, 0.9, 20.0)])
    assert [r.call for r in ranked] == [c, b, a]


@pytest.mark.parametrize("concentrated", [False, True])
def test_property_13_typical_call_and_yards_delta(concentrated):
    """Feature: coach-pass-rush-assistant, Property 13: Typical call and yards delta"""
    for seed in range(50):
        ctx, rng = make_ctx(seed, concentrated)
        s = pick_situation(ctx, rng)
        rec = recommend(ctx, s)
        counts = Counter(CandidateCall(r.actualCoverage, int(r.passRushers), int(r.defendersInBox))
                         for r in matching(ctx, s))
        assert counts, "la situación sale de una fila de entrenamiento"
        top = max(counts.values())
        assert rec.typical_call == min(c for c, n in counts.items() if n == top)
        typ = next(r for r in rec.ranked if r.call == rec.typical_call)
        assert rec.yards_delta_vs_typical == round(rec.best_call.expected_yards - typ.expected_yards, 1)


def test_property_14_situation_flags():
    """Feature: coach-pass-rush-assistant, Property 14: Situation flags"""
    seen_low = seen_high = seen_zero = False
    for seed in range(60):
        ctx, rng = make_ctx(seed, concentrated=seed % 2 == 0)
        s = pick_situation(ctx, rng) if seed % 3 else Situation(0, 5, "SHOTGUN", "Cover-1")
        rec = recommend(ctx, s)
        n = len(matching(ctx, s))
        assert rec.two_point_conversion == (s.down == 0)
        assert rec.low_confidence == (n < 30) and rec.n_matches == n
        assert len(rec.ranked) == len(ctx.candidates) > 0
        seen_low |= rec.low_confidence
        seen_high |= not rec.low_confidence
        seen_zero |= rec.two_point_conversion
    assert seen_low and seen_high and seen_zero


def test_no_matches_gives_no_typical_call():
    ctx, _ = make_ctx(1, False)
    rec = recommend(ctx, Situation(4, 99, "SHOTGUN", "Cover-1"))
    assert rec.typical_call is None and rec.yards_delta_vs_typical is None and rec.low_confidence


def test_invalid_and_unavailable():
    ctx, _ = make_ctx(2, False)
    with pytest.raises(InvalidSituation) as ei:
        recommend(ctx, Situation(9, 5, "ZZZ", "Cover-1"))
    assert ei.value.fields == ["down", "offense_formation"]
    with pytest.raises(PredictionUnavailable):
        recommend(SimpleNamespace(**{**ctx.__dict__, "pressure_model": None}), Situation(1, 5, "SHOTGUN", "Cover-1"))
    with pytest.raises(PredictionUnavailable):
        recommend(SimpleNamespace(**{**ctx.__dict__, "candidates": []}), Situation(1, 5, "SHOTGUN", "Cover-1"))
