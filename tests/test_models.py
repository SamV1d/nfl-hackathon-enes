"""Propiedades 8 a 11 y rutas de fallo (3.7, 4.6)."""
from types import SimpleNamespace

import numpy as np
import pytest

from backend import models
from backend.linreg import metrics, split_indices
from backend.models import InvalidSituation, Situation, fit_models, validate_situation
from backend.recommender import candidate_calls, recommend

from .fixtures import model_dataset, random_situation


def fitted(seed: int, n: int = 300):
    rng = np.random.default_rng(seed)
    ds = model_dataset(rng, n)
    tr, ho = split_indices(len(ds), seed)
    res = fit_models(ds, tr, ho)
    assert not res.errors
    return ds, tr, ho, res.pressure_model, res.yards_model, rng


def decimals(x: float, nd: int) -> bool:
    return abs(x - round(x, nd)) < 1e-12


def test_property_8_outputs_bounded_and_rounded():
    """Feature: coach-pass-rush-assistant, Property 8: Model outputs are bounded and rounded"""
    for seed in range(100):
        ds, tr, ho, pm, ym, rng = fitted(seed, 150)
        s = random_situation(rng, pm.vocab)
        assert 0.0 <= pm.predict(s) <= 1.0
        assert decimals(ym.predict(s), 2)
        if True:
            ctx = SimpleNamespace(pressure_model=pm, yards_model=ym, dataset=ds, train_idx=tr,
                                  candidates=candidate_calls(ds.iloc[tr]))
            rec = recommend(ctx, s)
            assert rec.ranked
            for r in rec.ranked:
                assert 0.0 <= r.pressure_probability <= 1.0 and decimals(r.expected_yards, 1)


def test_property_9_reported_metrics_match_heldout_recomputation():
    """Feature: coach-pass-rush-assistant, Property 9: Reported metrics match held-out recomputation"""
    for seed in range(100):
        ds, tr, ho, pm, ym, _ = fitted(seed, 150)
        vocab = pm.vocab
        y_p, p_p, y_y, p_y = [], [], [], []
        for i in ho:
            r = ds.iloc[i]
            if r.offenseFormation not in vocab["offenseFormation"] or r.expectedCoverage not in vocab["expectedCoverage"]:
                continue
            s = Situation(int(r.down), int(r.yardsToGo), r.offenseFormation, r.expectedCoverage,
                          int(r.passRushers), float(r.defendersInBox))
            p = pm.predict(s)
            y_p.append(r.pressure); p_p.append(p)
            y_y.append(r.yardsGained); p_y.append(ym.predict_raw(s, p))
        mp, my = metrics(y_p, p_p), metrics(y_y, p_y)
        a, b = pm.metrics(), ym.metrics()
        assert a["r2"] == pytest.approx(mp["r2"]) and a["mae"] == pytest.approx(mp["mae"])
        assert b["r2"] == pytest.approx(my["r2"]) and b["mae_yards"] == pytest.approx(my["mae"])
        for m in (a, b):
            assert m["n_train"] == len(tr) and m["n_heldout"] == len(ho)
            assert m["n_train"] + m["n_heldout"] == len(ds)


def test_property_10_invalid_situations_rejected_models_unchanged():
    """Feature: coach-pass-rush-assistant, Property 10: Invalid situations are rejected and leave models unchanged"""
    ds, tr, ho, pm, ym, rng = fitted(10)
    beta_p, beta_y = pm.beta.copy(), ym.beta.copy()
    bad_values = {
        "down": [-1, 5, None, "2", 2.5, True],
        "yards_to_go": [-1, 100, None, "7", 3.5],
        "offense_formation": ["ZZZ", None, "", 3],
        "expected_coverage": ["ZZZ", None, "", 3],
    }
    for i in range(100):
        s = random_situation(rng, pm.vocab)
        field = list(bad_values)[i % 4]
        value = bad_values[field][int(rng.integers(0, len(bad_values[field])))]
        bad = Situation(**{**s.__dict__, field: value})
        assert validate_situation(bad, pm.vocab) == [field]
        for model in (pm, ym):
            with pytest.raises(InvalidSituation) as ei:
                model.predict(bad)
            assert ei.value.fields == [field]
    assert np.array_equal(pm.beta, beta_p) and np.array_equal(ym.beta, beta_y)


def test_missing_call_features_are_rejected_by_pressure_model():
    ds, tr, ho, pm, ym, rng = fitted(11)
    s = random_situation(rng, pm.vocab)
    for field in ("pass_rushers", "defenders_in_box"):
        with pytest.raises(InvalidSituation) as ei:
            pm.predict(Situation(**{**s.__dict__, field: None}))
        assert ei.value.fields == [field]


def test_property_11_yards_respond_linearly_to_pressure():
    """Feature: coach-pass-rush-assistant, Property 11: Yards respond linearly to pressure"""
    ds, tr, ho, pm, ym, rng = fitted(12)
    for _ in range(100):
        s = random_situation(rng, pm.vocab)
        p1, p2 = rng.random(2)
        diff = ym.predict_raw(s, p2) - ym.predict_raw(s, p1)
        assert abs(diff - ym.pressure_coef * (p2 - p1)) <= 0.01


def test_coefficients_are_read_only():
    *_, pm, ym, _ = fitted(13)
    with pytest.raises(ValueError):
        pm.beta[0] = 0.0


def test_fewer_than_100_labeled_plays_gives_unavailable():
    rng = np.random.default_rng(14)
    ds = model_dataset(rng, 99)
    tr, ho = split_indices(len(ds), 1)
    res = fit_models(ds, tr, ho)
    assert res.pressure_model is None and res.yards_model is None
    assert "100" in res.errors["pressure"] and "yards" in res.errors


def test_nan_labels_do_not_count_as_labeled():
    rng = np.random.default_rng(15)
    ds = model_dataset(rng, 150)
    ds.loc[ds.index[:60], "pressure"] = np.nan
    res = fit_models(ds, *split_indices(len(ds), 1))
    assert res.pressure_model is None


def test_injected_fit_failure_gives_unavailable(monkeypatch):
    def boom(X, y):
        raise np.linalg.LinAlgError("injected")

    monkeypatch.setattr(models, "fit_ols", boom)
    rng = np.random.default_rng(16)
    ds = model_dataset(rng, 300)
    res = fit_models(ds, *split_indices(len(ds), 1))
    assert res.pressure_model is None and res.yards_model is None
    assert res.errors["pressure"] and res.errors["yards"]


def test_yards_failure_keeps_pressure_model(monkeypatch):
    real = models.fit_ols
    calls = []

    def second_fails(X, y):
        calls.append(1)
        if len(calls) == 2:
            raise np.linalg.LinAlgError("injected")
        return real(X, y)

    monkeypatch.setattr(models, "fit_ols", second_fails)
    ds = model_dataset(np.random.default_rng(17), 300)
    res = fit_models(ds, *split_indices(len(ds), 1))
    assert res.pressure_model is not None and res.yards_model is None and "yards" in res.errors
