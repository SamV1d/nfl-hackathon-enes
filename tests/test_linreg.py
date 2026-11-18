"""Propiedad 7 y pruebas de las utilidades OLS."""
import numpy as np
import pandas as pd
import pytest

from backend.linreg import fit_encoder, fit_ols, metrics, predict_ols, split_indices


def test_property_7_deterministic_shared_split():
    """Feature: coach-pass-rush-assistant, Property 7: Deterministic, shared 80/20 split"""
    rng = np.random.default_rng(7)
    for _ in range(100):
        n = int(rng.integers(100, 3000))
        seed = int(rng.integers(0, 2**31))
        tr1, ho1 = split_indices(n, seed)
        tr2, ho2 = split_indices(n, seed)
        assert np.array_equal(tr1, tr2) and np.array_equal(ho1, ho2)
        assert set(tr1).isdisjoint(ho1)
        assert np.array_equal(np.sort(np.concatenate([tr1, ho1])), np.arange(n))
        assert abs(len(tr1) - 0.8 * n) <= 1 and len(tr1) == int(np.floor(0.8 * n))


def test_split_differs_across_seeds():
    assert not np.array_equal(split_indices(1000, 1)[0], split_indices(1000, 2)[0])


def test_ols_recovers_coefficients_and_metrics():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=400), "g": rng.choice(["x", "y", "z"], 400)})
    y = 1 + 2 * df["a"] + np.where(df["g"] == "y", 3, 0) + rng.normal(scale=0.01, size=400)
    enc = fit_encoder(df, ["a"], ["g"])
    X = enc.transform(df)
    beta = fit_ols(X, y.to_numpy())
    m = metrics(y.to_numpy(), predict_ols(X, beta))
    assert beta[enc.feature_names.index("a")] == pytest.approx(2, abs=0.01)
    assert m["r2"] > 0.999 and m["mae"] < 0.02


def test_encoder_rejects_unseen_category_and_nulls():
    df = pd.DataFrame({"a": [1.0, 2.0], "g": ["x", "y"]})
    enc = fit_encoder(df, ["a"], ["g"])
    with pytest.raises(ValueError, match="g"):
        enc.transform(pd.DataFrame({"a": [1.0], "g": ["w"]}))
    with pytest.raises(ValueError, match="a"):
        enc.transform(pd.DataFrame({"a": [np.nan], "g": ["x"]}))


def test_metrics_values():
    m = metrics([1, 2, 3], [1, 2, 4])
    assert m["mae"] == pytest.approx(1 / 3)
    assert m["r2"] == pytest.approx(1 - 1 / 2)
    assert np.isnan(metrics([1, 1], [1, 1])["r2"])
