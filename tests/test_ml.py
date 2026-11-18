"""Pruebas del paquete backend.ml: métricas, estimadores, variables y partición temporal."""
import math

import numpy as np
import pandas as pd
import pytest

from backend.ml import metrics as M
from backend.ml.estimators import (GradientBoosting, LogisticRegression, MeanBaseline,
                                   RandomForest, RidgeRegression)
from backend.ml.features import (FEATURE_CATALOG, MODEL_FEATURES, POST_SNAP, FeatureEncoder,
                                 temporal_split, used_features_by_timing)


# ------------------------------------------------------------------ métricas
def test_brier_and_log_loss_known_values():
    y, p = [1, 0, 1, 0], [0.9, 0.1, 0.8, 0.3]
    assert M.brier(y, p) == pytest.approx((0.01 + 0.01 + 0.04 + 0.09) / 4)
    assert M.log_loss([1, 0], [0.5, 0.5]) == pytest.approx(math.log(2))


def test_roc_auc_perfect_random_and_ties():
    assert M.roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert M.roc_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == 0.0
    assert M.roc_auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5
    assert math.isnan(M.roc_auc([1, 1], [0.2, 0.3]))


def test_pr_auc_perfect_and_known():
    assert M.pr_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)
    # orden por score: 1,0,1,0 -> AP = (1/1 + 2/3) / 2
    assert M.pr_auc([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.1]) == pytest.approx((1 + 2 / 3) / 2)


def test_precision_recall_and_threshold():
    r = M.precision_recall_at([1, 1, 0, 0], [0.9, 0.4, 0.6, 0.1], 0.5)
    assert r["precision"] == 0.5 and r["recall"] == 0.5


def test_calibration_curve_perfectly_calibrated_has_low_ece():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 20000)
    y = (rng.uniform(0, 1, 20000) < p).astype(float)
    assert M.expected_calibration_error(y, p) < 0.02
    assert sum(c["n"] for c in M.calibration_curve(y, p)) == 20000


def test_regression_metrics():
    y, yhat = [1.0, 2.0, 3.0], [1.0, 2.0, 4.0]
    assert M.mae(y, yhat) == pytest.approx(1 / 3)
    assert M.rmse(y, yhat) == pytest.approx(math.sqrt(1 / 3))
    assert M.r2_score(y, y) == 1.0
    assert math.isnan(M.r2_score([2, 2], [1, 3]))


# --------------------------------------------------------------- estimadores
@pytest.fixture(scope="module")
def synthetic():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(3000, 5))
    logit = 1.5 * X[:, 0] - 1.0 * X[:, 1] + 0.8 * X[:, 0] * (X[:, 2] > 0)
    y = (rng.uniform(size=3000) < 1 / (1 + np.exp(-logit))).astype(float)
    return X[:2000], y[:2000], X[2000:], y[2000:]


@pytest.mark.parametrize("model", [
    LogisticRegression(l2=1.0),
    RandomForest(n_trees=30, max_depth=4, min_leaf=20),
    GradientBoosting("logistic", n_trees=60, max_depth=2),
])
def test_classifiers_beat_baseline_on_known_signal(synthetic, model):
    Xtr, ytr, Xte, yte = synthetic
    p = model.fit(Xtr, ytr).predict(Xte)
    assert p.min() >= 0 and p.max() <= 1
    assert M.roc_auc(yte, p) > 0.75
    base = MeanBaseline().fit(Xtr, ytr).predict(Xte)
    assert M.log_loss(yte, p) < M.log_loss(yte, base)


def test_regressors_recover_linear_signal():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(1500, 4))
    y = 3 * X[:, 0] - 2 * X[:, 1] + rng.normal(scale=0.5, size=1500)
    for model in (RidgeRegression(alpha=1.0), GradientBoosting("squared", n_trees=150, max_depth=2,
                                                                 learning_rate=0.1, min_leaf=20)):
        pred = model.fit(X[:1000], y[:1000]).predict(X[1000:])
        assert M.r2_score(y[1000:], pred) > 0.8


def test_estimators_are_deterministic(synthetic):
    Xtr, ytr, Xte, _ = synthetic
    a = GradientBoosting("logistic", n_trees=20, seed=3).fit(Xtr, ytr).predict(Xte)
    b = GradientBoosting("logistic", n_trees=20, seed=3).fit(Xtr, ytr).predict(Xte)
    assert np.array_equal(a, b)


def test_random_forest_on_pure_noise_does_not_claim_signal():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(2000, 4))
    y = (rng.uniform(size=2000) < 0.4).astype(float)
    p = RandomForest(n_trees=30, max_depth=4, min_leaf=30).fit(X[:1500], y[:1500]).predict(X[1500:])
    assert abs(M.roc_auc(y[1500:], p) - 0.5) < 0.08


# ---------------------------------------------------------- variables y fugas
def test_no_post_snap_feature_is_used_by_models():
    assert all(FEATURE_CATALOG[f][1] != POST_SNAP for f in MODEL_FEATURES)
    for banned in ("dropBackType", "pff_playAction", "passResult", "rushHits", "rushHurries", "pressure"):
        assert banned not in MODEL_FEATURES
    assert set(used_features_by_timing()) == {"pre_snap", "defensive_choice", "post_snap"}
    assert used_features_by_timing()["post_snap"] == []


def _frame(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "down": rng.integers(1, 5, n), "yardsToGo": rng.integers(1, 15, n),
        "offenseFormation": rng.choice(["SHOTGUN", "EMPTY", "RARE"], n, p=[0.6, 0.38, 0.02]),
        "expectedCoverage": rng.choice(["Cover-3", "Cover-1"], n),
    })


def test_encoder_groups_rare_categories_and_flags_ood():
    train = _frame()
    enc = FeatureEncoder(["down", "yardsToGo"], ["offenseFormation", "expectedCoverage"]).fit(train)
    X = enc.transform(train)
    assert X.shape == (len(train), len(enc.feature_names))
    assert not np.isnan(X).any()
    new = pd.DataFrame({"down": [3], "yardsToGo": [40], "offenseFormation": ["WILDCAT"],
                        "expectedCoverage": ["Cover-3"]})
    assert enc.transform(new).shape[1] == X.shape[1]  # categoría nueva no rompe la predicción
    assert set(enc.out_of_distribution(new)) == {"yardsToGo", "offenseFormation"}


def test_encoder_imputes_with_training_median_only():
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0, np.nan]})
    enc = FeatureEncoder(["x"], []).fit(train)
    out = enc.transform(pd.DataFrame({"x": [np.nan]}))
    assert out[0, 0] == 2.0


def test_temporal_split_is_disjoint_and_ordered_by_week():
    df = pd.DataFrame({"week": [1, 1, 2, 5, 6, 7, 8, 8], "gameId": [1, 1, 2, 5, 6, 7, 8, 8]})
    parts = temporal_split(df)
    assert set(parts) == {"train", "valid", "test"}
    wk = df["week"].to_numpy()
    assert wk[parts["train"]].max() < wk[parts["valid"]].min() <= wk[parts["valid"]].max() < wk[parts["test"]].min()
    assert not (set(parts["train"]) & set(parts["test"]))


def test_temporal_split_rejects_game_shared_across_sets():
    df = pd.DataFrame({"week": [1, 7], "gameId": [1, 1]})
    with pytest.raises(ValueError):
        temporal_split(df)

# ------------------------------------------------- integracion con datos reales
from pathlib import Path  # noqa: E402

from backend.config import data_dir_from_env  # noqa: E402

REAL = data_dir_from_env()


@pytest.mark.skipif(not (REAL / "plays.csv").is_file(), reason="datos reales no disponibles")
def test_real_audit_structure_and_honest_status():
    from backend.ml.audit import load_and_audit
    r = load_and_audit()
    assert r["protocol"]["train_weeks"] == [1, 2, 3, 4, 5] and r["protocol"]["test_weeks"] == [7, 8]
    assert sum(r["protocol"]["n_plays"].values()) == 8557
    for key in ("pressure_with_defensive_choice", "pressure_pre_snap_only", "sack"):
        a = r[key]
        assert a["status"] in {"validated", "limited", "not_reliable"}
        assert a["test"]["brier"] <= a["test_baseline"]["brier"] + 1e-9 or a["status"] == "not_reliable"
        assert 0 <= a["test"]["roc_auc"] <= 1 and a["calibration_curve"]
        assert not (set(a["features"]["numeric"]) & {"pressure", "sack", "rushHits"})
    for key in ("yards_net", "yards_on_attempts"):
        assert r[key]["test"]["rmse"] > 0 and r[key]["status"] in {"validated", "limited", "not_reliable"}
