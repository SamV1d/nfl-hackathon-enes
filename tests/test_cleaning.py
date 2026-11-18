"""Propiedad 3: la limpieza conserva etiquetas y elimina nulos."""
import numpy as np
import pandas as pd

from backend.cleaning import clean_plays, train_median_defenders_in_box
from backend.linreg import split_indices


def random_plays(rng: np.random.Generator, n: int) -> pd.DataFrame:
    def with_nulls(values, p=0.2):
        return [None if rng.random() < p else v for v in values]

    return pd.DataFrame({
        "gameId": rng.integers(1, 5, n),
        "playId": rng.integers(1, 1000, n),
        "down": rng.integers(0, 5, n),
        "quarter": rng.integers(1, 6, n),
        "dropBackType": with_nulls(rng.choice(["TRADITIONAL", "SCRAMBLE"], n)),
        "offenseFormation": with_nulls(rng.choice(["SHOTGUN", "EMPTY", "I_FORM"], n)),
        "defendersInBox": pd.array(with_nulls(rng.integers(3, 10, n).tolist(), 0.25), dtype="Float64")
        .astype(float),
    })


def test_property_3_cleaning_preserves_labels_and_removes_nulls():
    """Feature: coach-pass-rush-assistant, Property 3: Cleaning preserves labels and removes nulls"""
    rng = np.random.default_rng(3)
    for i in range(100):
        n = int(rng.integers(20, 120))
        plays = random_plays(rng, n)
        original = plays.copy(deep=True)
        train, _ = split_indices(n, seed=int(rng.integers(0, 10_000)))
        median = train_median_defenders_in_box(plays, train)
        expected_median = plays["defendersInBox"].to_numpy()[train]
        expected_median = float(np.median(expected_median[~np.isnan(expected_median)]))
        if np.isnan(expected_median):
            continue
        assert median == expected_median

        out = clean_plays(plays, median)
        pd.testing.assert_frame_equal(plays, original)  # no muta la entrada
        assert (out["down"] == original["down"]).all()
        assert (out["quarter"] == original["quarter"]).all()
        assert (out["is_two_point"] == (original["down"] == 0)).all()
        assert (out["is_overtime"] == (original["quarter"] == 5)).all()
        for col in ("dropBackType", "offenseFormation", "defendersInBox"):
            assert out[col].notna().all()
        for col in ("dropBackType", "offenseFormation"):
            was_null = original[col].isna()
            assert (out.loc[was_null, col] == "Unknown").all()
            assert (out.loc[~was_null, col] == original.loc[~was_null, col]).all()
        was_null = original["defendersInBox"].isna()
        assert (out.loc[was_null, "defendersInBox"] == median).all()
        assert (out.loc[~was_null, "defendersInBox"] == original.loc[~was_null, "defendersInBox"]).all()
