"""Propiedades 4, 5 y 6: dataset a nivel de jugada."""
import numpy as np
import pandas as pd
import pytest

from backend.dataset import PLAYS_REQUIRED, SCOUTING_REQUIRED, DataError, build_play_dataset

from .fixtures import raw_frames


def nz(v) -> bool:
    return not pd.isna(v) and v != 0


def oracle(plays, scouting, g, p):
    """Recalcula las métricas de una jugada con bucles simples sobre su (gameId, playId)."""
    rows = [r for r in scouting.to_dict("records") if r["gameId"] == g and r["playId"] == p]
    rush = [r for r in rows if r["pff_role"] == "Pass Rush"]
    block = [r for r in rows if r["pff_role"] == "Pass Block"]
    return {
        "passRushers": len(rush),
        "rushHits": sum(nz(r["pff_hit"]) for r in rush),
        "rushHurries": sum(nz(r["pff_hurry"]) for r in rush),
        "rushSacks": sum(nz(r["pff_sack"]) for r in rush),
        "passBlockers": len(block),
        "hitsAllowed": sum(nz(r["pff_hitAllowed"]) for r in block),
        "hurriesAllowed": sum(nz(r["pff_hurryAllowed"]) for r in block),
        "sacksAllowed": sum(nz(r["pff_sackAllowed"]) for r in block),
    }, bool(rush or block)


def test_property_4_role_specific_metrics_and_composite_key():
    """Feature: coach-pass-rush-assistant, Property 4: Role-specific metrics and composite-key join"""
    rng = np.random.default_rng(4)
    for _ in range(100):
        plays, scouting = raw_frames(rng)
        out = build_play_dataset(plays, scouting).set_index(["gameId", "playId"])
        for (g, p), row in out.iterrows():
            expected, _ = oracle(plays, scouting, g, p)
            for col, val in expected.items():
                assert row[col] == val, (g, p, col)


def test_property_5_one_labeled_row_per_pass_play():
    """Feature: coach-pass-rush-assistant, Property 5: One labeled row per Pass_Play

    La presión se toma de filas "Pass Rush" (regla 2.6); otros roles llevan ruido que no cuenta.
    """
    rng = np.random.default_rng(5)
    for _ in range(100):
        plays, scouting = raw_frames(rng)
        out = build_play_dataset(plays, scouting)
        assert not out.duplicated(["gameId", "playId"]).any()
        expected_keys = {(r.gameId, r.playId) for r in plays.itertuples()
                         if oracle(plays, scouting, r.gameId, r.playId)[1]}
        assert set(zip(out.gameId, out.playId)) == expected_keys
        for r in out.itertuples():
            m, _ = oracle(plays, scouting, r.gameId, r.playId)
            assert r.pressure == int(m["rushHits"] + m["rushHurries"] + m["rushSacks"] > 0)
            assert r.sack == int(m["rushSacks"] > 0)
        assert pd.api.types.is_numeric_dtype(out["yardsGained"])
        merged = out.merge(plays, on=["gameId", "playId"])
        assert (merged["yardsGained"] == merged["playResult"]).all()


def test_property_6_missing_columns_fail_without_partial_output():
    """Feature: coach-pass-rush-assistant, Property 6: Missing columns fail without partial output"""
    rng = np.random.default_rng(6)
    plays, scouting = raw_frames(rng)
    p0, s0 = plays.copy(deep=True), scouting.copy(deep=True)
    cases = [("plays.csv", c) for c in PLAYS_REQUIRED] + [("pffScoutingData.csv", c) for c in SCOUTING_REQUIRED]
    for i in range(100):
        file, col = cases[i % len(cases)]
        pl, sc = plays.copy(), scouting.copy()
        if file == "plays.csv":
            pl = pl.drop(columns=[col])
        else:
            sc = sc.drop(columns=[col])
        with pytest.raises(DataError) as ei:
            build_play_dataset(pl, sc)
        assert (ei.value.file, ei.value.column) == (file, col)
        assert col in str(ei.value)
    pd.testing.assert_frame_equal(plays, p0)
    pd.testing.assert_frame_equal(scouting, s0)


def test_role_matching_is_case_sensitive():
    plays, scouting = raw_frames(np.random.default_rng(1))
    assert build_play_dataset(plays, scouting.assign(pff_role="pass rush")).empty
    assert build_play_dataset(plays, scouting.assign(pff_role="PASS BLOCK")).empty
    assert not build_play_dataset(plays, scouting.assign(pff_role="Pass Rush")).empty
