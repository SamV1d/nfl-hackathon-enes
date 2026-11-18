"""Propiedades 16, 17, 21, 22 y 23: reportes, hallazgos, partidos y repetición."""
import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from backend import reports
from backend.linreg import split_indices
from backend.models import fit_models
from backend.recommender import candidate_calls

from .fixtures import TEAMS, model_dataset, report_frames


def same_rounding(reported, raw, nd):
    """reported == round(raw, nd), salvo en el borde exacto de redondeo (error de coma flotante)."""
    if raw is None:
        return reported is None
    scaled = raw * 10**nd
    if abs((scaled - math.floor(scaled)) - 0.5) < 1e-9:
        return abs(reported - raw) <= 10**-nd
    return reported == round(raw, nd)


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def oracle_side(rows):
    return (len(rows), mean([r.pressure for r in rows]), mean([r.sack for r in rows]),
            mean([r.yardsGained for r in rows]))


def test_property_21_opponent_report_aggregates():
    """Feature: coach-pass-rush-assistant, Property 21: Opponent report aggregates"""
    rng = np.random.default_rng(21)
    for _ in range(100):
        ds, scouting, players = report_frames(rng)
        stats, league = reports.build_team_stats(ds, scouting, players)
        recs = list(ds.itertuples())
        for team in TEAMS:
            rep = stats.get(team)
            off = [r for r in recs if r.possessionTeam == team]
            dfn = [r for r in recs if r.defensiveTeam == team]
            if not off and not dfn:
                assert rep is None
                assert reports.team_report(ds, reports.player_counts(ds, scouting), {}, team)["has_data"] is False
                continue
            assert rep["has_data"] and rep["message"] is None
            for side, rows in (("offense", off), ("defense", dfn)):
                n, pr, sk, yd = oracle_side(rows)
                got = rep[side]
                assert got["pass_plays"] == n
                if n == 0:
                    assert got["pressure_rate"] is None and got["by_coverage"] == []
                    continue
                assert same_rounding(got["pressure_rate"], pr, 3) and 0 <= got["pressure_rate"] <= 1
                assert same_rounding(got["sack_rate"], sk, 3) and 0 <= got["sack_rate"] <= 1
                assert same_rounding(got["avg_yards"], yd, 2)
                assert sum(g["count"] for g in got["by_coverage"]) == n
                assert sum(g["count"] for g in got["by_formation"]) == n
        assert league["pass_plays"] == len(ds)


def test_property_22_top_player_rankings():
    """Feature: coach-pass-rush-assistant, Property 22: Top player rankings"""
    rng = np.random.default_rng(22)
    for _ in range(100):
        ds, scouting, players = report_frames(rng)
        stats, _ = reports.build_team_stats(ds, scouting, players)
        stats2, _ = reports.build_team_stats(ds, scouting, players)
        names = {int(r.nflId): (r.displayName, r.officialPosition) for r in players.itertuples()}
        team_of = {(r.gameId, r.playId): r.defensiveTeam for r in ds.itertuples()}
        for team, rep in stats.items():
            assert rep == stats2[team]  # determinista
            sacks, pres = {}, {}
            for r in scouting.itertuples():
                if r.pff_role != "Pass Rush" or team_of.get((r.gameId, r.playId)) != team:
                    continue
                nz = lambda v: not pd.isna(v) and v != 0  # noqa: E731
                if nz(r.pff_sack):
                    sacks[r.nflId] = sacks.get(r.nflId, 0) + 1
                if nz(r.pff_hit) or nz(r.pff_hurry) or nz(r.pff_sack):
                    pres[r.nflId] = pres.get(r.nflId, 0) + 1
            for key, counts in (("top_sackers", sacks), ("top_pressure", pres)):
                got = rep[key]
                expected = sorted(counts.items(),
                                  key=lambda kv: (-kv[1], names.get(kv[0], (str(kv[0]),))[0], kv[0]))[:5]
                assert [(p["player_id"], p["count"]) for p in got] == expected
                assert len(got) <= 5 and all(p["count"] >= 1 for p in got)
                for p in got:
                    if p["player_id"] in names:
                        assert (p["name"], p["position"]) == names[p["player_id"]]
                    else:
                        assert (p["name"], p["position"]) == (str(p["player_id"]), "unknown")


def test_property_23_findings_computed_from_data():
    """Feature: coach-pass-rush-assistant, Property 23: Findings computed from data"""
    rng = np.random.default_rng(23)
    for _ in range(100):
        ds, scouting, players = report_frames(rng)
        f = reports.compute_findings(ds, scouting, players)
        recs = list(ds.itertuples())
        w = [r for r in recs if r.pressure == 1]
        wo = [r for r in recs if r.pressure == 0]
        s = f["summary"]
        assert same_rounding(s["pressure_rate"], len(w) / len(recs), 4)
        for key, rows, attr in (("avg_yards_with_pressure", w, "yardsGained"),
                                ("avg_yards_without_pressure", wo, "yardsGained"),
                                ("sack_rate_with_pressure", w, "sack"),
                                ("sack_rate_without_pressure", wo, "sack")):
            raw = mean([getattr(r, attr) for r in rows])
            assert same_rounding(s[key], raw, 2 if "yards" in key else 4), key
            if not rows:
                assert s[key] is None
        for key, col in (("by_coverage", "actualCoverage"), ("by_formation", "offenseFormation")):
            assert {g["group"] for g in f[key]} == {str(getattr(r, col)) for r in recs}
            assert sum(g["count"] for g in f[key]) == len(recs)
        assert sum(g["count"] for g in f["by_defenders_in_box"]) == len(recs)
        assert len({g["group"] for g in f["by_defenders_in_box"]}) == len(f["by_defenders_in_box"])
        counts = [p["count"] for p in f["top_sackers"]]
        assert counts == sorted(counts, reverse=True) and len(counts) <= 10
        rates = [t["pressure_rate"] for t in f["top_teams"]]
        assert rates == sorted(rates, reverse=True) and len(rates) <= 10
        teams = {r.defensiveTeam for r in recs}
        assert {t["team"] for t in f["top_teams"]} <= teams


def test_findings_zero_denominators_are_none():
    ds, scouting, players = report_frames(np.random.default_rng(5))
    ds["pressure"] = 0
    f = reports.compute_findings(ds, scouting, players)["summary"]
    assert f["avg_yards_with_pressure"] is None and f["sack_rate_with_pressure"] is None
    ds["pressure"] = 1
    f = reports.compute_findings(ds, scouting, players)["summary"]
    assert f["avg_yards_without_pressure"] is None and f["sack_rate_without_pressure"] is None


def test_property_16_games_sorted_by_date():
    """Feature: coach-pass-rush-assistant, Property 16: Games sorted by date"""
    rng = np.random.default_rng(16)
    for _ in range(100):
        n = int(rng.integers(1, 40))
        dates = pd.Timestamp("2021-09-01") + pd.to_timedelta(rng.integers(0, 120, n), unit="D")
        games = pd.DataFrame({
            "gameId": rng.permutation(n) + 100, "gameDate": dates.strftime("%m/%d/%Y"),
            "gameTimeEastern": [f"{h}:00:00" for h in rng.integers(10, 21, n)],
            "homeTeamAbbr": rng.choice(TEAMS, n), "visitorTeamAbbr": rng.choice(TEAMS, n)})
        out = reports.list_games(games)
        assert sorted(g["game_id"] for g in out) == sorted(games["gameId"])
        assert [g["date"] for g in out] == sorted(g["date"] for g in out)
        by_id = games.set_index("gameId")
        for g in out:
            assert g["home_team"] == by_id.loc[g["game_id"], "homeTeamAbbr"]
            assert g["visitor_team"] == by_id.loc[g["game_id"], "visitorTeamAbbr"]


def test_property_17_replay_play_ordering():
    """Feature: coach-pass-rush-assistant, Property 17: Replay play ordering"""
    rng = np.random.default_rng(17)
    for i in range(100):
        ds = model_dataset(rng, 120)
        ds["quarter"] = rng.integers(1, 6, len(ds))
        ds["gameClock"] = [f"{m:02d}:{s:02d}" for m, s in zip(rng.integers(0, 16, len(ds)), rng.integers(0, 60, len(ds)))]
        ds["gameId"] = rng.integers(1, 3, len(ds))
        tr, ho = split_indices(len(ds), i)
        fit = fit_models(ds, tr, ho)
        ctx = SimpleNamespace(pressure_model=fit.pressure_model, yards_model=fit.yards_model, dataset=ds,
                              train_idx=tr, candidates=candidate_calls(ds.iloc[tr]))
        game = int(rng.integers(1, 3))
        out = reports.replay_plays(ctx, game)
        expected_ids = set(ds.loc[ds.gameId == game, "playId"])
        assert [p["play_id"] for p in out["plays"]] and len({p["play_id"] for p in out["plays"]}) == out["total"]
        assert {p["play_id"] for p in out["plays"]} == expected_ids
        keys = [(p["quarter"], -reports.clock_seconds(p["game_clock"])) for p in out["plays"]]
        assert keys == sorted(keys)
        assert [p["position"] for p in out["plays"]] == list(range(1, out["total"] + 1))
        for p in out["plays"]:
            assert p["quarter_label"] == ("overtime" if p["quarter"] == 5 else str(p["quarter"]))
            assert p["down_label"] == ("two-point conversion" if p["down"] == 0 else str(p["down"]))


def test_replay_empty_game_has_message():
    rng = np.random.default_rng(1)
    ds = model_dataset(rng, 150)
    ctx = SimpleNamespace(pressure_model=None, yards_model=None, dataset=ds, train_idx=np.arange(10), candidates=[])
    out = reports.replay_plays(ctx, 999)
    assert out["total"] == 0 and out["plays"] == [] and out["message"]
    # sin modelos la repetición sigue funcionando, sin Best_Call
    assert all(p["best_call"] is None for p in reports.replay_plays(ctx, 1)["plays"])
