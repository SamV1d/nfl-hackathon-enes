"""Reporte del rival (y, en olas siguientes, hallazgos y repetición de partidos)."""
from __future__ import annotations

from typing import Any

import pandas as pd

from .dataset import KEYS, ROLE_RUSH, nonzero

NO_DATA_MESSAGE = "sin datos disponibles para este equipo"
TOP_K = 5


def _round(x: float, nd: int) -> float | None:
    return None if pd.isna(x) else round(float(x), nd)


def _rates(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {"pass_plays": 0, "pressure_rate": None, "sack_rate": None, "avg_yards": None}
    return {
        "pass_plays": int(len(df)),
        "pressure_rate": _round(df["pressure"].mean(), 3),
        "sack_rate": _round(df["sack"].mean(), 3),
        "avg_yards": _round(df["yardsGained"].mean(), 2),
    }


def _groups(df: pd.DataFrame, col: str) -> list[dict[str, Any]]:
    out = []
    for key, g in df.groupby(col, sort=True):
        out.append({"group": str(key), "count": int(len(g)),
                    "pressure_rate": _round(g["pressure"].mean(), 3),
                    "avg_yards": _round(g["yardsGained"].mean(), 2)})
    return sorted(out, key=lambda r: (-r["count"], r["group"]))


def side_report(df: pd.DataFrame) -> dict[str, Any]:
    """Tasas del equipo en un lado (ataque o defensa) y desgloses con conteo."""
    return {**_rates(df),
            "by_coverage": _groups(df, "actualCoverage"),
            "by_formation": _groups(df, "offenseFormation")}


def league_average(dataset: pd.DataFrame) -> dict[str, Any]:
    return _rates(dataset)


def make_player_lookup(players: pd.DataFrame) -> dict[int, tuple[str, str]]:
    """nflId -> (nombre, posición). Valores nulos se omiten para que apliquen los respaldos."""
    out: dict[int, tuple[str, str]] = {}
    for pid, name, pos in players[["nflId", "displayName", "officialPosition"]].itertuples(index=False):
        out[int(pid)] = (str(name) if pd.notna(name) else str(int(pid)),
                         str(pos) if pd.notna(pos) else "unknown")
    return out


def player_counts(dataset: pd.DataFrame, scouting: pd.DataFrame) -> pd.DataFrame:
    """Sacks y presiones por (equipo defensivo, jugador), solo con filas "Pass Rush"."""
    rush = scouting[scouting["pff_role"] == ROLE_RUSH]
    m = rush.merge(dataset[[*KEYS, "defensiveTeam"]], on=KEYS, how="inner")
    m = m.assign(
        _sack=nonzero(m["pff_sack"]),
        _pressure=nonzero(m["pff_hit"]) | nonzero(m["pff_hurry"]) | nonzero(m["pff_sack"]),
    )
    return (m.groupby(["defensiveTeam", "nflId"])
            .agg(sacks=("_sack", "sum"), pressures=("_pressure", "sum"))
            .astype(int).reset_index())


def top_players(counts: pd.DataFrame, col: str, lookup: dict[int, tuple[str, str]],
                k: int = TOP_K) -> list[dict[str, Any]]:
    """Mayor conteo primero, empate por nombre ascendente. Sin ceros ni relleno."""
    rows = []
    for pid, n in counts.loc[counts[col] >= 1, ["nflId", col]].itertuples(index=False):
        pid = int(pid)
        name, pos = lookup.get(pid, (str(pid), "unknown"))
        rows.append({"player_id": pid, "name": name, "position": pos, "count": int(n)})
    rows.sort(key=lambda r: (-r["count"], r["name"], r["player_id"]))
    return rows[:k]


def empty_report(team: str) -> dict[str, Any]:
    return {"team": team, "has_data": False, "message": NO_DATA_MESSAGE,
            "offense": None, "defense": None, "top_sackers": [], "top_pressure": [],
            "league_average": None}


def team_report(dataset: pd.DataFrame, counts: pd.DataFrame, lookup: dict[int, tuple[str, str]],
                team: str, league: dict[str, Any] | None = None) -> dict[str, Any]:
    """Reporte de un equipo; vacío con mensaje si no tiene Pass_Plays."""
    off = dataset[dataset["possessionTeam"] == team]
    dfn = dataset[dataset["defensiveTeam"] == team]
    if off.empty and dfn.empty:
        return empty_report(team)
    mine = counts[counts["defensiveTeam"] == team]
    return {
        "team": team,
        "has_data": True,
        "message": None,
        "offense": side_report(off),
        "defense": side_report(dfn),
        "top_sackers": top_players(mine, "sacks", lookup),
        "top_pressure": top_players(mine, "pressures", lookup),
        "league_average": league if league is not None else league_average(dataset),
    }


def list_teams(dataset: pd.DataFrame) -> list[str]:
    return sorted(set(dataset["possessionTeam"].dropna()) | set(dataset["defensiveTeam"].dropna()))


def build_team_stats(dataset: pd.DataFrame, scouting: pd.DataFrame,
                     players: pd.DataFrame) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Reportes de todos los equipos y promedio de la liga, para precalcular al arrancar."""
    counts = player_counts(dataset, scouting)
    lookup = make_player_lookup(players)
    league = league_average(dataset)
    stats = {t: team_report(dataset, counts, lookup, t, league) for t in list_teams(dataset)}
    return stats, league


# ---------------------------------------------------------------- hallazgos (5.3)
def _ratio(num: float, den: float, nd: int) -> float | None:
    return None if den == 0 else round(float(num) / float(den), nd)


def _mean_or_none(s: pd.Series, nd: int) -> float | None:
    s = s.dropna()
    return None if s.empty else round(float(s.mean()), nd)


def _box_label(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else str(v)


def compute_findings(dataset: pd.DataFrame, scouting: pd.DataFrame, players: pd.DataFrame) -> dict[str, Any]:
    """Hallazgos calculados de los datos cargados. Denominador cero -> None."""
    with_p = dataset[dataset["pressure"] == 1]
    without_p = dataset[dataset["pressure"] == 0]
    summary = {
        "pressure_rate": _ratio(len(with_p), len(dataset), 4),
        "avg_yards_with_pressure": _mean_or_none(with_p["yardsGained"], 2),
        "avg_yards_without_pressure": _mean_or_none(without_p["yardsGained"], 2),
        "sack_rate_with_pressure": _ratio(with_p["sack"].sum(), len(with_p), 4),
        "sack_rate_without_pressure": _ratio(without_p["sack"].sum(), len(without_p), 4),
    }

    box = dataset.assign(_box=dataset["defendersInBox"].map(
        lambda v: _box_label(v) if pd.notna(v) else "Unknown"))
    by_box = sorted(_groups(box, "_box"), key=lambda r: (not r["group"].replace(".", "").isdigit(),
                                                         float(r["group"]) if r["group"].replace(".", "").isdigit() else 0,
                                                         r["group"]))

    counts = player_counts(dataset, scouting)
    per_player = counts.groupby("nflId", as_index=False)[["sacks", "pressures"]].sum()
    top_sackers = top_players(per_player, "sacks", make_player_lookup(players), k=10)

    teams = (dataset.groupby("defensiveTeam")["pressure"].agg(["mean", "size"]).reset_index()
             .sort_values(["mean", "defensiveTeam"], ascending=[False, True]).head(10))
    top_teams = [{"team": str(t), "pressure_rate": _round(m, 3), "pass_plays": int(n)}
                 for t, m, n in teams.itertuples(index=False)]

    return {"summary": summary, "by_coverage": _groups(dataset, "actualCoverage"),
            "by_formation": _groups(dataset, "offenseFormation"), "by_defenders_in_box": by_box,
            "top_sackers": top_sackers, "top_teams": top_teams}


# ---------------------------------------------------- partidos y repetición (5.5)
NO_PASS_PLAYS_MESSAGE = "el partido no tiene jugadas de pase disponibles"


def list_games(games: pd.DataFrame) -> list[dict[str, Any]]:
    """Todos los partidos, por fecha ascendente (hora y gameId desempatan; fechas ilegibles al final)."""
    g = games.assign(
        sort_date=pd.to_datetime(games["gameDate"], format="%m/%d/%Y", errors="coerce"),
        sort_time=games["gameTimeEastern"].astype(str) if "gameTimeEastern" in games.columns else "",
    ).sort_values(["sort_date", "sort_time", "gameId"], na_position="last", kind="stable")
    return [{"game_id": int(r.gameId),
             "date": r.sort_date.strftime("%Y-%m-%d") if pd.notna(r.sort_date) else str(r.gameDate),
             "home_team": str(r.homeTeamAbbr), "visitor_team": str(r.visitorTeamAbbr)}
            for r in g.itertuples(index=False)]


def clock_seconds(clock: Any) -> int:
    """"MM:SS" (o "MM:SS:00") -> segundos; ilegible -> -1."""
    try:
        m, s = str(clock).split(":")[:2]
        return int(m) * 60 + int(s)
    except (ValueError, TypeError):
        return -1


def _best_call(ctx, row) -> dict[str, Any] | None:
    from .models import Situation, validate_situation
    from .recommender import PredictionUnavailable, rank_calls, score_calls

    if ctx.pressure_model is None or ctx.yards_model is None:
        return None
    try:
        s = Situation(int(row.down), int(row.yardsToGo), str(row.offenseFormation), str(row.expectedCoverage))
    except (ValueError, TypeError):
        return None
    if validate_situation(s, ctx.pressure_model.vocab, require_call=False):
        return None
    try:
        best = rank_calls(score_calls(ctx, s))[0]
    except PredictionUnavailable:
        return None
    c = best.call
    return {"call": {"coverage": c.coverage, "pass_rushers": c.pass_rushers,
                     "defenders_in_box": float(c.defenders_in_box)},
            "pressure_probability": best.pressure_probability,
            "expected_yards": round(best.expected_yards, 1)}


def replay_plays(ctx, game_id: int) -> dict[str, Any]:
    """Pass_Plays del partido por cuarto ascendente y reloj descendente, con la Best_Call de cada una."""
    g = ctx.dataset[ctx.dataset["gameId"] == game_id]
    g = (g.assign(_sec=g["gameClock"].map(clock_seconds))
         .sort_values(["quarter", "_sec", "playId"], ascending=[True, False, True], kind="stable"))
    plays = []
    for pos, r in enumerate(g.itertuples(index=False), start=1):
        yards = None if pd.isna(r.yardsGained) else float(r.yardsGained)
        plays.append({
            "position": pos, "play_id": int(r.playId),
            "quarter": int(r.quarter), "quarter_label": "overtime" if r.quarter == 5 else str(int(r.quarter)),
            "down": int(r.down), "down_label": "two-point conversion" if r.down == 0 else str(int(r.down)),
            "yards_to_go": int(r.yardsToGo), "offense_formation": str(r.offenseFormation),
            "expected_coverage": str(r.expectedCoverage), "game_clock": str(r.gameClock),
            "best_call": _best_call(ctx, r),
            "actual_call": {"coverage": str(r.actualCoverage), "pass_rushers": int(r.passRushers),
                            "defenders_in_box": float(r.defendersInBox)},
            "pressure": bool(r.pressure), "sack": bool(r.sack), "yards_gained": yards,
            "two_point_conversion": bool(r.down == 0), "overtime": bool(r.quarter == 5),
        })
    return {"game_id": int(game_id), "total": len(plays),
            "message": None if plays else NO_PASS_PLAYS_MESSAGE, "plays": plays}
