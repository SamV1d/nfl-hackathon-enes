"""Generadores sintéticos con semilla, con los nombres de columna reales."""
import numpy as np
import pandas as pd

from backend.dataset import PLAYS_REQUIRED, SCOUTING_REQUIRED
from backend.models import Situation

ROLES = ["Pass Rush", "Pass Block", "Pass Route", "Coverage", "Pass", "pass rush", "PASS BLOCK", "Pass block"]
FORMATIONS = ["SHOTGUN", "EMPTY", "I_FORM", "SINGLEBACK"]
COVERAGES = ["Cover-1", "Cover-2", "Cover-3", "Quarters"]


def raw_frames(rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    """plays y pffScoutingData sintéticos: playId repetido entre partidos, roles con mayúsculas
    mezcladas, nulos, valores de presión en roles que no los usan y filas huérfanas."""
    rows = []
    for game in (101, 102, 103):
        for pid in rng.choice(np.arange(1, 16), size=int(rng.integers(3, 9)), replace=False):
            rows.append({
                "gameId": game, "playId": int(pid),
                "quarter": int(rng.integers(1, 6)), "down": int(rng.integers(0, 5)),
                "yardsToGo": int(rng.integers(1, 20)),
                "possessionTeam": str(rng.choice(["AAA", "BBB"])), "defensiveTeam": str(rng.choice(["AAA", "BBB"])),
                "gameClock": f"{int(rng.integers(0, 15)):02d}:{int(rng.integers(0, 60)):02d}",
                "offenseFormation": None if rng.random() < 0.2 else str(rng.choice(FORMATIONS)),
                "dropBackType": None if rng.random() < 0.2 else "TRADITIONAL",
                "defendersInBox": None if rng.random() < 0.2 else float(rng.integers(3, 9)),
                "pff_passCoverage": str(rng.choice(COVERAGES)),
                "playResult": int(rng.integers(-8, 30)),
            })
    plays = pd.DataFrame(rows)[list(PLAYS_REQUIRED)]

    def flag():
        return rng.choice([np.nan, 0.0, 1.0])

    srows = []
    keys = [(r["gameId"], r["playId"]) for r in rows] + [(999, 1), (101, 99)]  # huérfanas
    for g, p in keys:
        for _ in range(int(rng.integers(0, 9))):
            role = str(rng.choice(ROLES))
            r = {c: np.nan for c in SCOUTING_REQUIRED}
            r.update(gameId=g, playId=p, nflId=int(rng.integers(1, 30)), pff_role=role)
            if role == "Pass Rush":
                r.update(pff_hit=flag(), pff_hurry=flag(), pff_sack=flag())
            elif role == "Pass Block":
                r.update(pff_hitAllowed=flag(), pff_hurryAllowed=flag(), pff_sackAllowed=flag())
            else:  # ruido: otros roles con valores que no deben contar
                r.update(pff_hit=flag(), pff_hurry=flag(), pff_sack=flag(),
                         pff_hitAllowed=flag(), pff_hurryAllowed=flag(), pff_sackAllowed=flag())
            srows.append(r)
    scouting = pd.DataFrame(srows, columns=list(SCOUTING_REQUIRED))
    return plays, scouting.sample(frac=1, random_state=int(rng.integers(0, 1000))).reset_index(drop=True)


def model_dataset(rng: np.random.Generator, n: int = 300) -> pd.DataFrame:
    """Dataset a nivel de jugada con estructura aprendible (presión depende de rushers)."""
    rushers = rng.integers(3, 8, n)
    box = rng.integers(4, 9, n)
    down = rng.integers(0, 5, n)
    ytg = rng.integers(0, 25, n)
    cov = rng.choice(COVERAGES, n)
    form = rng.choice(FORMATIONS, n)
    p = np.clip(0.1 + 0.08 * (rushers - 3) + 0.02 * (down == 3), 0, 1)
    pressure = (rng.random(n) < p).astype(int)
    yards = 8 - 4 * pressure + 0.1 * ytg + rng.normal(0, 3, n)
    return pd.DataFrame({
        "gameId": 1, "playId": np.arange(n), "quarter": rng.integers(1, 5, n), "down": down,
        "yardsToGo": ytg, "offenseFormation": form, "dropBackType": "TRADITIONAL",
        "expectedCoverage": cov, "actualCoverage": cov, "passRushers": rushers,
        "defendersInBox": box.astype(float), "pressure": pressure, "sack": (pressure & (rng.random(n) < .2)).astype(int),
        "yardsGained": yards, "possessionTeam": "AAA", "defensiveTeam": "BBB", "gameClock": "10:00",
    })


def random_situation(rng: np.random.Generator, vocab: dict) -> Situation:
    return Situation(
        down=int(rng.integers(0, 5)), yards_to_go=int(rng.integers(0, 100)),
        offense_formation=str(rng.choice(vocab["offenseFormation"])),
        expected_coverage=str(rng.choice(vocab["expectedCoverage"])),
        pass_rushers=int(rng.integers(0, 12)), defenders_in_box=int(rng.integers(0, 12)),
    )


TEAMS = ["AAA", "BBB", "CCC", "DDD"]
NAMES = ["Ñandú Pérez", "Zoë Müller", "Álvaro Ruiz", "Bob Ray", "Álvaro Ruiz", "Zed Adams"]


def report_frames(rng: np.random.Generator):
    """(dataset, scouting, players) para reportes y hallazgos, con equipos variados."""
    n = int(rng.integers(5, 80))
    ds = model_dataset(rng, n)
    ds["possessionTeam"] = rng.choice(TEAMS[:3], n)
    ds["defensiveTeam"] = rng.choice(TEAMS[1:], n)
    ds["gameId"] = rng.integers(1, 4, n)
    if rng.random() < 0.15:
        ds["pressure"] = 1
        ds["sack"] = 0
    rows = []
    for g, p in zip(ds.gameId, ds.playId):
        for _ in range(int(rng.integers(0, 6))):
            role = "Pass Rush" if rng.random() < 0.8 else str(rng.choice(["pass rush", "Pass Block"]))
            rows.append({"gameId": g, "playId": p, "nflId": int(rng.integers(1, 16)), "pff_role": role,
                         "pff_hit": rng.choice([np.nan, 0.0, 1.0]), "pff_hurry": rng.choice([np.nan, 0.0, 1.0]),
                         "pff_sack": rng.choice([np.nan, 0.0, 1.0])})
    scouting = pd.DataFrame(rows, columns=["gameId", "playId", "nflId", "pff_role", "pff_hit", "pff_hurry", "pff_sack"])
    known = [i for i in range(1, 16) if rng.random() < 0.8]
    players = pd.DataFrame({"nflId": known,
                            "displayName": [NAMES[int(rng.integers(0, len(NAMES)))] for _ in known],
                            "officialPosition": [str(rng.choice(["DE", "LB", "DT"])) for _ in known]})
    return ds, scouting, players


def write_csv_dir(path, rng: np.random.Generator, n_plays: int = 400, n_games: int = 8) -> None:
    """Escribe los cuatro CSV con la forma de los reales. Un partido queda sin jugadas de pase."""
    from pathlib import Path

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    games = []
    base = pd.Timestamp("2021-09-09")
    for g in range(n_games):
        d = base + pd.Timedelta(days=int(rng.integers(0, 60)))
        h, v = rng.choice(TEAMS, 2, replace=False)
        games.append({"gameId": 2021000000 + g, "season": 2021, "week": 1, "gameDate": d.strftime("%m/%d/%Y"),
                      "gameTimeEastern": f"{int(rng.integers(12, 21))}:00:00",
                      "homeTeamAbbr": h, "visitorTeamAbbr": v})
    games_df = pd.DataFrame(games).sample(frac=1, random_state=1)  # fuera de orden a propósito
    plays, scouting = [], []
    for i in range(n_plays):
        g = 2021000000 + int(rng.integers(0, n_games - 1))  # el último partido no tiene jugadas
        pid = i + 1
        k = int(rng.integers(3, 8))
        pressured = rng.random() < 0.1 + 0.06 * (k - 3)
        off, dfn = rng.choice(TEAMS, 2, replace=False)
        plays.append({
            "gameId": g, "playId": pid, "quarter": int(rng.choice([1, 2, 3, 4, 5], p=[.24, .24, .24, .24, .04])),
            "down": int(rng.choice([0, 1, 2, 3, 4], p=[.02, .38, .3, .27, .03])), "yardsToGo": int(rng.integers(1, 15)),
            "possessionTeam": off, "defensiveTeam": dfn,
            "gameClock": f"{int(rng.integers(0, 15)):02d}:{int(rng.integers(0, 60)):02d}",
            "offenseFormation": None if rng.random() < .03 else str(rng.choice(["SHOTGUN", "EMPTY", "I_FORM"])),
            "dropBackType": None if rng.random() < .05 else "TRADITIONAL",
            "defendersInBox": None if rng.random() < .03 else int(rng.integers(4, 9)),
            "pff_passCoverage": str(rng.choice(["Cover-1", "Cover-3", "Quarters"])),
            "playResult": int(round(8 - 4 * pressured + rng.normal(0, 3))),
        })
        scouting.append({"gameId": g, "playId": pid, "nflId": 1, "pff_role": "Pass"})
        for _ in range(k):
            hurry = int(pressured and rng.random() < .5)
            scouting.append({"gameId": g, "playId": pid, "nflId": int(rng.integers(2, 41)), "pff_role": "Pass Rush",
                             "pff_hit": 0, "pff_hurry": hurry, "pff_sack": int(hurry and rng.random() < .2)})
        for _ in range(4):
            scouting.append({"gameId": g, "playId": pid, "nflId": int(rng.integers(41, 60)), "pff_role": "Pass Block",
                             "pff_hitAllowed": 0, "pff_hurryAllowed": 0, "pff_sackAllowed": 0})
    players = pd.DataFrame({"nflId": range(1, 36), "displayName": [f"Jugador {i}" for i in range(1, 36)],
                            "officialPosition": "DE"})  # los ids 36 a 59 no existen: prueban el respaldo
    pd.DataFrame(plays).to_csv(path / "plays.csv", index=False, na_rep="NA")
    pd.DataFrame(scouting).to_csv(path / "pffScoutingData.csv", index=False, na_rep="NA")
    players.to_csv(path / "players.csv", index=False)
    games_df.to_csv(path / "games.csv", index=False)
