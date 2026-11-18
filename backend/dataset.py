"""Dataset a nivel de jugada: una fila por Pass_Play."""
from __future__ import annotations

import pandas as pd

KEYS = ["gameId", "playId"]
ROLE_RUSH, ROLE_BLOCK = "Pass Rush", "Pass Block"  # comparaciÃ³n sensible a mayÃºsculas

PLAYS_REQUIRED = (
    "gameId", "playId", "quarter", "down", "yardsToGo", "possessionTeam", "defensiveTeam",
    "gameClock", "offenseFormation", "dropBackType", "defendersInBox", "pff_passCoverage",
    "playResult",
)
SCOUTING_REQUIRED = (
    "gameId", "playId", "nflId", "pff_role", "pff_hit", "pff_hurry", "pff_sack",
    "pff_hitAllowed", "pff_hurryAllowed", "pff_sackAllowed",
)

OUTPUT_COLUMNS = [
    "gameId", "playId", "quarter", "down", "yardsToGo", "offenseFormation", "dropBackType",
    "expectedCoverage", "actualCoverage", "passRushers", "defendersInBox",
    "pressure", "sack", "yardsGained", "possessionTeam", "defensiveTeam", "gameClock",
    "passBlockers", "rushHits", "rushHurries", "rushSacks",
    "hitsAllowed", "hurriesAllowed", "sacksAllowed",
]


class DataError(Exception):
    """Falta una columna necesaria. `missing` = [(archivo, columna), ...]."""

    def __init__(self, missing: list[tuple[str, str]]):
        self.missing = missing
        self.file, self.column = missing[0]
        super().__init__(
            "Columnas faltantes: " + ", ".join(f"{f}:{c}" for f, c in missing)
        )


def nonzero(s: pd.Series) -> pd.Series:
    """True donde el valor es numÃ©rico y distinto de 0 (nulo = no aplica = False)."""
    return pd.to_numeric(s, errors="coerce").fillna(0) != 0


def _check_columns(plays: pd.DataFrame, scouting: pd.DataFrame) -> None:
    missing = [("plays.csv", c) for c in PLAYS_REQUIRED if c not in plays.columns]
    missing += [("pffScoutingData.csv", c) for c in SCOUTING_REQUIRED if c not in scouting.columns]
    if missing:
        raise DataError(missing)


def build_play_dataset(plays: pd.DataFrame, scouting: pd.DataFrame) -> pd.DataFrame:
    """Une plays y pffScoutingData por (gameId, playId) y devuelve una fila por Pass_Play.

    Pass_Play = jugada con al menos una fila con rol "Pass Rush" o "Pass Block".
    PresiÃ³n/sack salen solo de filas "Pass Rush"; los *Allowed solo de "Pass Block".
    Los nulos de columnas de otro rol no se cuentan ni se imputan.
    No limpia nulos de plays (eso lo hace clean_plays). Lanza DataError sin salida parcial.
    """
    _check_columns(plays, scouting)

    rush = scouting[scouting["pff_role"] == ROLE_RUSH]
    block = scouting[scouting["pff_role"] == ROLE_BLOCK]

    rush_agg = (
        rush.assign(
            _hit=nonzero(rush["pff_hit"]),
            _hurry=nonzero(rush["pff_hurry"]),
            _sack=nonzero(rush["pff_sack"]),
        )
        .groupby(KEYS)
        .agg(passRushers=("pff_role", "size"), rushHits=("_hit", "sum"),
             rushHurries=("_hurry", "sum"), rushSacks=("_sack", "sum"))
        .reset_index()
    )
    block_agg = (
        block.assign(
            _hit=nonzero(block["pff_hitAllowed"]),
            _hurry=nonzero(block["pff_hurryAllowed"]),
            _sack=nonzero(block["pff_sackAllowed"]),
        )
        .groupby(KEYS)
        .agg(passBlockers=("pff_role", "size"), hitsAllowed=("_hit", "sum"),
             hurriesAllowed=("_hurry", "sum"), sacksAllowed=("_sack", "sum"))
        .reset_index()
    )

    pass_keys = pd.concat([rush_agg[KEYS], block_agg[KEYS]]).drop_duplicates()
    base = plays.drop_duplicates(KEYS).merge(pass_keys, on=KEYS, how="inner")
    df = base.merge(rush_agg, on=KEYS, how="left").merge(block_agg, on=KEYS, how="left")

    count_cols = ["passRushers", "rushHits", "rushHurries", "rushSacks",
                  "passBlockers", "hitsAllowed", "hurriesAllowed", "sacksAllowed"]
    df[count_cols] = df[count_cols].fillna(0).astype(int)

    coverage = df["pff_passCoverage"].fillna("Unknown")
    df["expectedCoverage"] = coverage
    df["actualCoverage"] = coverage
    df["pressure"] = ((df["rushHits"] + df["rushHurries"] + df["rushSacks"]) > 0).astype(int)
    df["sack"] = (df["rushSacks"] > 0).astype(int)
    df["yardsGained"] = pd.to_numeric(df["playResult"], errors="coerce")

    return df[OUTPUT_COLUMNS].sort_values(KEYS).reset_index(drop=True)
