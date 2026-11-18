"""Ingeniería de variables y codificación para los modelos.

Cada variable declara cuándo se conoce. Esto evita mezclar información previa y posterior
a la jugada (fuga de datos):

* PRE_SNAP: el cuerpo técnico la conoce antes del snap (situación, personal, formación).
* DEFENSIVE_CHOICE: decisión defensiva que se evalúa en What-If. Los datos la traen
  etiquetada por PFF después de la jugada (cobertura y número de rushers), de modo que es
  una variable de entrada del escenario y no una observación que el rival conozca.
* POST_SNAP: solo se observa al ejecutar la jugada. NO se usa como entrada de ningún modelo.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

PRE_SNAP, DEFENSIVE_CHOICE, POST_SNAP = "pre_snap", "defensive_choice", "post_snap"

# nombre -> (grupo de interfaz, momento en que se conoce)
FEATURE_CATALOG: dict[str, tuple[str, str]] = {
    "down": ("situacion", PRE_SNAP),
    "yardsToGo": ("situacion", PRE_SNAP),
    "quarter": ("situacion", PRE_SNAP),
    "secondsInQuarter": ("situacion", PRE_SNAP),
    "halfSecondsLeft": ("situacion", PRE_SNAP),
    "scoreDiffOffense": ("situacion", PRE_SNAP),
    "yardsToGoal": ("situacion", PRE_SNAP),
    "offenseFormation": ("ofensivo", PRE_SNAP),
    "offRB": ("ofensivo", PRE_SNAP),
    "offTE": ("ofensivo", PRE_SNAP),
    "offWR": ("ofensivo", PRE_SNAP),
    "defendersInBox": ("defensivo", PRE_SNAP),
    "defDL": ("defensivo", PRE_SNAP),
    "defLB": ("defensivo", PRE_SNAP),
    "defDB": ("defensivo", PRE_SNAP),
    "expectedCoverage": ("defensivo", DEFENSIVE_CHOICE),
    "passRushers": ("defensivo", DEFENSIVE_CHOICE),
    # Observadas después de la jugada: documentadas aquí y excluidas de los modelos.
    "dropBackType": ("resultado", POST_SNAP),
    "pff_playAction": ("resultado", POST_SNAP),
    "passResult": ("resultado", POST_SNAP),
}

NUMERIC = ["down", "yardsToGo", "quarter", "secondsInQuarter", "halfSecondsLeft",
           "scoreDiffOffense", "yardsToGoal", "offRB", "offTE", "offWR",
           "defendersInBox", "defDL", "defLB", "defDB", "passRushers"]
CATEGORICAL = ["offenseFormation", "expectedCoverage"]
MODEL_FEATURES = NUMERIC + CATEGORICAL
MIN_CATEGORY_COUNT = 30
OTHER = "Other"


def used_features_by_timing() -> dict[str, list[str]]:
    """Variables de los modelos agrupadas por momento de disponibilidad (para documentar)."""
    out: dict[str, list[str]] = {PRE_SNAP: [], DEFENSIVE_CHOICE: [], POST_SNAP: []}
    for f in MODEL_FEATURES:
        out[FEATURE_CATALOG[f][1]].append(f)
    return out


def _count(text, label: str) -> float:
    if not isinstance(text, str):
        return np.nan
    m = re.search(rf"(\d+)\s*{label}", text)
    return float(m.group(1)) if m else 0.0


def _clock_seconds(clock) -> float:
    try:
        m, s = str(clock).split(":")[:2]
        return float(int(m) * 60 + int(s))
    except (ValueError, TypeError):
        return np.nan


def build_ml_frame(dataset: pd.DataFrame, plays: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Une el dataset de jugadas con columnas adicionales de plays/games.

    No modifica `dataset`. Devuelve una fila por jugada con variables y objetivos:
    pressure, sack, netYards (playResult) y attemptYards (NaN en sacks).
    """
    keys = ["gameId", "playId"]
    extra_cols = keys + ["personnelO", "personnelD", "preSnapHomeScore", "preSnapVisitorScore",
                         "yardlineSide", "yardlineNumber", "passResult", "prePenaltyPlayResult"]
    extra = plays[extra_cols].drop_duplicates(keys)
    g = games[["gameId", "week", "homeTeamAbbr", "visitorTeamAbbr"]]
    df = dataset.merge(extra, on=keys, how="left").merge(g, on="gameId", how="left")

    secs = df["gameClock"].map(_clock_seconds)
    df["secondsInQuarter"] = secs
    in_first_quarter_of_half = df["quarter"].isin([1, 3])
    df["halfSecondsLeft"] = np.where(df["quarter"] <= 4, secs + np.where(in_first_quarter_of_half, 900, 0), secs)

    possession_home = df["possessionTeam"] == df["homeTeamAbbr"]
    df["scoreDiffOffense"] = np.where(
        possession_home, df["preSnapHomeScore"] - df["preSnapVisitorScore"],
        df["preSnapVisitorScore"] - df["preSnapHomeScore"])

    own_side = df["yardlineSide"] == df["possessionTeam"]
    to_goal = np.where(own_side, 100 - df["yardlineNumber"], df["yardlineNumber"])
    df["yardsToGoal"] = np.where(df["yardlineSide"].isna(), 50.0, to_goal)

    for col, label in (("offRB", "RB"), ("offTE", "TE"), ("offWR", "WR")):
        df[col] = df["personnelO"].map(lambda t, l=label: _count(t, l))
    for col, label in (("defDL", "DL"), ("defLB", "LB"), ("defDB", "DB")):
        df[col] = df["personnelD"].map(lambda t, l=label: _count(t, l))

    df["offenseFormation"] = df["offenseFormation"].fillna("Unknown")
    df["netYards"] = pd.to_numeric(df["yardsGained"], errors="coerce")
    df["attemptYards"] = df["netYards"].where(df["passResult"] != "S")
    return df


class FeatureEncoder:
    """Numéricas (imputadas con la mediana de entrenamiento) + one-hot de categóricas.

    Las categorías con menos de MIN_CATEGORY_COUNT filas en entrenamiento se agrupan en
    "Other". Una categoría no vista en entrenamiento también se agrupa en "Other" y se
    reporta como fuera de distribución mediante `out_of_distribution`.
    """

    def __init__(self, numeric=None, categorical=None):
        self.numeric = list(NUMERIC if numeric is None else numeric)
        self.categorical = list(CATEGORICAL if categorical is None else categorical)

    def fit(self, df: pd.DataFrame) -> "FeatureEncoder":
        self.medians_ = {c: float(pd.to_numeric(df[c], errors="coerce").median()) for c in self.numeric}
        self.ranges_ = {c: (float(pd.to_numeric(df[c], errors="coerce").min()),
                            float(pd.to_numeric(df[c], errors="coerce").max())) for c in self.numeric}
        self.vocab_ = {}
        for c in self.categorical:
            counts = df[c].fillna("Unknown").astype(str).value_counts()
            keep = sorted(counts[counts >= MIN_CATEGORY_COUNT].index)
            if (counts < MIN_CATEGORY_COUNT).any() and OTHER not in keep:
                keep.append(OTHER)
            self.vocab_[c] = keep
        return self

    @property
    def feature_names(self) -> list[str]:
        names = list(self.numeric)
        for c, cats in self.vocab_.items():
            names += [f"{c}={k}" for k in cats]
        return names

    def _cat(self, s: pd.Series, col: str) -> pd.Series:
        known = set(self.vocab_[col])
        s = s.fillna("Unknown").astype(str)
        return s.where(s.isin(known), OTHER)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        n = len(df)
        blocks = []
        if self.numeric:
            num = np.column_stack([
                pd.to_numeric(df[c], errors="coerce").fillna(self.medians_[c]).to_numpy(dtype=float)
                for c in self.numeric])
            blocks.append(num)
        for c in self.categorical:
            vals = self._cat(df[c], c).to_numpy()
            for k in self.vocab_[c]:
                blocks.append((vals == k).astype(float).reshape(n, 1))
        return np.hstack(blocks) if blocks else np.zeros((n, 0))

    def out_of_distribution(self, df: pd.DataFrame) -> list[str]:
        """Variables de una única fila (o varias) fuera del soporte de entrenamiento."""
        flags: list[str] = []
        for c in self.numeric:
            v = pd.to_numeric(df[c], errors="coerce")
            lo, hi = self.ranges_[c]
            if ((v < lo) | (v > hi)).any():
                flags.append(c)
        for c in self.categorical:
            s = df[c].fillna("Unknown").astype(str)
            seen = set(self.vocab_[c]) - {OTHER}
            if (~s.isin(seen)).any():
                flags.append(c)
        return flags


def temporal_split(df: pd.DataFrame, train_weeks=(1, 2, 3, 4, 5), valid_weeks=(6,),
                   test_weeks=(7, 8)) -> dict[str, np.ndarray]:
    """Índices (posicionales) por semana. Las tres partes no comparten jugadas ni partidos."""
    wk = df["week"].to_numpy()
    parts = {"train": np.flatnonzero(np.isin(wk, train_weeks)),
             "valid": np.flatnonzero(np.isin(wk, valid_weeks)),
             "test": np.flatnonzero(np.isin(wk, test_weeks))}
    games = {k: set(df["gameId"].iloc[v]) for k, v in parts.items()}
    if games["train"] & games["valid"] or games["train"] & games["test"] or games["valid"] & games["test"]:
        raise ValueError("la partición temporal comparte partidos entre conjuntos")
    return parts
