"""Estado en memoria de la aplicación, construido una vez al arrancar."""
from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import reports
from .cleaning import clean_plays, train_median_defenders_in_box
from .dataset import build_play_dataset
from .linreg import split_indices
from .loader import read_all
from .ml.service import AuditCache
from .models import PressureModel, YardsModel, fit_models
from .recommender import CandidateCall, Recommendation, candidate_calls

log = logging.getLogger("backend.state")

SEED = 42
MAX_STORED_RECOMMENDATIONS = 1000


@dataclass(frozen=True)
class AppState:
    games: pd.DataFrame
    players: pd.DataFrame
    dataset: pd.DataFrame
    train_idx: np.ndarray
    heldout_idx: np.ndarray
    pressure_model: PressureModel | None
    yards_model: YardsModel | None
    fit_errors: dict[str, str]
    vocab: dict[str, list[str]]
    candidates: list[CandidateCall]
    findings: dict
    team_stats: dict
    league_avg: dict
    teams: list[str]
    recommendations: "OrderedDict[str, Recommendation]" = field(default_factory=OrderedDict)
    lock: threading.Lock = field(default_factory=threading.Lock)
    plays: pd.DataFrame | None = None  # plays.csv sin tocar; lo usa la auditoria de modelos
    audit: AuditCache = field(default_factory=AuditCache)

    def store(self, rec: Recommendation) -> None:
        with self.lock:
            self.recommendations[rec.recommendation_id] = rec
            while len(self.recommendations) > MAX_STORED_RECOMMENDATIONS:
                self.recommendations.popitem(last=False)

    def lookup(self, rec_id: str) -> Recommendation | None:
        with self.lock:
            return self.recommendations.get(rec_id)


def build_state(data_dir: Path, seed: int = SEED) -> AppState:
    """Carga, limpia, ajusta modelos y precalcula. Lanza StartupError si faltan o fallan CSV."""
    t0 = time.monotonic()
    files = read_all(data_dir)
    scouting, players, games = files["pffScoutingData.csv"], files["players.csv"], files["games.csv"]

    dataset = build_play_dataset(files["plays.csv"], scouting)
    train_idx, heldout_idx = split_indices(len(dataset), seed)
    median = train_median_defenders_in_box(dataset, train_idx)
    dataset = clean_plays(dataset, median)

    fit = fit_models(dataset, train_idx, heldout_idx)
    for name, reason in fit.errors.items():
        log.error("modelo %s no disponible: %s", name, reason)

    pm, ym = fit.pressure_model, fit.yards_model
    candidates = candidate_calls(dataset.iloc[train_idx]) if pm and ym else []
    team_stats, league = reports.build_team_stats(dataset, scouting, players)

    state = AppState(
        games=games, players=players, dataset=dataset, train_idx=train_idx, heldout_idx=heldout_idx,
        pressure_model=pm, yards_model=ym, fit_errors=fit.errors,
        vocab=pm.vocab if pm else {}, candidates=candidates,
        findings=reports.compute_findings(dataset, scouting, players),
        team_stats=team_stats, league_avg=league, teams=reports.list_teams(dataset),
        plays=files["plays.csv"],
    )
    log.info("listo en %.1f s: %d jugadas de pase, %d candidatas", time.monotonic() - t0,
             len(dataset), len(candidates))
    return state
