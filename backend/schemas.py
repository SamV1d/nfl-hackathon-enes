"""Modelos Pydantic de petición y respuesta de la API."""
from __future__ import annotations

import math
from typing import Annotated

from pydantic import BaseModel, BeforeValidator, Field


def _nan_to_none(v):
    """JSON no admite NaN: R² indefinido (varianza 0) se serializa como null."""
    return None if isinstance(v, float) and math.isnan(v) else v


OptFloat = Annotated[float | None, BeforeValidator(_nan_to_none)]


# ---- /api/vocab, /api/recommend -------------------------------------------------
class VocabOut(BaseModel):
    offense_formations: list[str]
    expected_coverages: list[str]


class RecommendRequest(BaseModel):
    down: int = Field(ge=0, le=4)
    yards_to_go: int = Field(ge=0, le=99)
    offense_formation: str = Field(min_length=1)
    expected_coverage: str = Field(min_length=1)


class CandidateCallOut(BaseModel):
    coverage: str
    pass_rushers: int
    defenders_in_box: float


class ScoredCallOut(BaseModel):
    call: CandidateCallOut
    pressure_probability: float = Field(ge=0, le=1)
    expected_yards: float


class RecommendationOut(BaseModel):
    recommendation_id: str
    best_call: ScoredCallOut
    ranked: list[ScoredCallOut]
    typical_call: CandidateCallOut | None
    yards_delta_vs_typical: float | None
    two_point_conversion: bool
    low_confidence: bool
    n_matches: int


# ---- /api/metrics ---------------------------------------------------------------
class SplitCounts(BaseModel):
    n_train: int
    n_heldout: int
    n_train_used: int
    n_heldout_scored: int


class PressureMetrics(SplitCounts):
    r2: OptFloat
    mae: OptFloat


class YardsMetrics(SplitCounts):
    r2: OptFloat
    mae_yards: OptFloat


class MetricsOut(BaseModel):
    pressure_model: PressureMetrics
    yards_model: YardsMetrics


# ---- /api/games, /api/games/{id}/plays -------------------------------------------
class GameOut(BaseModel):
    game_id: int
    date: str
    home_team: str
    visitor_team: str


class ReplayPlayOut(BaseModel):
    position: int
    play_id: int
    quarter: int
    quarter_label: str
    down: int
    down_label: str
    yards_to_go: int
    offense_formation: str
    expected_coverage: str
    game_clock: str
    best_call: ScoredCallOut | None
    actual_call: CandidateCallOut
    pressure: bool
    sack: bool
    yards_gained: float | None
    two_point_conversion: bool
    overtime: bool


class ReplayOut(BaseModel):
    game_id: int
    total: int
    message: str | None = None
    plays: list[ReplayPlayOut]


# ---- /api/teams -------------------------------------------------------------------
class GroupStatOut(BaseModel):
    group: str
    count: int
    pressure_rate: float | None
    avg_yards: float | None


class SideReportOut(BaseModel):
    pass_plays: int
    pressure_rate: float | None
    sack_rate: float | None
    avg_yards: float | None
    by_coverage: list[GroupStatOut]
    by_formation: list[GroupStatOut]


class PlayerCountOut(BaseModel):
    player_id: int
    name: str
    position: str
    count: int


class LeagueAverageOut(BaseModel):
    pass_plays: int
    pressure_rate: float | None
    sack_rate: float | None
    avg_yards: float | None


class TeamReportOut(BaseModel):
    team: str
    has_data: bool
    message: str | None
    offense: SideReportOut | None
    defense: SideReportOut | None
    top_sackers: list[PlayerCountOut]
    top_pressure: list[PlayerCountOut]
    league_average: LeagueAverageOut | None


class TeamsOut(BaseModel):
    teams: list[str]


# ---- /api/findings ----------------------------------------------------------------
class FindingsSummary(BaseModel):
    pressure_rate: float | None
    avg_yards_with_pressure: float | None
    avg_yards_without_pressure: float | None
    sack_rate_with_pressure: float | None
    sack_rate_without_pressure: float | None


class TeamRankOut(BaseModel):
    team: str
    pressure_rate: float | None
    pass_plays: int


class FindingsOut(BaseModel):
    summary: FindingsSummary
    by_coverage: list[GroupStatOut]
    by_formation: list[GroupStatOut]
    by_defenders_in_box: list[GroupStatOut]
    top_sackers: list[PlayerCountOut]
    top_teams: list[TeamRankOut]


# ---- /api/recommendations/{id}/explain, errores -----------------------------------
class ExplainOut(BaseModel):
    explanation: str = Field(max_length=1000)


class ErrorOut(BaseModel):
    detail: str
    fields: list[str] | None = None
    missing_files: list[str] | None = None
