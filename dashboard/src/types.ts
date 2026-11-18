// Tipos del contrato de la API. Las tasas y probabilidades son fracciones (0 a 1).

export interface Call {
  coverage: string
  pass_rushers: number
  defenders_in_box: number
}

export interface ScoredCall {
  call: Call
  pressure_probability: number
  expected_yards: number
}

export interface Vocab {
  offense_formations: string[]
  expected_coverages: string[]
}

export interface RecommendRequest {
  down: number
  yards_to_go: number
  offense_formation: string
  expected_coverage: string
}

export interface Recommendation {
  recommendation_id: string
  best_call: ScoredCall
  ranked: ScoredCall[]
  typical_call: Call | null
  yards_delta_vs_typical: number | null
  two_point_conversion: boolean
  low_confidence: boolean
  n_matches: number
}

export interface ExplainResponse {
  explanation: string
}

export interface ModelMetrics {
  r2: number | null
  mae?: number
  mae_yards?: number
  n_train: number
  n_heldout: number
  n_train_used: number
  n_heldout_scored: number
}

export interface MetricsResponse {
  pressure_model: ModelMetrics
  yards_model: ModelMetrics
}

export interface Game {
  game_id: number
  date: string
  home_team: string
  visitor_team: string
}

export interface ReplayPlay {
  position: number
  play_id: number
  quarter: number
  quarter_label: string
  down: number
  down_label: string
  yards_to_go: number
  offense_formation: string
  expected_coverage: string
  game_clock: string
  best_call: ScoredCall | null
  actual_call: Call
  pressure: boolean
  sack: boolean
  yards_gained: number | null
  two_point_conversion: boolean
  overtime: boolean
}

export interface PlaysResponse {
  game_id: number
  total: number
  message: string | null
  plays: ReplayPlay[]
}

export interface Group {
  group: string
  count: number
  pressure_rate: number | null
  avg_yards: number | null
}

export interface Side {
  pass_plays: number
  pressure_rate: number | null
  sack_rate: number | null
  avg_yards: number | null
  by_coverage: Group[]
  by_formation: Group[]
}

export interface Player {
  player_id: string
  name: string
  position: string
  count: number
}

export interface LeagueAverage {
  pass_plays: number
  pressure_rate: number | null
  sack_rate: number | null
  avg_yards: number | null
}

export interface TeamReport {
  team: string
  has_data: boolean
  message: string | null
  offense: Side | null
  defense: Side | null
  top_sackers: Player[]
  top_pressure: Player[]
  league_average: LeagueAverage | null
}

export interface TeamsResponse {
  teams: string[]
}

export interface FindingsSummary {
  pressure_rate: number | null
  avg_yards_with_pressure: number | null
  avg_yards_without_pressure: number | null
  sack_rate_with_pressure: number | null
  sack_rate_without_pressure: number | null
}

export interface TopTeam {
  team: string
  pressure_rate: number | null
  pass_plays: number
}

export interface Findings {
  summary: FindingsSummary
  by_coverage: Group[]
  by_formation: Group[]
  by_defenders_in_box: Group[]
  top_sackers: Player[]
  top_teams: TopTeam[]
}
