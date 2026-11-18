// Navegación del replay y comparación acumulada. Funciones puras.
import type { ReplayPlay } from '../types'

/** index es 0-based; total es el número de jugadas (posiciones 1..total). */
export interface NavState {
  index: number
  total: number
}

export type NavAction = { type: 'next' } | { type: 'prev' } | { type: 'jump'; position: number }

export const canPrev = (s: NavState): boolean => s.total > 0 && s.index > 0
export const canNext = (s: NavState): boolean => s.total > 0 && s.index < s.total - 1

export function navReduce(s: NavState, a: NavAction): NavState {
  if (s.total <= 0) return s
  switch (a.type) {
    case 'next':
      return canNext(s) ? { ...s, index: s.index + 1 } : s
    case 'prev':
      return canPrev(s) ? { ...s, index: s.index - 1 } : s
    case 'jump':
      return Number.isInteger(a.position) && a.position >= 1 && a.position <= s.total
        ? { ...s, index: a.position - 1 }
        : s
  }
}

const round1 = (n: number): number => {
  const r = Math.round(n * 10) / 10
  return r === 0 ? 0 : r
}

export interface Cumulative {
  actual: number
  predicted: number
  diff: number
  /** Jugadas sin Best_Call (omitidas del lado predicho). */
  omittedPredicted: number
  /** Jugadas sin yardas reales (omitidas del lado real). */
  omittedActual: number
}

/**
 * Suma de las jugadas 1..upTo (posición 1-based, inclusive).
 * actual = suma de yards_gained; predicted = suma de best_call.expected_yards;
 * diff = actual - predicted. Todo redondeado a 1 decimal.
 */
export function cumulative(plays: ReplayPlay[], upTo: number): Cumulative {
  const n = Math.max(0, Math.min(Math.floor(upTo), plays.length))
  let actual = 0
  let predicted = 0
  let omittedPredicted = 0
  let omittedActual = 0
  for (let i = 0; i < n; i++) {
    const p = plays[i]
    if (p.yards_gained === null) omittedActual++
    else actual += p.yards_gained
    if (p.best_call === null) omittedPredicted++
    else predicted += p.best_call.expected_yards
  }
  return {
    actual: round1(actual),
    predicted: round1(predicted),
    diff: round1(actual - predicted),
    omittedPredicted,
    omittedActual,
  }
}
