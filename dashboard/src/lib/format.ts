// Funciones puras de formato. Nunca devuelven NaN, Infinity ni cadena vacía.

export const NA = 'N/A'

const isNum = (n: unknown): n is number => typeof n === 'number' && Number.isFinite(n)

/** Un decimal, sin "-0.0". */
function fixed1(n: number): string {
  const s = n.toFixed(1)
  return s === '-0.0' ? '0.0' : s
}

/** Fracción (0.3713) a porcentaje con un decimal ("37.1%"). */
export const formatPercent = (p: number | null | undefined): string =>
  isNum(p) ? `${fixed1(p * 100)}%` : NA

/** Yardas con un decimal ("7.3 yardas"). */
export const formatYards = (y: number | null | undefined): string =>
  isNum(y) ? `${fixed1(y)} yardas` : NA

/** Diferencia de yardas con signo ("+1.2 yardas"). */
export const formatSignedYards = (y: number | null | undefined): string => {
  if (!isNum(y)) return NA
  const s = fixed1(y)
  return `${y > 0 && s !== '0.0' ? '+' : ''}${s} yardas`
}

/** num/den con unidad; "N/A" si el denominador es cero o algún valor no es finito. */
export function formatOrNA(num: number, den: number, unit: '%' | 'yards'): string {
  if (!isNum(num) || !isNum(den) || den === 0) return NA
  const v = num / den
  if (!isNum(v)) return NA
  return unit === '%' ? `${fixed1(v * 100)}%` : `${fixed1(v)} yardas`
}

/** Métricas de modelo (R², MAE) con 3 decimales. */
export const formatMetric = (v: number | null | undefined, digits = 3): string =>
  isNum(v) ? v.toFixed(digits) : NA

export const quarterLabel = (q: number): string => (q === 5 ? 'overtime' : String(q))

export const downLabel = (d: number): string => (d === 0 ? 'two-point conversion' : String(d))
