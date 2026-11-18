import { Component, type ErrorInfo, type ReactNode } from 'react'
import { describeError } from '../api'
import { formatPercent, formatYards } from '../lib/format'
import type { Call, ScoredCall } from '../types'

export function Spinner({ label = 'Cargando…' }: { label?: string }) {
  return (
    <div role="status" className="flex items-center gap-2 text-sm text-gray-600">
      <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-gray-300 border-t-green-700" />
      {label}
    </div>
  )
}

export function ErrorBox({ error, message, onRetry }: { error?: unknown; message?: string; onRetry?: () => void }) {
  return (
    <div role="alert" className="rounded border border-red-300 bg-red-50 p-3 text-sm text-red-800">
      <p>{message ?? describeError(error)}</p>
      {onRetry && (
        <button type="button" onClick={onRetry} className="mt-2 rounded bg-red-700 px-3 py-1 text-white hover:bg-red-800">
          Reintentar
        </button>
      )}
    </div>
  )
}

export function Card({ title, children, className = '' }: { title?: string; children: ReactNode; className?: string }) {
  return (
    <section className={`rounded-lg border border-gray-200 bg-white p-4 shadow-sm ${className}`}>
      {title && <h3 className="mb-3 text-base font-semibold text-gray-800">{title}</h3>}
      {children}
    </section>
  )
}

export function Badge({ children, tone = 'gray' }: { children: ReactNode; tone?: 'gray' | 'amber' | 'blue' | 'green' | 'red' }) {
  const tones = {
    gray: 'bg-gray-100 text-gray-700',
    amber: 'bg-amber-100 text-amber-800',
    blue: 'bg-blue-100 text-blue-800',
    green: 'bg-green-100 text-green-800',
    red: 'bg-red-100 text-red-800',
  }
  return <span className={`inline-block rounded px-2 py-0.5 text-xs font-medium ${tones[tone]}`}>{children}</span>
}

export function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded border border-gray-200 bg-gray-50 p-3">
      <div className="text-xs text-gray-500">{label}</div>
      <div className="text-xl font-semibold text-gray-900">{value}</div>
    </div>
  )
}

export const callText = (c: Call): string =>
  `${c.coverage} · ${c.pass_rushers} rushers · ${c.defenders_in_box} en la caja`

export const scoredText = (s: ScoredCall): string =>
  `${callText(s.call)} (presión ${formatPercent(s.pressure_probability)}, ${formatYards(s.expected_yards)})`

export const sameCall = (a: Call, b: Call): boolean =>
  a.coverage === b.coverage && a.pass_rushers === b.pass_rushers && a.defenders_in_box === b.defenders_in_box

/** Aísla fallos de render: una vista rota no tumba a las demás. */
export class ViewBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false }
  static getDerivedStateFromError() {
    return { failed: true }
  }
  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error(error, info.componentStack)
  }
  render() {
    if (this.state.failed) {
      return <ErrorBox message="Esta vista tuvo un error al mostrarse." onRetry={() => this.setState({ failed: false })} />
    }
    return this.props.children
  }
}

export const selectCls = 'w-full rounded border border-gray-300 bg-white px-2 py-1.5 text-sm'
export const btnCls =
  'rounded bg-green-700 px-3 py-1.5 text-sm font-medium text-white hover:bg-green-800 disabled:cursor-not-allowed disabled:bg-gray-300'
export const btnSecondaryCls =
  'rounded border border-gray-300 bg-white px-3 py-1.5 text-sm hover:bg-gray-50 disabled:cursor-not-allowed disabled:text-gray-400'
