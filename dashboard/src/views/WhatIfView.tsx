import { useEffect, useRef, useState } from 'react'
import { ApiError, describeError, FIELD_LABELS, isAbort, postJson } from '../api'
import { useGet } from '../hooks/useGet'
import { formatMetric, formatPercent, formatSignedYards, formatYards } from '../lib/format'
import { invalidFields, validateWhatIf, type WhatIfInput } from '../lib/validate'
import type { ExplainResponse, MetricsResponse, Recommendation, Vocab } from '../types'
import { Badge, btnCls, btnSecondaryCls, callText, Card, ErrorBox, sameCall, selectCls, Spinner } from '../components/ui'

const PAGE = 10
const EMPTY: WhatIfInput = { down: '', yardsToGo: '', formation: '', coverage: '' }

type Status = { loading: boolean; error: unknown | null; validation: string[] }

export default function WhatIfView() {
  const vocab = useGet<Vocab>('vocab')
  const [form, setForm] = useState<WhatIfInput>(EMPTY)
  const [result, setResult] = useState<Recommendation | null>(null)
  const [resultInput, setResultInput] = useState<WhatIfInput | null>(null)
  const [outdated, setOutdated] = useState(false)
  const [status, setStatus] = useState<Status>({ loading: false, error: null, validation: [] })
  const [visible, setVisible] = useState(PAGE)
  const ctlRef = useRef<AbortController | null>(null)
  const dirtyRef = useRef(false)

  useEffect(() => () => ctlRef.current?.abort(), [])

  const complete = Object.values(form).every((v) => v.trim() !== '')

  const change = (key: keyof WhatIfInput, value: string) => {
    setForm((f) => ({ ...f, [key]: value }))
    dirtyRef.current = true
    if (result) setOutdated(true) // el resultado anterior sigue visible, marcado como desactualizado
  }

  const submit = async () => {
    const messages = validateWhatIf(form)
    if (messages.length > 0) {
      setStatus({ loading: false, error: null, validation: messages })
      return
    }
    ctlRef.current?.abort()
    const ctl = new AbortController()
    ctlRef.current = ctl
    dirtyRef.current = false
    const sent = { ...form }
    setStatus({ loading: true, error: null, validation: [] })
    try {
      const r = await postJson<Recommendation>(
        'recommend',
        {
          down: Number(sent.down),
          yards_to_go: Number(sent.yardsToGo),
          offense_formation: sent.formation,
          expected_coverage: sent.coverage,
        },
        ctl.signal,
      )
      setResult(r)
      setResultInput(sent)
      setOutdated(dirtyRef.current)
      setVisible(PAGE)
      setStatus({ loading: false, error: null, validation: [] })
    } catch (e) {
      if (isAbort(e)) return
      setStatus({ loading: false, error: e, validation: [] })
    }
  }

  const bad = new Set(status.validation.length > 0 ? invalidFields(form) : [])
  const fieldCls = (k: keyof WhatIfInput) => `${selectCls} ${bad.has(k) ? 'border-red-500' : ''}`

  return (
    <div className="space-y-4">
      <Card title="Situación de juego">
        {vocab.loading && !vocab.data && <Spinner />}
        {vocab.error !== null && !vocab.data && <ErrorBox error={vocab.error} onRetry={vocab.retry} />}
        {vocab.data && (
          <form
            className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4"
            onSubmit={(e) => {
              e.preventDefault()
              void submit()
            }}
            noValidate
          >
            <label className="text-sm">
              Down (0 a 4)
              <input className={fieldCls('down')} inputMode="numeric" value={form.down} onChange={(e) => change('down', e.target.value)} />
            </label>
            <label className="text-sm">
              Yardas por avanzar (0 a 99)
              <input className={fieldCls('yardsToGo')} inputMode="numeric" value={form.yardsToGo} onChange={(e) => change('yardsToGo', e.target.value)} />
            </label>
            <label className="text-sm">
              Formación ofensiva
              <select className={fieldCls('formation')} value={form.formation} onChange={(e) => change('formation', e.target.value)}>
                <option value="">Elige…</option>
                {vocab.data.offense_formations.map((o) => (
                  <option key={o} value={o}>
                    {o}
                  </option>
                ))}
              </select>
            </label>
            <label className="text-sm">
              Cobertura esperada
              <select className={fieldCls('coverage')} value={form.coverage} onChange={(e) => change('coverage', e.target.value)}>
                <option value="">Elige…</option>
                {vocab.data.expected_coverages.map((o) => (
                  <option key={o} value={o}>
                    {o}
                  </option>
                ))}
              </select>
            </label>
            <div className="flex items-center gap-3 sm:col-span-2 lg:col-span-4">
              <button type="submit" className={btnCls} disabled={!complete || status.loading}>
                Recomendar
              </button>
              {status.loading && <Spinner label="Calculando…" />}
            </div>
          </form>
        )}
        {status.validation.length > 0 && (
          <ul role="alert" className="mt-3 list-disc rounded border border-amber-300 bg-amber-50 p-3 pl-7 text-sm text-amber-900">
            {status.validation.map((m) => (
              <li key={m}>{m}</li>
            ))}
          </ul>
        )}
        {status.error !== null && (
          <div className="mt-3">
            <ErrorBox
              message={
                status.error instanceof ApiError && status.error.status === 422 && status.error.fields.length > 0
                  ? `Entrada no válida en: ${status.error.fields.map((f) => FIELD_LABELS[f] ?? f).join(', ')}.`
                  : `La solicitud falló. ${describeError(status.error)}`
              }
              onRetry={() => void submit()}
            />
          </div>
        )}
      </Card>

      {result && resultInput && (
        <div className="space-y-4">
          {outdated && (
            <div role="status" className="rounded border border-amber-300 bg-amber-50 p-2 text-sm text-amber-900">
              Desactualizado: cambiaste los datos después de este resultado. Vuelve a pulsar «Recomendar».
            </div>
          )}
          <div className={outdated ? 'space-y-4 opacity-60' : 'space-y-4'}>
            <ResultSummary result={result} input={resultInput} />
            <Ranking result={result} visible={visible} setVisible={setVisible} />
            <ExplainPanel key={result.recommendation_id} id={result.recommendation_id} />
          </div>
        </div>
      )}

      <MetricsPanel />
    </div>
  )
}

function ResultSummary({ result, input }: { result: Recommendation; input: WhatIfInput }) {
  const typicalIdx = result.typical_call ? result.ranked.findIndex((r) => sameCall(r.call, result.typical_call!)) : -1
  return (
    <Card title="Recomendación">
      <p className="text-xs text-gray-500">
        Para: down {input.down}, {input.yardsToGo} yardas, {input.formation}, {input.coverage}
      </p>
      <div className="mt-2 flex flex-wrap items-center gap-2">
        <span className="text-lg font-semibold">{callText(result.best_call.call)}</span>
        {result.two_point_conversion && <Badge tone="blue">Conversión de dos puntos</Badge>}
        {result.low_confidence && <Badge tone="amber">Baja confianza</Badge>}
      </div>
      <dl className="mt-3 grid grid-cols-2 gap-3 text-sm sm:grid-cols-4">
        <div>
          <dt className="text-gray-500">Probabilidad de presión</dt>
          <dd className="text-lg font-semibold">{formatPercent(result.best_call.pressure_probability)}</dd>
        </div>
        <div>
          <dt className="text-gray-500">Yardas esperadas</dt>
          <dd className="text-lg font-semibold">{formatYards(result.best_call.expected_yards)}</dd>
        </div>
        <div>
          <dt className="text-gray-500">Jugadas parecidas</dt>
          <dd className="text-lg font-semibold">{result.n_matches}</dd>
        </div>
        <div>
          <dt className="text-gray-500">Diferencia vs. llamada típica</dt>
          <dd className="text-lg font-semibold">{formatSignedYards(result.yards_delta_vs_typical)}</dd>
        </div>
      </dl>
      <div className="mt-3 text-sm text-gray-700">
        {result.typical_call ? (
          <p>
            Llamada típica en situaciones parecidas: <strong>{callText(result.typical_call)}</strong>
            {typicalIdx >= 0 && ` (posición ${typicalIdx + 1} del ranking)`}.
          </p>
        ) : (
          <p>No hay jugadas parecidas, así que no hay llamada típica para comparar.</p>
        )}
        {result.low_confidence && <p className="mt-1 text-amber-800">Hay menos de 30 jugadas parecidas: toma el resultado con cautela.</p>}
        <p className="mt-1 text-gray-500">
          El ranking ordena solo por probabilidad de presión (a igualdad, menos yardas). Con un modelo lineal, la presión crece con
          los rushers, por eso suele ganar la misma llamada. Revisa el ranking completo y las yardas esperadas.
        </p>
      </div>
    </Card>
  )
}

function Ranking({ result, visible, setVisible }: { result: Recommendation; visible: number; setVisible: (n: number) => void }) {
  const total = result.ranked.length
  const shown = result.ranked.slice(0, visible)
  return (
    <Card title={`Ranking de llamadas (${total})`}>
      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm">
          <thead className="text-xs text-gray-500">
            <tr>
              <th className="py-1 pr-3">#</th>
              <th className="pr-3">Cobertura</th>
              <th className="pr-3">Rushers</th>
              <th className="pr-3">En la caja</th>
              <th className="pr-3">Presión</th>
              <th>Yardas esp.</th>
            </tr>
          </thead>
          <tbody>
            {shown.map((r, i) => {
              const isTypical = result.typical_call !== null && sameCall(r.call, result.typical_call)
              return (
                <tr key={`${r.call.coverage}-${r.call.pass_rushers}-${r.call.defenders_in_box}`} className={`border-t border-gray-100 ${i === 0 ? 'bg-green-50 font-medium' : ''}`}>
                  <td className="py-1 pr-3">{i + 1}</td>
                  <td className="pr-3">
                    {r.call.coverage} {isTypical && <Badge tone="blue">típica</Badge>}
                  </td>
                  <td className="pr-3">{r.call.pass_rushers}</td>
                  <td className="pr-3">{r.call.defenders_in_box}</td>
                  <td className="pr-3">{formatPercent(r.pressure_probability)}</td>
                  <td>{formatYards(r.expected_yards)}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      {total > visible && (
        <div className="mt-3 flex gap-2">
          <button type="button" className={btnSecondaryCls} onClick={() => setVisible(visible + PAGE)}>
            Mostrar {Math.min(PAGE, total - visible)} más
          </button>
          <button type="button" className={btnSecondaryCls} onClick={() => setVisible(total)}>
            Mostrar todas
          </button>
        </div>
      )}
      {visible > PAGE && (
        <button type="button" className={`${btnSecondaryCls} mt-3`} onClick={() => setVisible(PAGE)}>
          Mostrar menos
        </button>
      )}
    </Card>
  )
}

type ExplainState = { kind: 'idle' } | { kind: 'loading' } | { kind: 'ok'; text: string } | { kind: 'unavailable' }

function ExplainPanel({ id }: { id: string }) {
  const [state, setState] = useState<ExplainState>({ kind: 'idle' })
  const ask = async () => {
    setState({ kind: 'loading' })
    try {
      const r = await postJson<ExplainResponse>(`recommendations/${encodeURIComponent(id)}/explain`)
      setState({ kind: 'ok', text: r.explanation })
    } catch {
      // 503 es un estado normal (sin credenciales de IA); cualquier fallo deja la recomendación intacta.
      setState({ kind: 'unavailable' })
    }
  }
  return (
    <Card title="Explicación (opcional, con IA)">
      {state.kind === 'idle' && (
        <button type="button" className={btnSecondaryCls} onClick={() => void ask()}>
          Pedir explicación
        </button>
      )}
      {state.kind === 'loading' && <Spinner label="Generando explicación…" />}
      {state.kind === 'ok' && <p className="whitespace-pre-line text-sm">{state.text}</p>}
      {state.kind === 'unavailable' && (
        <p className="text-sm text-gray-600">
          La explicación no está disponible por ahora. La recomendación y el ranking siguen siendo válidos.{' '}
          <button type="button" className="underline" onClick={() => void ask()}>
            Reintentar
          </button>
        </p>
      )}
    </Card>
  )
}

function MetricsPanel() {
  const m = useGet<MetricsResponse>('metrics')
  return (
    <Card title="Calidad de los modelos">
      {m.loading && !m.data && <Spinner />}
      {m.error !== null && !m.data && <ErrorBox error={m.error} onRetry={m.retry} />}
      {m.data && (
        <div className="space-y-2 text-sm">
          <p>
            <strong>Presión:</strong> R² {formatMetric(m.data.pressure_model.r2)}, error absoluto medio {formatMetric(m.data.pressure_model.mae)}{' '}
            ({m.data.pressure_model.n_train} jugadas de entrenamiento, {m.data.pressure_model.n_heldout} de prueba).
          </p>
          <p>
            <strong>Yardas:</strong> R² {formatMetric(m.data.yards_model.r2)}, error absoluto medio {formatMetric(m.data.yards_model.mae_yards, 2)} yardas
            ({m.data.yards_model.n_train} de entrenamiento, {m.data.yards_model.n_heldout} de prueba).
          </p>
          <p className="text-gray-600">
            Son modelos lineales simples con poco poder explicativo (R² cercano a 0). Úsalos como referencia, no como predicción precisa.
          </p>
        </div>
      )}
    </Card>
  )
}
