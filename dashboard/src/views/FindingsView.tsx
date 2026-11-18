import { Bar, BarChart, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { useGet } from '../hooks/useGet'
import { formatPercent, formatYards } from '../lib/format'
import type { Findings, Group } from '../types'
import { Card, ErrorBox, Spinner, Stat } from '../components/ui'

/** Respuesta vacía: sin resumen utilizable y sin ningún grupo ni ranking. */
function isEmpty(f: Findings | null): boolean {
  if (!f || !f.summary) return true
  const noSummary = Object.values(f.summary).every((v) => v === null || v === undefined)
  const noGroups = [f.by_coverage, f.by_formation, f.by_defenders_in_box, f.top_sackers, f.top_teams].every((a) => !a || a.length === 0)
  return noSummary && noGroups
}

export default function FindingsView() {
  const q = useGet<Findings>('findings')
  const f = q.data

  return (
    <div className="space-y-4">
      {q.loading && !f && <Spinner />}
      {q.error !== null && !f && <ErrorBox message="Los hallazgos no están disponibles." onRetry={q.retry} />}
      {q.error === null && !q.loading && isEmpty(f) && <ErrorBox message="Los hallazgos no están disponibles: el servidor devolvió una respuesta vacía." onRetry={q.retry} />}
      {f && !isEmpty(f) && (
        <>
          {q.error !== null && <ErrorBox message="No se pudieron actualizar los hallazgos." onRetry={q.retry} />}
          <Card title="Resumen">
            <div className="grid grid-cols-2 gap-3 lg:grid-cols-5">
              <Stat label="Tasa de presión" value={formatPercent(f.summary.pressure_rate)} />
              <Stat label="Yardas por jugada con presión" value={formatYards(f.summary.avg_yards_with_pressure)} />
              <Stat label="Yardas por jugada sin presión" value={formatYards(f.summary.avg_yards_without_pressure)} />
              <Stat label="Tasa de sack con presión" value={formatPercent(f.summary.sack_rate_with_pressure)} />
              <Stat label="Tasa de sack sin presión" value={formatPercent(f.summary.sack_rate_without_pressure)} />
            </div>
          </Card>
          <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
            <GroupChart title="Por cobertura" groups={f.by_coverage} />
            <GroupChart title="Por formación ofensiva" groups={f.by_formation} />
            <GroupChart title="Por defensores en la caja" groups={sortNumeric(f.by_defenders_in_box)} />
          </div>
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            <Card title="Top 10 jugadores por sacks">
              <Ranking
                rows={f.top_sackers.map((p) => ({ key: p.player_id, label: `${p.name} (${p.position})`, value: String(p.count) }))}
              />
            </Card>
            <Card title="Top 10 equipos por tasa de presión (defensa)">
              <Ranking
                rows={f.top_teams.map((t) => ({ key: t.team, label: `${t.team} · ${t.pass_plays} jugadas`, value: formatPercent(t.pressure_rate) }))}
              />
            </Card>
          </div>
        </>
      )}
    </div>
  )
}

function sortNumeric(groups: Group[]): Group[] {
  if (!groups.every((g) => g.group.trim() !== '' && Number.isFinite(Number(g.group)))) return groups
  return [...groups].sort((a, b) => Number(a.group) - Number(b.group))
}

function GroupChart({ title, groups }: { title: string; groups: Group[] }) {
  const data = groups.map((g) => ({
    name: g.group,
    presion: g.pressure_rate === null ? null : Number((g.pressure_rate * 100).toFixed(1)),
    yardas: g.avg_yards === null ? null : Number(g.avg_yards.toFixed(1)),
    count: g.count,
  }))
  return (
    <Card title={title}>
      {data.length === 0 ? (
        <p className="text-sm text-gray-600">Sin datos.</p>
      ) : (
        <div className="h-72">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 8 }}>
              <XAxis dataKey="name" interval={0} angle={data.length > 6 ? -35 : 0} textAnchor={data.length > 6 ? 'end' : 'middle'} height={data.length > 6 ? 70 : 30} tick={{ fontSize: 11 }} />
              <YAxis yAxisId="p" width={44} unit="%" />
              <YAxis yAxisId="y" width={36} orientation="right" />
              <Tooltip />
              <Legend verticalAlign="top" />
              <Bar yAxisId="p" dataKey="presion" name="Presión (%)" fill="#2F7D4F" isAnimationActive={false} />
              <Bar yAxisId="y" dataKey="yardas" name="Yardas por jugada" fill="#2563EB" isAnimationActive={false} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </Card>
  )
}

function Ranking({ rows }: { rows: { key: string; label: string; value: string }[] }) {
  if (rows.length === 0) return <p className="text-sm text-gray-600">Sin datos.</p>
  return (
    <ol className="list-decimal space-y-1 pl-5 text-sm">
      {rows.map((r) => (
        <li key={r.key}>
          {r.label} · <strong>{r.value}</strong>
        </li>
      ))}
    </ol>
  )
}
