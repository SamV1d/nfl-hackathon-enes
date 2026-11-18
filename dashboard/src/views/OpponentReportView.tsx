import { useState } from 'react'
import { Bar, BarChart, Cell, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { useGet } from '../hooks/useGet'
import { formatPercent, formatYards, NA } from '../lib/format'
import type { Group, LeagueAverage, Player, Side, TeamReport, TeamsResponse } from '../types'
import { btnSecondaryCls, Card, ErrorBox, selectCls, Spinner } from '../components/ui'

type Which = 'defense' | 'offense'

export default function OpponentReportView() {
  const teams = useGet<TeamsResponse>('teams')
  const [team, setTeam] = useState('')
  const [which, setWhich] = useState<Which>('defense')
  const report = useGet<TeamReport>(team ? `teams/${encodeURIComponent(team)}/report` : null)

  // Si la carga falla se conserva el equipo mostrado anteriormente.
  const shown = report.data?.team ?? ''
  const selectValue = report.error !== null && shown ? shown : team

  return (
    <div className="space-y-4">
      <Card title="Equipo rival">
        {teams.loading && !teams.data && <Spinner />}
        {teams.error !== null && !teams.data && <ErrorBox error={teams.error} onRetry={teams.retry} />}
        {teams.data && (
          <div className="flex flex-wrap items-center gap-3">
            <select aria-label="Equipo" className={`${selectCls} max-w-xs`} value={selectValue} onChange={(e) => setTeam(e.target.value)}>
              <option value="">Elige un equipo…</option>
              {teams.data.teams.map((t) => (
                <option key={t} value={t}>
                  {t}
                </option>
              ))}
            </select>
            {report.loading && <Spinner label="Cargando reporte…" />}
          </div>
        )}
      </Card>

      {report.error !== null && (
        <ErrorBox
          message={
            report.data
              ? `No se pudo cargar el reporte de ${team}. Se conserva el de ${report.data.team}.`
              : `No se pudo cargar el reporte de ${team}.`
          }
          onRetry={report.retry}
        />
      )}

      {report.data && !report.data.has_data && (
        <Card title={report.data.team}>
          <p className="text-sm text-gray-700">{report.data.message || 'Sin datos disponibles para este equipo.'}</p>
        </Card>
      )}

      {report.data && report.data.has_data && (
        <ReportBody report={report.data} which={which} setWhich={setWhich} />
      )}
    </div>
  )
}

function ReportBody({ report, which, setWhich }: { report: TeamReport; which: Which; setWhich: (w: Which) => void }) {
  const side: Side | null = which === 'defense' ? report.defense : report.offense
  const league = report.league_average
  return (
    <div className="space-y-4">
      <div className="flex gap-2">
        {(['defense', 'offense'] as const).map((w) => (
          <button
            key={w}
            type="button"
            aria-pressed={which === w}
            onClick={() => setWhich(w)}
            className={`${btnSecondaryCls} ${which === w ? 'border-green-700 bg-green-50 font-semibold' : ''}`}
          >
            {w === 'defense' ? 'Defensa' : 'Ofensiva'}
          </button>
        ))}
      </div>

      {side && league ? (
        <Card title={`${report.team} contra el promedio de la liga (${which === 'defense' ? 'defensa' : 'ofensiva'}, ${side.pass_plays} jugadas de pase)`}>
          <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
            <CompareChart title="Tasa de presión" team={report.team} a={side.pressure_rate} b={league.pressure_rate} unit="%" />
            <CompareChart title="Tasa de sack" team={report.team} a={side.sack_rate} b={league.sack_rate} unit="%" />
            <CompareChart title="Yardas por jugada" team={report.team} a={side.avg_yards} b={league.avg_yards} unit="yardas" />
          </div>
        </Card>
      ) : (
        <Card>
          <p className="text-sm text-gray-700">No hay datos de {which === 'defense' ? 'defensa' : 'ofensiva'} para este equipo.</p>
        </Card>
      )}

      {side && (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          <GroupTable title="Por cobertura" groups={side.by_coverage} />
          <GroupTable title="Por formación" groups={side.by_formation} />
        </div>
      )}

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <PlayerList title="Más sacks (defensa)" players={report.top_sackers} />
        <PlayerList title="Más presiones (defensa)" players={report.top_pressure} />
      </div>
    </div>
  )
}

const fmt = (v: number | null, unit: '%' | 'yardas') => (unit === '%' ? formatPercent(v) : formatYards(v))

function CompareChart({ title, team, a, b, unit }: { title: string; team: string; a: number | null; b: number | null; unit: '%' | 'yardas' }) {
  const scale = unit === '%' ? 100 : 1
  const data = [
    { name: team, value: a === null ? null : a * scale, raw: a, fill: '#2F7D4F' },
    { name: 'Liga', value: b === null ? null : b * scale, raw: b, fill: '#9CA3AF' },
  ]
  return (
    <div>
      <h4 className="mb-1 text-sm font-medium">{title}</h4>
      <div className="h-48">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 20, right: 8, left: 0, bottom: 0 }}>
            <XAxis dataKey="name" />
            <YAxis width={40} />
            <Tooltip formatter={(_v, _n, item) => fmt((item.payload as { raw: number | null }).raw, unit)} />
            <Bar dataKey="value" isAnimationActive={false}>
              {data.map((d) => (
                <Cell key={d.name} fill={d.fill} />
              ))}
              <LabelList dataKey="raw" position="top" formatter={(v: unknown) => fmt(typeof v === 'number' ? v : null, unit)} />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

function GroupTable({ title, groups }: { title: string; groups: Group[] }) {
  return (
    <Card title={title}>
      {groups.length === 0 ? (
        <p className="text-sm text-gray-600">Sin datos.</p>
      ) : (
        <div className="max-h-64 overflow-auto">
          <table className="w-full text-left text-sm">
            <thead className="text-xs text-gray-500">
              <tr>
                <th className="py-1 pr-3">Grupo</th>
                <th className="pr-3">Jugadas</th>
                <th className="pr-3">Presión</th>
                <th>Yardas</th>
              </tr>
            </thead>
            <tbody>
              {groups.map((g) => (
                <tr key={g.group} className="border-t border-gray-100">
                  <td className="py-1 pr-3">{g.group}</td>
                  <td className="pr-3">{g.count}</td>
                  <td className="pr-3">{formatPercent(g.pressure_rate)}</td>
                  <td>{formatYards(g.avg_yards)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  )
}

function PlayerList({ title, players }: { title: string; players: Player[] }) {
  return (
    <Card title={title}>
      {players.length === 0 ? (
        <p className="text-sm text-gray-600">{NA}</p>
      ) : (
        <ol className="list-decimal space-y-1 pl-5 text-sm">
          {players.map((p) => (
            <li key={p.player_id}>
              {p.name} <span className="text-gray-500">({p.position})</span> · <strong>{p.count}</strong>
            </li>
          ))}
        </ol>
      )}
    </Card>
  )
}

export type { LeagueAverage }
