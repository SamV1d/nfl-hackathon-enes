import { useReducer, useState } from 'react'
import { useGet } from '../hooks/useGet'
import { formatSignedYards, formatYards, NA, downLabel, quarterLabel } from '../lib/format'
import { canNext, canPrev, cumulative, navReduce } from '../lib/replay'
import type { Game, PlaysResponse, ReplayPlay } from '../types'
import { Badge, btnCls, btnSecondaryCls, callText, Card, ErrorBox, scoredText, selectCls, Spinner } from '../components/ui'

const EMPTY_MESSAGE = 'Este partido no tiene jugadas de pase disponibles.'

export default function ReplayView() {
  const games = useGet<Game[]>('games')
  const [gameId, setGameId] = useState<string>('')
  const plays = useGet<PlaysResponse>(gameId ? `games/${encodeURIComponent(gameId)}/plays` : null)

  // El partido mostrado es el último cargado con éxito; si una carga falla se conserva.
  const shownId = plays.data ? String(plays.data.game_id) : ''
  const selectValue = plays.error !== null && shownId ? shownId : gameId

  return (
    <div className="space-y-4">
      <Card title="Partido">
        {games.loading && !games.data && <Spinner />}
        {games.error !== null && !games.data && (
          <ErrorBox message="No se pudo cargar la lista de partidos." onRetry={games.retry} />
        )}
        {games.data && (
          <div className="flex flex-wrap items-center gap-3">
            <select
              aria-label="Partido"
              className={`${selectCls} max-w-md`}
              value={selectValue}
              onChange={(e) => setGameId(e.target.value)}
            >
              <option value="">Elige un partido…</option>
              {games.data.map((g) => (
                <option key={g.game_id} value={String(g.game_id)}>
                  {g.date} · {g.visitor_team} @ {g.home_team}
                </option>
              ))}
            </select>
            {plays.loading && <Spinner label="Cargando jugadas…" />}
          </div>
        )}
        {games.error !== null && games.data && (
          <div className="mt-3">
            <ErrorBox message="No se pudo actualizar la lista de partidos." onRetry={games.retry} />
          </div>
        )}
      </Card>

      {plays.error !== null && (
        <ErrorBox
          message={
            plays.data
              ? 'No se pudieron cargar los datos del replay. Se conserva el partido y la jugada anteriores.'
              : 'No se pudieron cargar los datos del replay.'
          }
          onRetry={plays.retry}
        />
      )}

      {plays.data && <ReplayPlayer key={plays.data.game_id} data={plays.data} game={games.data?.find((g) => g.game_id === plays.data!.game_id)} />}
    </div>
  )
}

function ReplayPlayer({ data, game }: { data: PlaysResponse; game: Game | undefined }) {
  const [nav, dispatch] = useReducer(navReduce, { index: 0, total: data.total })
  const [jump, setJump] = useState('')
  const [jumpError, setJumpError] = useState<string | null>(null)

  if (data.total === 0 || data.plays.length === 0) {
    return (
      <Card title="Jugadas">
        <p className="text-sm text-gray-700">{data.message || EMPTY_MESSAGE}</p>
        <Controls nav={{ index: 0, total: 0 }} dispatch={dispatch} jump={jump} setJump={setJump} jumpError={null} onJump={() => undefined} />
      </Card>
    )
  }

  const play = data.plays[nav.index]
  const cum = cumulative(data.plays, nav.index + 1)

  const onJump = () => {
    const n = Number(jump.trim())
    if (jump.trim() === '' || !Number.isInteger(n) || n < 1 || n > data.total) {
      setJumpError(`Escribe una posición entera entre 1 y ${data.total}.`)
      return
    }
    setJumpError(null)
    dispatch({ type: 'jump', position: n })
  }

  return (
    <div className="space-y-4">
      <Card title={game ? `${game.visitor_team} @ ${game.home_team} · ${game.date}` : `Partido ${data.game_id}`}>
        <Controls nav={nav} dispatch={dispatch} jump={jump} setJump={setJump} jumpError={jumpError} onJump={onJump} />
      </Card>
      <PlayDetail play={play} total={data.total} />
      <Card title="Comparación acumulada (jugadas 1 a la actual)">
        <dl className="grid grid-cols-3 gap-3 text-sm">
          <div>
            <dt className="text-gray-500">Yardas reales</dt>
            <dd className="text-lg font-semibold">{formatYards(cum.actual)}</dd>
          </div>
          <div>
            <dt className="text-gray-500">Yardas predichas (Best_Call)</dt>
            <dd className="text-lg font-semibold">{formatYards(cum.predicted)}</dd>
          </div>
          <div>
            <dt className="text-gray-500">Diferencia (real − predicha)</dt>
            <dd className="text-lg font-semibold">{formatSignedYards(cum.diff)}</dd>
          </div>
        </dl>
        {cum.omittedPredicted > 0 && (
          <p className="mt-2 text-sm text-amber-800">
            {cum.omittedPredicted} jugada(s) sin recomendación quedan fuera del lado predicho, así que la comparación no es exacta.
          </p>
        )}
        {cum.omittedActual > 0 && (
          <p className="mt-2 text-sm text-amber-800">{cum.omittedActual} jugada(s) sin yardas reales quedan fuera del lado real.</p>
        )}
      </Card>
    </div>
  )
}

function Controls(props: {
  nav: { index: number; total: number }
  dispatch: (a: { type: 'next' } | { type: 'prev' } | { type: 'jump'; position: number }) => void
  jump: string
  setJump: (v: string) => void
  jumpError: string | null
  onJump: () => void
}) {
  const { nav, dispatch, jump, setJump, jumpError, onJump } = props
  const empty = nav.total === 0
  return (
    <div>
      <div className="flex flex-wrap items-center gap-2">
        <button type="button" className={btnSecondaryCls} disabled={!canPrev(nav)} onClick={() => dispatch({ type: 'prev' })}>
          ← Atrás
        </button>
        <button type="button" className={btnSecondaryCls} disabled={!canNext(nav)} onClick={() => dispatch({ type: 'next' })}>
          Adelante →
        </button>
        <span className="mx-2 text-sm text-gray-700">{empty ? 'Sin jugadas' : `Jugada ${nav.index + 1} de ${nav.total}`}</span>
        <form
          className="flex items-center gap-2"
          onSubmit={(e) => {
            e.preventDefault()
            onJump()
          }}
        >
          <input
            aria-label="Ir a la posición"
            className="w-20 rounded border border-gray-300 px-2 py-1.5 text-sm disabled:bg-gray-100"
            inputMode="numeric"
            placeholder="Posición"
            disabled={empty}
            value={jump}
            onChange={(e) => setJump(e.target.value)}
          />
          <button type="submit" className={btnCls} disabled={empty}>
            Ir
          </button>
        </form>
      </div>
      {jumpError && (
        <p role="alert" className="mt-2 text-sm text-red-700">
          {jumpError}
        </p>
      )}
    </div>
  )
}

function PlayDetail({ play, total }: { play: ReplayPlay; total: number }) {
  const q = quarterLabel(play.quarter)
  const down = downLabel(play.down)
  return (
    <Card title={`Jugada ${play.position} de ${total}`}>
      <div className="flex flex-wrap items-center gap-2 text-sm">
        <Badge>{play.quarter === 5 ? q : `Cuarto ${q}`}</Badge>
        <Badge>{play.down === 0 ? down : `Down ${down}`}</Badge>
        <Badge>{play.yards_to_go} yardas por avanzar</Badge>
        <Badge>Reloj {play.game_clock}</Badge>
      </div>
      <dl className="mt-3 grid grid-cols-1 gap-3 text-sm sm:grid-cols-2">
        <div>
          <dt className="text-gray-500">Formación / cobertura esperada</dt>
          <dd>
            {play.offense_formation} / {play.expected_coverage}
          </dd>
        </div>
        <div>
          <dt className="text-gray-500">Resultado real</dt>
          <dd>
            Presión: {play.pressure ? 'sí' : 'no'} · Sack: {play.sack ? 'sí' : 'no'} · Yardas:{' '}
            {play.yards_gained === null ? NA : formatYards(play.yards_gained)}
          </dd>
        </div>
        <div>
          <dt className="text-gray-500">Best_Call (recomendada)</dt>
          <dd>{play.best_call ? scoredText(play.best_call) : 'No disponible para esta jugada.'}</dd>
        </div>
        <div>
          <dt className="text-gray-500">Llamada real</dt>
          <dd>{callText(play.actual_call)}</dd>
        </div>
      </dl>
    </Card>
  )
}
