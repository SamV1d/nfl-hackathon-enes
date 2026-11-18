import { useState } from 'react'
import { ViewBoundary } from './components/ui'
import FindingsView from './views/FindingsView'
import OpponentReportView from './views/OpponentReportView'
import ReplayView from './views/ReplayView'
import WhatIfView from './views/WhatIfView'

const TABS = [
  { id: 'whatif', label: 'What-If', Component: WhatIfView },
  { id: 'replay', label: 'Game Replay', Component: ReplayView },
  { id: 'opponent', label: 'Reporte del rival', Component: OpponentReportView },
  { id: 'findings', label: 'Hallazgos', Component: FindingsView },
] as const

type TabId = (typeof TABS)[number]['id']

export default function App() {
  const [active, setActive] = useState<TabId>('whatif')
  // Cada vista se monta la primera vez que se visita y se conserva oculta, para no perder su estado.
  const [visited, setVisited] = useState<Set<TabId>>(new Set(['whatif']))

  const open = (id: TabId) => {
    setActive(id)
    setVisited((v) => (v.has(id) ? v : new Set(v).add(id)))
  }

  return (
    <div className="min-h-screen bg-gray-50 text-gray-900">
      <header className="border-b border-gray-200 bg-white">
        <div className="mx-auto max-w-6xl px-4 py-3">
          <h1 className="text-xl font-bold">Coach Pass Rush Assistant</h1>
          <p className="text-xs text-gray-500">NFL 2021, semanas 1 a 8. Recomienda la llamada defensiva con mayor presión prevista.</p>
          <nav className="mt-3 flex flex-wrap gap-1" role="tablist">
            {TABS.map((t) => (
              <button
                key={t.id}
                type="button"
                role="tab"
                aria-selected={active === t.id}
                onClick={() => open(t.id)}
                className={`rounded-t px-4 py-2 text-sm font-medium ${
                  active === t.id ? 'bg-green-700 text-white' : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                {t.label}
              </button>
            ))}
          </nav>
        </div>
      </header>
      <main className="mx-auto max-w-6xl px-4 py-4">
        {TABS.map(({ id, Component }) =>
          visited.has(id) ? (
            <div key={id} role="tabpanel" hidden={active !== id}>
              <ViewBoundary>
                <Component />
              </ViewBoundary>
            </div>
          ) : null,
        )}
      </main>
    </div>
  )
}
