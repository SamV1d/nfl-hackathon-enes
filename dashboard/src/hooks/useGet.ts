import { useCallback, useEffect, useState } from 'react'
import { getJson, isAbort } from '../api'

export interface GetState<T> {
  /** Último dato cargado con éxito (se conserva si una carga posterior falla). */
  data: T | null
  /** Ruta con la que se cargó `data`. */
  dataPath: string | null
  error: unknown | null
  loading: boolean
  retry: () => void
}

/** GET con estado propio de carga, error y reintento. `path` null = no pedir nada. */
export function useGet<T>(path: string | null): GetState<T> {
  const [state, setState] = useState<{ data: T | null; dataPath: string | null; error: unknown | null; loading: boolean }>({
    data: null,
    dataPath: null,
    error: null,
    loading: path !== null,
  })
  const [nonce, setNonce] = useState(0)

  useEffect(() => {
    if (path === null) {
      setState((s) => ({ ...s, loading: false, error: null }))
      return
    }
    const ctl = new AbortController()
    setState((s) => ({ ...s, loading: true, error: null }))
    getJson<T>(path, ctl.signal).then(
      (data) => {
        if (!ctl.signal.aborted) setState({ data, dataPath: path, error: null, loading: false })
      },
      (error) => {
        if (ctl.signal.aborted || isAbort(error)) return
        setState((s) => ({ ...s, loading: false, error }))
      },
    )
    return () => ctl.abort()
  }, [path, nonce])

  const retry = useCallback(() => setNonce((n) => n + 1), [])
  return { ...state, retry }
}
