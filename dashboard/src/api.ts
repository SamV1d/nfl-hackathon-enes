// Cliente de la API. Rutas relativas SIN diagonal inicial: `api/<ruta>`.
// Timeout de 10 s con AbortController.

export const TIMEOUT_MS = 10_000

export type ApiErrorKind = 'network' | 'timeout' | 'http'

export class ApiError extends Error {
  kind: ApiErrorKind
  status: number
  detail: string
  fields: string[]
  missingFiles: string[]
  constructor(kind: ApiErrorKind, status: number, detail: string, fields: string[] = [], missingFiles: string[] = []) {
    super(detail)
    this.name = 'ApiError'
    this.kind = kind
    this.status = status
    this.detail = detail
    this.fields = fields
    this.missingFiles = missingFiles
  }
}

const isAbort = (e: unknown) => e instanceof DOMException && e.name === 'AbortError'

/** `signal` opcional: permite cancelar desde fuera (cambio de vista o de entrada). */
async function request<T>(path: string, init: RequestInit, signal?: AbortSignal): Promise<T> {
  const ctl = new AbortController()
  let timedOut = false
  const timer = setTimeout(() => {
    timedOut = true
    ctl.abort()
  }, TIMEOUT_MS)
  const onAbort = () => ctl.abort()
  if (signal) {
    if (signal.aborted) ctl.abort()
    else signal.addEventListener('abort', onAbort)
  }
  const fail = (e: unknown): never => {
    if (timedOut) throw new ApiError('timeout', 0, 'timeout')
    if (signal?.aborted || isAbort(e)) throw new DOMException('Aborted', 'AbortError')
    throw new ApiError('network', 0, 'network error')
  }
  try {
    let res: Response
    try {
      res = await fetch(`api/${path}`, { ...init, signal: ctl.signal })
    } catch (e) {
      return fail(e)
    }
    if (!res.ok) {
      let body: { detail?: unknown; fields?: unknown; missing_files?: unknown } = {}
      try {
        body = await res.json()
      } catch {
        // cuerpo sin JSON: se usa el estado HTTP
      }
      const strs = (v: unknown) => (Array.isArray(v) ? v.map(String) : [])
      throw new ApiError(
        'http',
        res.status,
        typeof body.detail === 'string' ? body.detail : `HTTP ${res.status}`,
        strs(body.fields),
        strs(body.missing_files),
      )
    }
    try {
      return (await res.json()) as T
    } catch (e) {
      if (timedOut || signal?.aborted) return fail(e)
      throw new ApiError('http', res.status, 'invalid response')
    }
  } finally {
    clearTimeout(timer)
    signal?.removeEventListener('abort', onAbort)
  }
}

export const getJson = <T>(path: string, signal?: AbortSignal): Promise<T> =>
  request<T>(path, { method: 'GET' }, signal)

export const postJson = <T>(path: string, body?: unknown, signal?: AbortSignal): Promise<T> =>
  request<T>(
    path,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: body === undefined ? undefined : JSON.stringify(body),
    },
    signal,
  )

export { isAbort }

const KNOWN_DETAILS: Record<string, string> = {
  'invalid input': 'Entrada no válida',
  'model unavailable': 'El modelo no está disponible',
  'prediction unavailable': 'No se pudo calcular la predicción',
  'explanation unavailable': 'Explicación no disponible',
  'not found': 'No encontrado',
}

export const FIELD_LABELS: Record<string, string> = {
  down: 'Down',
  yards_to_go: 'Yardas por avanzar',
  offense_formation: 'Formación ofensiva',
  expected_coverage: 'Cobertura esperada',
}

/** Texto en español para mostrar al coach. */
export function describeError(e: unknown): string {
  if (!(e instanceof ApiError)) return 'Ocurrió un error inesperado.'
  if (e.kind === 'timeout') return 'El servidor no respondió en 10 segundos.'
  if (e.kind === 'network') return 'No se pudo conectar con el servidor.'
  if (e.missingFiles.length > 0) {
    return `El servidor no pudo cargar los datos. Archivos faltantes: ${e.missingFiles.join(', ')}.`
  }
  const base = KNOWN_DETAILS[e.detail] ?? e.detail
  if (e.fields.length > 0) {
    return `${base}: ${e.fields.map((f) => FIELD_LABELS[f] ?? f).join(', ')}.`
  }
  return base
}
