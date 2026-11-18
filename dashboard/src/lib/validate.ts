// Validación de la entrada de What-If. Función pura: no toca la red ni el DOM.

export interface WhatIfInput {
  down: string
  yardsToGo: string
  formation: string
  coverage: string
}

export type WhatIfField = keyof WhatIfInput

const INT = /^-?\d+$/

function intInRange(raw: string, min: number, max: number): boolean {
  const s = raw.trim()
  if (!INT.test(s)) return false
  const n = Number(s)
  return Number.isSafeInteger(n) && n >= min && n <= max
}

/** Campos inválidos, en el orden del formulario. */
export function invalidFields(i: WhatIfInput): WhatIfField[] {
  const bad: WhatIfField[] = []
  if (!intInRange(i.down, 0, 4)) bad.push('down')
  if (!intInRange(i.yardsToGo, 0, 99)) bad.push('yardsToGo')
  if (i.formation.trim() === '') bad.push('formation')
  if (i.coverage.trim() === '') bad.push('coverage')
  return bad
}

const MESSAGES: Record<WhatIfField, string> = {
  down: 'Down: debe ser un entero de 0 a 4.',
  yardsToGo: 'Yardas por avanzar: debe ser un entero de 0 a 99.',
  formation: 'Formación ofensiva: elige una opción.',
  coverage: 'Cobertura esperada: elige una opción.',
}

/** Un mensaje por cada campo inválido; lista vacía si todo es válido. */
export function validateWhatIf(i: WhatIfInput): string[] {
  return invalidFields(i).map((f) => MESSAGES[f])
}
