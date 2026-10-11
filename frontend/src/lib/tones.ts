export const TONES = ['sage', 'amber', 'teal', 'clay', 'plum'] as const

/** Stable muted colour for any label (tag, company, project). */
export function toneFor(label: string) {
  let h = 0
  for (const ch of label.toLowerCase()) h = (h * 31 + ch.charCodeAt(0)) >>> 0
  return `var(--tone-${TONES[h % TONES.length]})`
}

export const toneAt = (i: number) => `var(--tone-${TONES[i % TONES.length]})`
