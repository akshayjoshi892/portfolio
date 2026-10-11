import { motion } from 'framer-motion'
import { MapPin } from 'lucide-react'
import { EXPERIENCE } from '@/data/sample'
import { EASE_OUT } from '@/lib/motion'
import { toneAt, toneFor } from '@/lib/tones'

function formatDate(value: string | null) {
  if (!value) return 'Present'
  const [year, month] = value.split('-')
  return new Date(Number(year), Number(month) - 1).toLocaleDateString('en-US', { month: 'short', year: 'numeric' })
}

function duration(start: string, end: string | null) {
  const [sy, sm] = start.split('-').map(Number)
  const e = end ? end.split('-').map(Number) : [new Date().getFullYear(), new Date().getMonth() + 1]
  const months = (e[0] - sy) * 12 + (e[1] - sm) + 1
  const y = Math.floor(months / 12)
  const m = months % 12
  return [y && `${y} yr${y > 1 ? 's' : ''}`, m && `${m} mo`].filter(Boolean).join(' ')
}

export function Experience() {
  return (
    <section id="experience" className="px-4 py-20 sm:px-6 md:py-28">
      <div className="mx-auto max-w-7xl">
        <div className="mb-10">
          <p className="eyebrow mb-2">// experience</p>
          <h2 className="text-3xl font-semibold tracking-tight sm:text-4xl">Where I've worked</h2>
        </div>

        <ol className="relative space-y-6 border-l border-[var(--border)] pl-6 sm:pl-8">
          {EXPERIENCE.map((entry, i) => {
            const tone = toneAt(i + 2)
            const current = !entry.endDate
            return (
              <motion.li
                key={entry.id}
                initial={{ opacity: 0, y: 16 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, margin: '-40px' }}
                transition={{ duration: 0.5, delay: i * 0.06, ease: EASE_OUT }}
                className="relative"
              >
                <span
                  aria-hidden
                  className="absolute -left-[31px] top-6 h-3 w-3 rounded-full ring-4 ring-[var(--bg)] sm:-left-[39px]"
                  style={{ background: tone }}
                />
                <article className="rounded-xl border border-[var(--border)] bg-[var(--bg-elevated)] p-5 sm:p-6">
                  <header className="flex flex-wrap items-start justify-between gap-x-6 gap-y-2">
                    <div>
                      <h3 className="text-lg font-semibold tracking-tight sm:text-xl">
                        {entry.role} <span className="text-[var(--fg-subtle)]">@</span>{' '}
                        <span style={{ color: tone }}>{entry.company}</span>
                      </h3>
                      <p className="mt-1 flex items-center gap-1.5 text-sm text-[var(--fg-muted)]">
                        <MapPin size={13} /> {entry.location}
                      </p>
                    </div>
                    <div className="text-right font-mono text-xs">
                      <p className="text-[var(--fg)]">
                        {formatDate(entry.startDate)} — {formatDate(entry.endDate)}
                      </p>
                      <p className="mt-1 text-[var(--fg-subtle)]">
                        {current && <span className="chip mr-1.5" style={{ ['--c' as string]: 'var(--tone-sage)' }}>current</span>}
                        {duration(entry.startDate, entry.endDate)}
                      </p>
                    </div>
                  </header>

                  <ul className="mt-5 grid gap-2.5 lg:grid-cols-2 lg:gap-x-8">
                    {entry.achievements.map((a, idx) => (
                      <li key={idx} className="flex gap-3 text-[15px] leading-relaxed text-[var(--fg-muted)]">
                        <span className="mt-[0.7em] h-1.5 w-1.5 shrink-0 rounded-full" style={{ background: tone, opacity: 0.8 }} />
                        {a}
                      </li>
                    ))}
                  </ul>

                  <div className="mt-5 flex flex-wrap gap-1.5 border-t border-[var(--border)] pt-4">
                    {entry.techStack.map((tech) => (
                      <span key={tech} className="chip" style={{ ['--c' as string]: toneFor(tech) }}>
                        {tech}
                      </span>
                    ))}
                  </div>
                </article>
              </motion.li>
            )
          })}
        </ol>
      </div>
    </section>
  )
}
