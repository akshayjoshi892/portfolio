import { motion } from 'framer-motion'

/**
 * Generated cover art used when a project has no screenshot yet.
 * Three motifs cycle by index: orbit, bars, dot field.
 */
export function ProjectArt({ index, title }: { index: number; title: string }) {
  const variant = index % 3
  return (
    <div className="absolute inset-0 overflow-hidden bg-[var(--bg-inset)]">
      <div className="bg-grid absolute inset-0 opacity-70" />
      <div
        className="absolute -right-1/4 -top-1/3 h-[140%] w-[80%] rounded-full opacity-25 blur-3xl"
        style={{ background: 'radial-gradient(circle, var(--accent), transparent 65%)' }}
      />

      <svg viewBox="0 0 400 225" className="absolute inset-0 h-full w-full" preserveAspectRatio="xMidYMid slice" aria-hidden>
        {variant === 0 && (
          <g transform="translate(270 112)">
            {[30, 58, 86].map((r, i) => (
              <motion.g
                key={r}
                animate={{ rotate: i % 2 ? -360 : 360 }}
                transition={{ duration: 18 + i * 8, repeat: Infinity, ease: 'linear' }}
              >
                <circle r={r} fill="none" stroke="var(--fg)" strokeOpacity={0.18} strokeDasharray={i === 1 ? '3 6' : undefined} />
                <circle cx={r} r={i === 0 ? 4 : 3} fill={i === 0 ? 'var(--accent)' : 'var(--fg)'} />
              </motion.g>
            ))}
            <circle r="10" fill="var(--accent)" />
          </g>
        )}

        {variant === 1 && (
          <g transform="translate(190 180)">
            {[46, 80, 62, 110, 90, 130, 104].map((h, i) => (
              <motion.rect
                key={i}
                x={i * 26}
                width="16"
                rx="3"
                height={h}
                y={-h}
                style={{ transformBox: 'fill-box', transformOrigin: 'bottom' }}
                fill={i === 5 ? 'var(--accent)' : 'var(--fg)'}
                fillOpacity={i === 5 ? 1 : 0.16}
                animate={{ scaleY: [0.6, 1, 0.75, 0.6] }}
                transition={{ duration: 3.2, repeat: Infinity, delay: i * 0.18, ease: 'easeInOut' }}
              />
            ))}
          </g>
        )}

        {variant === 2 && (
          <g transform="translate(190 36)">
            {Array.from({ length: 8 * 6 }).map((_, i) => {
              const cx = (i % 8) * 24
              const cy = Math.floor(i / 8) * 28
              const hot = i === 19 || i === 28 || i === 21
              return (
                <motion.circle
                  key={i}
                  cx={cx}
                  cy={cy}
                  r={hot ? 5 : 2.5}
                  fill={hot ? 'var(--accent)' : 'var(--fg)'}
                  fillOpacity={hot ? 1 : 0.25}
                  animate={hot ? { scale: [1, 1.6, 1] } : { opacity: [0.3, 1, 0.3] }}
                  transition={{ duration: 2.4, repeat: Infinity, delay: (i % 8) * 0.12 + Math.floor(i / 8) * 0.08 }}
                  style={{ transformOrigin: `${cx}px ${cy}px` }}
                />
              )
            })}
          </g>
        )}
      </svg>

      <span className="absolute bottom-4 left-5 font-mono text-[10px] uppercase tracking-[0.2em] text-[var(--fg-muted)]">
        {title}
      </span>
    </div>
  )
}
