import { useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { ArrowUpRight } from 'lucide-react'
import { GithubIcon } from '@/components/ui/GithubIcon'

const GITHUB_USER = 'akshayjoshi892'
const LEETCODE_USER = 'akshayjoshi_aj'

type DayMap = Record<string, number> // 'YYYY-MM-DD' -> count

const iso = (d: Date) => d.toISOString().slice(0, 10)

async function fetchGithub() {
  const res = await fetch(`https://github-contributions-api.jogruber.de/v4/${GITHUB_USER}?y=last`)
  if (!res.ok) throw new Error('github')
  const json: { total: Record<string, number>; contributions: { date: string; count: number }[] } = await res.json()
  const days: DayMap = {}
  for (const c of json.contributions) days[c.date] = c.count
  return { days, total: json.total.lastYear ?? Object.values(days).reduce((a, b) => a + b, 0) }
}

async function fetchLeetcode() {
  const base = `https://alfa-leetcode-api.onrender.com/${LEETCODE_USER}`
  const [cal, solved] = await Promise.all([
    fetch(`${base}/calendar`).then((r) => (r.ok ? r.json() : Promise.reject(new Error('lc')))),
    fetch(`${base}/solved`).then((r) => (r.ok ? r.json() : null)).catch(() => null),
  ])
  const raw: Record<string, number> =
    typeof cal.submissionCalendar === 'string' ? JSON.parse(cal.submissionCalendar) : cal.submissionCalendar ?? {}
  const days: DayMap = {}
  for (const [ts, n] of Object.entries(raw)) {
    const k = iso(new Date(Number(ts) * 1000))
    days[k] = (days[k] ?? 0) + n
  }
  return {
    days,
    streak: cal.streak as number | undefined,
    solved: solved
      ? {
          total: solved.solvedProblem as number,
          easy: solved.easySolved as number,
          medium: solved.mediumSolved as number,
          hard: solved.hardSolved as number,
        }
      : null,
  }
}

/** Build 53 columns of 7 days ending today, GitHub-style (weeks start Sunday). */
function useGrid(days: DayMap | undefined) {
  return useMemo(() => {
    const today = new Date()
    today.setHours(12, 0, 0, 0)
    const start = new Date(today)
    start.setDate(start.getDate() - 52 * 7 - start.getDay())
    const weeks: { date: string; count: number; future: boolean }[][] = []
    const months: { col: number; label: string }[] = []
    const cur = new Date(start)
    let lastMonth = -1
    for (let w = 0; w < 53; w++) {
      const week = []
      for (let d = 0; d < 7; d++) {
        const k = iso(cur)
        if (d === 0 && cur.getMonth() !== lastMonth) {
          lastMonth = cur.getMonth()
          if (w < 51) months.push({ col: w, label: cur.toLocaleString('en', { month: 'short' }) })
        }
        week.push({ date: k, count: days?.[k] ?? 0, future: cur > today })
        cur.setDate(cur.getDate() + 1)
      }
      weeks.push(week)
    }
    const counts = Object.values(days ?? {}).filter(Boolean).sort((a, b) => a - b)
    const q = (p: number) => counts[Math.floor(p * (counts.length - 1))] ?? 1
    const thresholds = [q(0.25), q(0.5), q(0.75)]
    const level = (n: number) => (n <= 0 ? 0 : n <= thresholds[0] ? 1 : n <= thresholds[1] ? 2 : n <= thresholds[2] ? 3 : 4)
    return { weeks, months, level }
  }, [days])
}

function Heatmap({ days, variant, loading }: { days?: DayMap; variant: 'gh' | 'lc'; loading: boolean }) {
  const { weeks, months, level } = useGrid(days)
  const cls = variant === 'lc' ? 'heat-lc ' : ''
  return (
    <div className="overflow-x-auto pb-1" data-lenis-prevent>
      <div className="inline-block min-w-max">
        <div className="relative mb-1.5 h-4 font-mono text-[10px] text-[var(--fg-subtle)]">
          {months.map((m) => (
            <span key={m.col} className="absolute" style={{ left: m.col * 13 }}>
              {m.label}
            </span>
          ))}
        </div>
        <div className={`flex gap-[3px] ${loading ? 'animate-pulse' : ''}`}>
          {weeks.map((week, i) => (
            <div key={i} className="flex flex-col gap-[3px]">
              {week.map((d) => (
                <span
                  key={d.date}
                  title={d.future ? undefined : `${d.count} on ${d.date}`}
                  className={`h-2.5 w-2.5 rounded-[2px] ${d.future ? 'opacity-0' : cls + 'heat-' + level(d.count)}`}
                />
              ))}
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

function Legend({ variant }: { variant: 'gh' | 'lc' }) {
  const cls = variant === 'lc' ? 'heat-lc ' : ''
  return (
    <div className="flex items-center gap-1 font-mono text-[10px] text-[var(--fg-subtle)]">
      less
      {[0, 1, 2, 3, 4].map((l) => (
        <span key={l} className={`h-2.5 w-2.5 rounded-[2px] ${cls}heat-${l}`} />
      ))}
      more
    </div>
  )
}

function Stat({ label, value }: { label: string; value: string | number }) {
  return (
    <div>
      <p className="font-mono text-xl font-medium tabular-nums">{value}</p>
      <p className="text-xs text-[var(--fg-muted)]">{label}</p>
    </div>
  )
}

function Panel({
  title,
  href,
  icon,
  stats,
  children,
  error,
}: {
  title: string
  href: string
  icon: React.ReactNode
  stats: React.ReactNode
  children: React.ReactNode
  error?: boolean
}) {
  return (
    <div className="card-surface rounded-xl p-5 sm:p-6">
      <div className="mb-5 flex items-center justify-between gap-4">
        <div className="flex items-center gap-2.5">
          {icon}
          <h3 className="font-mono text-sm font-medium">{title}</h3>
        </div>
        <a
          href={href}
          target="_blank"
          rel="noreferrer"
          className="inline-flex items-center gap-1 text-xs text-[var(--fg-muted)] hover:text-[var(--fg)]"
        >
          profile <ArrowUpRight size={13} />
        </a>
      </div>
      <div className="mb-5 flex flex-wrap gap-x-8 gap-y-3">{stats}</div>
      {error ? (
        <p className="py-6 font-mono text-xs text-[var(--fg-subtle)]">// couldn't load activity right now</p>
      ) : (
        children
      )}
    </div>
  )
}

function longestStreak(days?: DayMap) {
  if (!days) return 0
  const keys = Object.keys(days).filter((k) => days[k] > 0).sort()
  let best = 0
  let run = 0
  let prev = 0
  for (const k of keys) {
    const t = Date.parse(k)
    run = t - prev === 86_400_000 ? run + 1 : 1
    best = Math.max(best, run)
    prev = t
  }
  return best
}

export function Activity() {
  const gh = useQuery({ queryKey: ['gh-contrib'], queryFn: fetchGithub, staleTime: 3_600_000, retry: 1 })
  const lc = useQuery({ queryKey: ['lc-calendar'], queryFn: fetchLeetcode, staleTime: 3_600_000, retry: 1 })

  const lcYear = useMemo(() => {
    if (!lc.data) return undefined
    const cutoff = iso(new Date(Date.now() - 365 * 86_400_000))
    return Object.entries(lc.data.days)
      .filter(([k]) => k >= cutoff)
      .reduce((a, [, n]) => a + n, 0)
  }, [lc.data])

  const dash = (v: number | undefined, loading: boolean) => (v === undefined ? (loading ? '…' : '—') : v)

  return (
    <section id="activity" className="px-4 py-20 sm:px-6 md:py-28">
      <div className="mx-auto max-w-7xl">
        <div className="mb-8 flex flex-wrap items-end justify-between gap-4">
          <div>
            <p className="eyebrow mb-2">// activity</p>
            <h2 className="text-3xl font-semibold tracking-tight sm:text-4xl">Shipping &amp; practice, last 12 months</h2>
          </div>
        </div>

        <div className="grid gap-4 lg:grid-cols-2">
          <Panel
            title={`github.com/${GITHUB_USER}`}
            href={`https://github.com/${GITHUB_USER}`}
            icon={<GithubIcon className="h-4 w-4" />}
            error={gh.isError}
            stats={
              <>
                <Stat label="contributions" value={dash(gh.data?.total, gh.isLoading)} />
                <Stat label="longest streak (days)" value={gh.data ? longestStreak(gh.data.days) : dash(undefined, gh.isLoading)} />
              </>
            }
          >
            <Heatmap days={gh.data?.days} variant="gh" loading={gh.isLoading} />
            <div className="mt-3 flex justify-end">
              <Legend variant="gh" />
            </div>
          </Panel>

          <Panel
            title={`leetcode.com/u/${LEETCODE_USER}`}
            href={`https://leetcode.com/u/${LEETCODE_USER}/`}
            icon={<span className="font-mono text-sm font-semibold text-[var(--accent-2)]">{'{}'}</span>}
            error={lc.isError}
            stats={
              <>
                <Stat label="solved" value={dash(lc.data?.solved?.total, lc.isLoading)} />
                {lc.data?.solved && (
                  <div className="flex items-end gap-3 pb-0.5 font-mono text-xs">
                    <span className="text-[var(--accent)]">E {lc.data.solved.easy}</span>
                    <span className="text-[var(--accent-2)]">M {lc.data.solved.medium}</span>
                    <span className="text-[var(--tone-clay)]">H {lc.data.solved.hard}</span>
                  </div>
                )}
                <Stat label="submissions (1y)" value={dash(lcYear, lc.isLoading)} />
              </>
            }
          >
            <Heatmap days={lc.data?.days} variant="lc" loading={lc.isLoading} />
            <div className="mt-3 flex justify-end">
              <Legend variant="lc" />
            </div>
          </Panel>
        </div>
      </div>
    </section>
  )
}
