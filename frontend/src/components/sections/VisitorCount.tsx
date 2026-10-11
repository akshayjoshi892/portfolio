import { useQuery } from '@tanstack/react-query'
import { Users } from 'lucide-react'
import { Counter } from '@/components/animations/Counter'
import { api } from '@/lib/api'
import type { VisitorStats } from '@/types'

async function fetchStats(): Promise<VisitorStats> {
  const { data } = await api.get<VisitorStats>('/visits/stats')
  return data
}

/** Unique visitor count, plus how many are here right now. Hidden if the API is down. */
export function VisitorCount() {
  const { data } = useQuery({ queryKey: ['visitor-stats'], queryFn: fetchStats, refetchInterval: 60_000, retry: 1 })
  if (!data) return null

  return (
    <div
      className="glass inline-flex items-center gap-2 rounded-full border border-[var(--border)] py-1.5 pl-3 pr-4 text-xs"
      title={`${data.totalVisits.toLocaleString()} visits · ${data.todayVisitors.toLocaleString()} visitors today`}
    >
      <Users size={13} className="text-[var(--fg-muted)]" />
      <span className="font-medium">
        <Counter to={data.totalVisitors} /> visitors
      </span>
      {data.activeNow > 0 && (
        <span className="text-[var(--fg-muted)]">
          · <span className="text-[var(--fg)]">{data.activeNow}</span> here now
        </span>
      )}
    </div>
  )
}
