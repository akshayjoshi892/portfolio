import { useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api'
import { timeAgo } from '@/lib/time'
import type { OnlineStatus } from '@/types'

const MANUAL_LABEL: Partial<Record<OnlineStatus['status'], string>> = {
  coding: 'Currently coding',
  away: 'Away',
}

async function fetchStatus(): Promise<OnlineStatus> {
  const { data } = await api.get<OnlineStatus>('/status')
  return data
}

/**
 * Green "Online now" while the owner is active (heartbeat from a signed-in
 * session), otherwise "Last online 2 h ago". Falls back to an availability
 * note when the API isn't reachable.
 */
export function StatusPill() {
  const { data } = useQuery({ queryKey: ['status'], queryFn: fetchStatus, refetchInterval: 30_000, retry: 1 })

  let color = '#22c55e'
  let label = 'Available for select projects'
  let pulse = true
  let detail: string | undefined

  if (data?.online) {
    label = MANUAL_LABEL[data.status] ?? 'Online now'
    if (data.status === 'away') color = '#f59e0b'
    detail = data.activity || undefined
  } else if (data?.lastSeenAt) {
    color = 'var(--fg-subtle)'
    pulse = false
    label = `Last online ${timeAgo(data.lastSeenAt)}`
  }

  return (
    <div
      className="glass inline-flex items-center gap-2.5 rounded-full border border-[var(--border)] py-1.5 pl-3 pr-4 text-xs"
      title={data?.lastSeenAt ? `Last seen ${new Date(data.lastSeenAt).toLocaleString()}` : undefined}
    >
      <span className="relative flex h-2 w-2">
        {pulse && (
          <span className="absolute inline-flex h-full w-full animate-ping rounded-full opacity-60" style={{ backgroundColor: color }} />
        )}
        <span className="relative inline-flex h-2 w-2 rounded-full" style={{ backgroundColor: color }} />
      </span>
      <span className="font-medium">{label}</span>
      {detail && <span className="hidden text-[var(--fg-muted)] sm:inline">— {detail}</span>}
    </div>
  )
}
