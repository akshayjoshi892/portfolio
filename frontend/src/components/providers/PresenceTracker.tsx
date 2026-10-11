import { useEffect } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { useAuth } from '@/context/AuthContext'
import { api } from '@/lib/api'

const OWNER_INTERVAL_MS = 60_000
const VISITOR_INTERVAL_MS = 120_000

function visitorId() {
  try {
    let id = localStorage.getItem('visitor_id')
    if (!id) {
      id = crypto.randomUUID()
      localStorage.setItem('visitor_id', id)
    }
    return id
  } catch {
    return null // storage blocked: don't track
  }
}

function isNewSession() {
  try {
    if (sessionStorage.getItem('visit_counted')) return false
    sessionStorage.setItem('visit_counted', '1')
    return true
  } catch {
    return false
  }
}

/**
 * Signed in (the owner): sends a heartbeat so the site can show "Online now" /
 * "Last online …". Everyone else is counted as an anonymous visitor.
 * Pings only while the tab is visible.
 */
export function PresenceTracker() {
  const { isAuthenticated } = useAuth()
  const queryClient = useQueryClient()

  useEffect(() => {
    let firstPing = true
    const ping = () => {
      if (document.visibilityState !== 'visible') return
      if (isAuthenticated) {
        const refresh = firstPing
        api
          .post('/admin/status/heartbeat')
          .then(() => {
            if (refresh) void queryClient.invalidateQueries({ queryKey: ['status'] })
          })
          .catch(() => {})
      } else {
        const id = visitorId()
        if (!id) return
        api.post('/visits', { visitorId: id, newSession: firstPing && isNewSession() }).catch(() => {})
      }
      firstPing = false
    }

    ping()
    const timer = setInterval(ping, isAuthenticated ? OWNER_INTERVAL_MS : VISITOR_INTERVAL_MS)
    document.addEventListener('visibilitychange', ping)
    return () => {
      clearInterval(timer)
      document.removeEventListener('visibilitychange', ping)
    }
  }, [isAuthenticated, queryClient])

  return null
}
