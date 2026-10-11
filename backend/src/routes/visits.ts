import { Router } from 'express'
import { z } from 'zod'
import { prisma } from '../lib/prisma.js'

export const visitsRouter = Router()

/** A visitor counts as "active" if they pinged within this window. */
const ACTIVE_WINDOW_MS = 5 * 60 * 1000
const STATS_TTL_MS = 30 * 1000

const pingSchema = z.object({
  visitorId: z.string().uuid(),
  /** True on the first ping of a browser session, which counts as a new visit. */
  newSession: z.boolean().default(false),
})

visitsRouter.post('/', async (req, res) => {
  const parsed = pingSchema.safeParse(req.body)
  if (!parsed.success) return res.status(400).json({ error: 'Invalid visitor id' })

  const { visitorId, newSession } = parsed.data
  const now = new Date()
  await prisma.visitor.upsert({
    where: { id: visitorId },
    create: { id: visitorId, firstSeenAt: now, lastSeenAt: now },
    update: { lastSeenAt: now, ...(newSession ? { visits: { increment: 1 } } : {}) },
  })
  cachedStats = null
  res.status(204).end()
})

interface Stats {
  totalVisitors: number
  totalVisits: number
  todayVisitors: number
  activeNow: number
}

let cachedStats: { at: number; data: Stats } | null = null

visitsRouter.get('/stats', async (_req, res) => {
  if (cachedStats && Date.now() - cachedStats.at < STATS_TTL_MS) return res.json(cachedStats.data)

  const now = Date.now()
  const startOfDay = new Date()
  startOfDay.setHours(0, 0, 0, 0)

  const [totalVisitors, sum, todayVisitors, activeNow] = await Promise.all([
    prisma.visitor.count(),
    prisma.visitor.aggregate({ _sum: { visits: true } }),
    prisma.visitor.count({ where: { lastSeenAt: { gte: startOfDay } } }),
    prisma.visitor.count({ where: { lastSeenAt: { gte: new Date(now - ACTIVE_WINDOW_MS) } } }),
  ])

  const data: Stats = { totalVisitors, totalVisits: sum._sum.visits ?? 0, todayVisitors, activeNow }
  cachedStats = { at: now, data }
  res.json(data)
})
