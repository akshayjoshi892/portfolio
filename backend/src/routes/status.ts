import { Router } from 'express'
import { z } from 'zod'
import { prisma } from '../lib/prisma.js'
import { requireAuth } from '../middleware/requireAuth.js'

export const statusRouter = Router()
export const adminStatusRouter = Router()

const SINGLETON_ID = 'status-singleton'

/** The owner counts as online if a heartbeat arrived within this window. */
const ONLINE_WINDOW_MS = 3 * 60 * 1000

statusRouter.get('/', async (_req, res) => {
  const status = await prisma.projectStatus.findUnique({ where: { id: SINGLETON_ID } })
  const lastSeenAt = status?.lastSeenAt ?? null
  res.json({
    status: status?.status ?? 'offline',
    activity: status?.activity ?? '',
    updatedAt: (status?.updatedAt ?? new Date()).toISOString(),
    lastSeenAt: lastSeenAt?.toISOString() ?? null,
    online: lastSeenAt ? Date.now() - lastSeenAt.getTime() < ONLINE_WINDOW_MS : false,
  })
})

const updateStatusSchema = z.object({
  status: z.enum(['online', 'away', 'coding', 'offline']),
  activity: z.string().max(200).default(''),
})

adminStatusRouter.use(requireAuth)

adminStatusRouter.put('/', async (req, res) => {
  const parsed = updateStatusSchema.safeParse(req.body)
  if (!parsed.success) return res.status(400).json({ error: parsed.error.flatten() })

  const { status, activity } = parsed.data
  const updated = await prisma.projectStatus.upsert({
    where: { id: SINGLETON_ID },
    create: { id: SINGLETON_ID, status, activity },
    update: { status, activity },
  })
  res.json(updated)
})

/** Called periodically by the site while the owner is signed in. */
adminStatusRouter.post('/heartbeat', async (_req, res) => {
  const now = new Date()
  await prisma.projectStatus.upsert({
    where: { id: SINGLETON_ID },
    create: { id: SINGLETON_ID, lastSeenAt: now },
    update: { lastSeenAt: now },
  })
  res.status(204).end()
})
