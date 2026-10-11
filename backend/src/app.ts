import express from 'express'
import cors from 'cors'
import helmet from 'helmet'
import { authRouter } from './routes/auth.js'
import { blogRouter, adminBlogRouter } from './routes/blog.js'
import { studyRouter, adminStudyRouter } from './routes/study.js'
import { statusRouter, adminStatusRouter } from './routes/status.js'
import { contactRouter } from './routes/contact.js'
import { visitsRouter } from './routes/visits.js'

export function createApp() {
  const app = express()

  app.use(helmet())
  app.use(
    cors({
      origin: process.env.CORS_ORIGIN?.split(',') ?? '*',
    }),
  )
  app.use(express.json({ limit: '1mb' }))

  app.get('/api/health', (_req, res) => res.json({ ok: true }))

  app.use('/api/auth', authRouter)
  app.use('/api/blog', blogRouter)
  app.use('/api/study', studyRouter)
  app.use('/api/status', statusRouter)
  app.use('/api/contact', contactRouter)
  app.use('/api/visits', visitsRouter)

  app.use('/api/admin/blog', adminBlogRouter)
  app.use('/api/admin/study', adminStudyRouter)
  app.use('/api/admin/status', adminStatusRouter)

  app.use((_req, res) => res.status(404).json({ error: 'Not found' }))

  return app
}
