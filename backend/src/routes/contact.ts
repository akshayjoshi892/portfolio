import { Router } from 'express'
import { z } from 'zod'
import { prisma } from '../lib/prisma.js'
import { sendContactEmail } from '../lib/mail.js'

export const contactRouter = Router()

const contactSchema = z.object({
  name: z.string().min(2).max(100),
  email: z.string().email(),
  message: z.string().min(10).max(5000),
})

contactRouter.post('/', async (req, res) => {
  const parsed = contactSchema.safeParse(req.body)
  if (!parsed.success) return res.status(400).json({ error: parsed.error.flatten() })

  const { name, email, message } = parsed.data
  await prisma.contactMessage.create({ data: { name, email, message } })

  try {
    await sendContactEmail({ name, email, message })
  } catch (err) {
    console.error('Failed to send contact email', err)
  }

  res.status(201).json({ ok: true })
})
