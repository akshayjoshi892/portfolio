import { Router } from 'express'
import { z } from 'zod'
import { prisma } from '../lib/prisma.js'
import { requireAuth } from '../middleware/requireAuth.js'
import { slugify, estimateReadTime } from '../lib/slugify.js'

export const blogRouter = Router()
export const adminBlogRouter = Router()

blogRouter.get('/', async (_req, res) => {
  const posts = await prisma.blogPost.findMany({
    where: { published: true },
    orderBy: { publishedAt: 'desc' },
  })
  res.json(posts)
})

blogRouter.get('/:slug', async (req, res) => {
  const post = await prisma.blogPost.findUnique({ where: { slug: req.params.slug } })
  if (!post || !post.published) return res.status(404).json({ error: 'Post not found' })
  res.json(post)
})

const createPostSchema = z.object({
  title: z.string().min(3),
  excerpt: z.string().min(3),
  content: z.string().min(10),
  tags: z.array(z.string()).default([]),
  coverImage: z.string().url().optional(),
  published: z.boolean().default(true),
})

adminBlogRouter.use(requireAuth)

adminBlogRouter.get('/', async (_req, res) => {
  const posts = await prisma.blogPost.findMany({ orderBy: { publishedAt: 'desc' } })
  res.json(posts)
})

adminBlogRouter.post('/', async (req, res) => {
  const parsed = createPostSchema.safeParse(req.body)
  if (!parsed.success) return res.status(400).json({ error: parsed.error.flatten() })

  const { title, excerpt, content, tags, coverImage, published } = parsed.data
  const baseSlug = slugify(title)
  let slug = baseSlug
  let suffix = 1
  while (await prisma.blogPost.findUnique({ where: { slug } })) {
    slug = `${baseSlug}-${suffix++}`
  }

  const post = await prisma.blogPost.create({
    data: {
      slug,
      title,
      excerpt,
      content,
      tags,
      coverImage,
      published,
      readTimeMinutes: estimateReadTime(content),
    },
  })
  res.status(201).json(post)
})

adminBlogRouter.put('/:id', async (req, res) => {
  const parsed = createPostSchema.partial().safeParse(req.body)
  if (!parsed.success) return res.status(400).json({ error: parsed.error.flatten() })

  const data = parsed.data
  const post = await prisma.blogPost.update({
    where: { id: req.params.id },
    data: {
      ...data,
      ...(data.content ? { readTimeMinutes: estimateReadTime(data.content) } : {}),
    },
  })
  res.json(post)
})

adminBlogRouter.delete('/:id', async (req, res) => {
  await prisma.blogPost.delete({ where: { id: req.params.id } })
  res.status(204).send()
})
