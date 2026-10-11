import { Router } from 'express'
import { z } from 'zod'
import { prisma } from '../lib/prisma.js'
import { requireAuth } from '../middleware/requireAuth.js'
import { slugify } from '../lib/slugify.js'

export const studyRouter = Router()
export const adminStudyRouter = Router()

interface TreeNode {
  id: string
  slug: string
  title: string
  icon: string | null
  order: number
  parentId: string | null
  children: TreeNode[]
}

function buildTree(pages: { id: string; slug: string; title: string; icon: string | null; order: number; parentId: string | null }[]): TreeNode[] {
  const byId = new Map<string, TreeNode>(pages.map((p) => [p.id, { ...p, children: [] }]))
  const roots: TreeNode[] = []

  for (const page of byId.values()) {
    if (page.parentId && byId.has(page.parentId)) {
      byId.get(page.parentId)!.children.push(page)
    } else {
      roots.push(page)
    }
  }

  const sortRec = (nodes: TreeNode[]) => {
    nodes.sort((a, b) => a.order - b.order)
    nodes.forEach((n) => sortRec(n.children))
  }
  sortRec(roots)
  return roots
}

studyRouter.get('/', async (_req, res) => {
  const pages = await prisma.studyPage.findMany({
    select: { id: true, slug: true, title: true, icon: true, order: true, parentId: true },
  })
  res.json(buildTree(pages))
})

studyRouter.get('/:slug', async (req, res) => {
  const page = await prisma.studyPage.findUnique({ where: { slug: req.params.slug } })
  if (!page) return res.status(404).json({ error: 'Page not found' })
  res.json(page)
})

const createPageSchema = z.object({
  title: z.string().min(1),
  content: z.string().default(''),
  parentId: z.string().nullable().default(null),
  icon: z.string().optional(),
  order: z.number().default(0),
})

adminStudyRouter.use(requireAuth)

adminStudyRouter.post('/', async (req, res) => {
  const parsed = createPageSchema.safeParse(req.body)
  if (!parsed.success) return res.status(400).json({ error: parsed.error.flatten() })

  const { title, content, parentId, icon, order } = parsed.data
  const baseSlug = slugify(title)
  let slug = baseSlug
  let suffix = 1
  while (await prisma.studyPage.findUnique({ where: { slug } })) {
    slug = `${baseSlug}-${suffix++}`
  }

  const page = await prisma.studyPage.create({
    data: { slug, title, content, parentId, icon, order },
  })
  res.status(201).json(page)
})

adminStudyRouter.put('/:id', async (req, res) => {
  const parsed = createPageSchema.partial().safeParse(req.body)
  if (!parsed.success) return res.status(400).json({ error: parsed.error.flatten() })

  const page = await prisma.studyPage.update({ where: { id: req.params.id }, data: parsed.data })
  res.json(page)
})

adminStudyRouter.delete('/:id', async (req, res) => {
  await prisma.studyPage.delete({ where: { id: req.params.id } })
  res.status(204).send()
})
