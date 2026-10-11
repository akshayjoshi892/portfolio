/**
 * File-based study library.
 *
 * Every `.md` file under `frontend/content/study/` becomes a page. Folders become
 * sections. Numeric prefixes (`01-intro.md`) control order and are stripped from
 * titles and URLs. A folder's `index.md` / `README.md` is used as its overview page.
 * Document bodies are code-split and only fetched when opened (or when searching).
 */

const ROOT = '/content/study/'

const loaders = import.meta.glob<string>('/content/study/**/*.md', { query: '?raw', import: 'default' })
const assetUrls = import.meta.glob<string>('/content/study/**/*.{png,jpg,jpeg,gif,svg,webp,avif}', {
  query: '?url',
  import: 'default',
  eager: true,
})

export interface StudyDoc {
  kind: 'doc'
  /** URL slug, e.g. `data-structures/arrays` */
  slug: string
  /** Source path relative to the study root, e.g. `02-data-structures/01-arrays.md` */
  file: string
  title: string
  /** Titles of the parent folders, outermost first. */
  trail: string[]
}

export interface StudyFolder {
  kind: 'folder'
  slug: string
  title: string
  /** The folder's own overview page (index.md / README.md), if present. */
  index?: StudyDoc
  children: StudyNode[]
}

export type StudyNode = StudyDoc | StudyFolder

const ORDER_PREFIX = /^(\d+)[\s._-]+/
const INDEX_NAMES = new Set(['index', 'readme'])

function stripExt(name: string) {
  return name.replace(/\.md$/i, '')
}

function orderOf(name: string) {
  const m = stripExt(name).match(ORDER_PREFIX)
  return m ? Number(m[1]) : Number.POSITIVE_INFINITY
}

export function prettify(name: string) {
  const base = stripExt(name).replace(ORDER_PREFIX, '').replace(/[-_]+/g, ' ').trim()
  if (!base) return stripExt(name)
  // Keep words that are already cased (e.g. "SQL", "useEffect"); capitalise the rest.
  return base
    .split(' ')
    .map((w) => (w === w.toLowerCase() ? w.charAt(0).toUpperCase() + w.slice(1) : w))
    .join(' ')
}

function slugPart(name: string) {
  return (
    stripExt(name)
      .replace(ORDER_PREFIX, '')
      .toLowerCase()
      .replace(/[^\p{L}\p{N}]+/gu, '-')
      .replace(/^-+|-+$/g, '') || stripExt(name).toLowerCase()
  )
}

function compare(a: { order: number; title: string }, b: { order: number; title: string }) {
  return a.order - b.order || a.title.localeCompare(b.title, undefined, { numeric: true })
}

interface MutableFolder extends StudyFolder {
  order: number
  folders: Map<string, MutableFolder>
  docs: (StudyDoc & { order: number })[]
}

function buildTree() {
  const root: MutableFolder = { kind: 'folder', slug: '', title: 'Study', children: [], order: 0, folders: new Map(), docs: [] }
  const bySlug = new Map<string, StudyDoc>()
  const byFile = new Map<string, StudyDoc>()

  for (const path of Object.keys(loaders)) {
    const file = path.slice(ROOT.length)
    const parts = file.split('/')
    const name = parts.pop()!

    let folder = root
    const trail: string[] = []
    for (const dir of parts) {
      let next = folder.folders.get(dir)
      if (!next) {
        next = {
          kind: 'folder',
          slug: [folder.slug, slugPart(dir)].filter(Boolean).join('/'),
          title: prettify(dir),
          children: [],
          order: orderOf(dir),
          folders: new Map(),
          docs: [],
        }
        folder.folders.set(dir, next)
      }
      trail.push(next.title)
      folder = next
    }

    const isIndex = INDEX_NAMES.has(stripExt(name).toLowerCase())
    const slug = isIndex ? folder.slug || 'index' : [folder.slug, slugPart(name)].filter(Boolean).join('/')
    const doc: StudyDoc = {
      kind: 'doc',
      slug,
      file,
      title: isIndex ? folder.title === 'Study' ? 'Overview' : folder.title : prettify(name),
      trail,
    }
    if (isIndex && folder !== root) folder.index = doc
    else folder.docs.push({ ...doc, order: isIndex ? -1 : orderOf(name) })
    bySlug.set(slug, doc)
    byFile.set(file, doc)
  }

  const finalize = (f: MutableFolder): StudyNode[] => {
    const folders = [...f.folders.values()].sort(compare)
    const items: (StudyNode & { order: number })[] = [
      ...f.docs.map((d) => ({ ...d })),
      ...folders.map((sub) => {
        const node: StudyFolder & { order: number } = {
          kind: 'folder',
          slug: sub.slug,
          title: sub.title,
          index: sub.index,
          children: finalize(sub),
          order: sub.order,
        }
        return node
      }),
    ]
    // Docs and folders are interleaved by their numeric prefix; unnumbered docs first, then folders.
    items.sort((a, b) => {
      if (a.order !== b.order) return a.order - b.order
      if (a.kind !== b.kind) return a.kind === 'doc' ? -1 : 1
      return a.title.localeCompare(b.title, undefined, { numeric: true })
    })
    return items.map(({ order: _order, ...rest }) => rest as StudyNode)
  }

  const tree = finalize(root)

  // Reading order = depth-first walk (folder overview before its children).
  const ordered: StudyDoc[] = []
  const walk = (nodes: StudyNode[]) => {
    for (const n of nodes) {
      if (n.kind === 'doc') ordered.push(bySlug.get(n.slug)!)
      else {
        if (n.index) ordered.push(n.index)
        walk(n.children)
      }
    }
  }
  walk(tree)

  return { tree, ordered, bySlug, byFile }
}

export const study = buildTree()

const cache = new Map<string, string>()

export async function loadDoc(doc: StudyDoc): Promise<string> {
  const hit = cache.get(doc.file)
  if (hit !== undefined) return hit
  const raw = await loaders[ROOT + doc.file]()
  cache.set(doc.file, raw)
  return raw
}

export async function loadAll() {
  await Promise.all(study.ordered.map(loadDoc))
  return study.ordered.map((doc) => ({ doc, raw: cache.get(doc.file)! }))
}

/* ---------- Parsing helpers ---------- */

/** Splits simple YAML frontmatter (`key: value` lines) from the body. */
export function parseDoc(raw: string) {
  const meta: Record<string, string> = {}
  let body = raw.replace(/^﻿/, '')
  const fm = body.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n?/)
  if (fm) {
    for (const line of fm[1].split(/\r?\n/)) {
      const m = line.match(/^([\w-]+)\s*:\s*(.*)$/)
      if (m) meta[m[1].toLowerCase()] = m[2].replace(/^["']|["']$/g, '').trim()
    }
    body = body.slice(fm[0].length)
  }
  // Promote a leading `# Heading` to the page title so it isn't rendered twice.
  let title = meta.title
  const h1 = body.match(/^\s*#\s+(.+?)\s*#*\s*(?:\r?\n|$)/)
  if (h1) {
    title ??= h1[1]
    body = body.slice(h1[0].length)
  }
  const words = body.replace(/```[\s\S]*?```/g, ' ').split(/\s+/).filter(Boolean).length
  return { meta, title, body, minutes: Math.max(1, Math.round(words / 220)) }
}

function dirOf(file: string) {
  const i = file.lastIndexOf('/')
  return i === -1 ? '' : file.slice(0, i + 1)
}

function resolvePath(fromFile: string, rel: string) {
  const out: string[] = []
  for (const seg of (dirOf(fromFile) + rel).split('/')) {
    if (seg === '..') out.pop()
    else if (seg && seg !== '.') out.push(seg)
  }
  return out.join('/')
}

function isExternal(href: string) {
  return /^([a-z][a-z\d+.-]*:|\/\/)/i.test(href)
}

/** Maps a link inside a doc to an in-app route (for `.md` targets) or leaves it alone. */
export function resolveLink(fromFile: string, href: string): { to?: string; href: string; external: boolean } {
  if (!href || href.startsWith('#')) return { href, external: false }
  if (isExternal(href)) return { href, external: true }
  const [pathPart, hash] = href.split('#')
  const decoded = decodeURIComponent(pathPart)
  if (/\.md$/i.test(decoded)) {
    const target = study.byFile.get(resolvePath(fromFile, decoded))
    if (target) return { to: `/study/${target.slug}${hash ? `#${hash}` : ''}`, href, external: false }
  }
  return { href, external: false }
}

/** Resolves a relative image path to the bundled asset URL. */
export function resolveAsset(fromFile: string, src: string) {
  if (!src || isExternal(src) || src.startsWith('/')) return src
  return assetUrls[ROOT + resolvePath(fromFile, decodeURIComponent(src))] ?? src
}

export function countDocs(node: StudyNode): number {
  if (node.kind === 'doc') return 1
  return (node.index ? 1 : 0) + node.children.reduce((n, c) => n + countDocs(c), 0)
}
