import { useEffect, useMemo, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { AnimatePresence, motion } from 'framer-motion'
import { ChevronRight, Loader2, Search, X } from 'lucide-react'
import { countDocs, loadAll, parseDoc, study, type StudyDoc, type StudyNode } from '@/lib/study'
import { EASE_OUT } from '@/lib/motion'

function ancestorsOf(slug?: string) {
  const open = new Set<string>()
  if (!slug) return open
  const parts = slug.split('/')
  for (let i = 1; i < parts.length; i++) open.add(parts.slice(0, i).join('/'))
  open.add(slug) // a folder's own overview page keeps the folder open
  return open
}

function TreeNode({
  node,
  depth,
  active,
  open,
  toggle,
  onNavigate,
}: {
  node: StudyNode
  depth: number
  active?: string
  open: Set<string>
  toggle: (slug: string, force?: boolean) => void
  onNavigate?: () => void
}) {
  const pad = { paddingLeft: `${depth * 14 + 10}px` }

  if (node.kind === 'doc') {
    const isActive = node.slug === active
    return (
      <Link
        to={`/study/${node.slug}`}
        onClick={onNavigate}
        style={pad}
        aria-current={isActive ? 'page' : undefined}
        className={`relative flex items-center gap-2 rounded-lg py-1.5 pr-2 text-[13.5px] leading-snug transition-colors ${
          isActive ? 'font-medium text-[var(--fg)]' : 'text-[var(--fg-muted)] hover:text-[var(--fg)]'
        }`}
      >
        {isActive && (
          <motion.span
            layoutId="study-active"
            className="absolute inset-0 rounded-lg bg-[var(--bg-inset)]"
            transition={{ type: 'spring', stiffness: 420, damping: 36 }}
          />
        )}
        <span
          className={`relative h-1 w-1 shrink-0 rounded-full transition-colors ${isActive ? 'bg-[var(--accent)]' : 'bg-[var(--border-strong)]'}`}
        />
        <span className="relative">{node.title}</span>
      </Link>
    )
  }

  const isOpen = open.has(node.slug)
  const isActive = node.index?.slug === active
  return (
    <div>
      <div className="relative flex items-center" style={pad}>
        {isActive && (
          <motion.span
            layoutId="study-active"
            className="absolute inset-0 rounded-lg bg-[var(--bg-inset)]"
            transition={{ type: 'spring', stiffness: 420, damping: 36 }}
          />
        )}
        <button
          onClick={() => toggle(node.slug)}
          aria-label={isOpen ? `Collapse ${node.title}` : `Expand ${node.title}`}
          aria-expanded={isOpen}
          className="relative -ml-1 grid h-6 w-6 shrink-0 place-items-center rounded text-[var(--fg-subtle)] hover:text-[var(--fg)]"
        >
          <motion.span animate={{ rotate: isOpen ? 90 : 0 }} transition={{ duration: 0.25, ease: EASE_OUT }}>
            <ChevronRight size={14} />
          </motion.span>
        </button>
        {node.index ? (
          <Link
            to={`/study/${node.index.slug}`}
            onClick={() => {
              toggle(node.slug, true)
              onNavigate?.()
            }}
            className="relative flex-1 py-1.5 pr-2 text-[13.5px] font-medium text-[var(--fg)]"
          >
            {node.title}
          </Link>
        ) : (
          <button onClick={() => toggle(node.slug)} className="relative flex-1 py-1.5 pr-2 text-left text-[13.5px] font-medium text-[var(--fg)]">
            {node.title}
          </button>
        )}
        <span className="relative pr-2 font-mono text-[10px] text-[var(--fg-subtle)]">{countDocs(node)}</span>
      </div>
      <AnimatePresence initial={false}>
        {isOpen && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3, ease: EASE_OUT }}
            className="relative overflow-hidden"
          >
            <span
              aria-hidden
              className="absolute bottom-1 top-1 w-px bg-[var(--border)]"
              style={{ left: `${depth * 14 + 21}px` }}
            />
            {node.children.map((child) => (
              <TreeNode
                key={child.slug + child.kind}
                node={child}
                depth={depth + 1}
                active={active}
                open={open}
                toggle={toggle}
                onNavigate={onNavigate}
              />
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

/* ---------- Search ---------- */

interface Hit {
  doc: StudyDoc
  snippet?: { before: string; match: string; after: string }
}

function useSearch(query: string) {
  const [corpus, setCorpus] = useState<{ doc: StudyDoc; text: string }[] | null>(null)
  const [loading, setLoading] = useState(false)
  const q = query.trim().toLowerCase()

  useEffect(() => {
    if (q.length < 2 || corpus || loading) return
    setLoading(true)
    loadAll()
      .then((all) => setCorpus(all.map(({ doc, raw }) => ({ doc, text: parseDoc(raw).body }))))
      .finally(() => setLoading(false))
  }, [q, corpus, loading])

  const hits = useMemo<Hit[]>(() => {
    if (!q) return []
    const out: Hit[] = []
    const seen = new Set<string>()
    for (const doc of study.ordered) {
      const hay = [...doc.trail, doc.title].join(' ').toLowerCase()
      if (hay.includes(q)) {
        out.push({ doc })
        seen.add(doc.slug)
      }
    }
    if (corpus && q.length >= 2) {
      for (const { doc, text } of corpus) {
        if (seen.has(doc.slug)) continue
        const i = text.toLowerCase().indexOf(q)
        if (i === -1) continue
        const clean = (s: string) => s.replace(/[#*_`>[\]]/g, '').replace(/\s+/g, ' ')
        out.push({
          doc,
          snippet: {
            before: (i > 50 ? '…' : '') + clean(text.slice(Math.max(0, i - 50), i)).trimStart(),
            match: text.slice(i, i + q.length),
            after: clean(text.slice(i + q.length, i + q.length + 70)) + '…',
          },
        })
      }
    }
    return out.slice(0, 50)
  }, [q, corpus])

  return { hits, loading: loading && q.length >= 2 }
}

/* ---------- Sidebar ---------- */

export function StudySidebar({ active, onNavigate }: { active?: string; onNavigate?: () => void }) {
  const [query, setQuery] = useState('')
  const [open, setOpen] = useState(() => ancestorsOf(active))
  const inputRef = useRef<HTMLInputElement>(null)
  const { hits, loading } = useSearch(query)

  // Reveal the active doc whenever it changes.
  const [lastActive, setLastActive] = useState(active)
  if (active !== lastActive) {
    setLastActive(active)
    setOpen((prev) => new Set([...prev, ...ancestorsOf(active)]))
  }

  // "/" focuses search, Esc clears it.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const typing = (e.target as HTMLElement)?.closest('input, textarea, [contenteditable]')
      if (e.key === '/' && !typing) {
        e.preventDefault()
        inputRef.current?.focus()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  function toggle(slug: string, force?: boolean) {
    setOpen((prev) => {
      const next = new Set(prev)
      if (force ?? !next.has(slug)) next.add(slug)
      else next.delete(slug)
      return next
    })
  }

  return (
    <div className="flex h-full flex-col">
      <label className="relative mb-3 block shrink-0">
        <Search size={14} className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-[var(--fg-subtle)]" />
        <input
          ref={inputRef}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === 'Escape' && setQuery('')}
          placeholder="Search notes…"
          aria-label="Search study notes"
          className="w-full rounded-xl border border-[var(--border)] bg-[var(--bg-elevated)] py-2 pl-9 pr-9 text-sm outline-none transition-colors placeholder:text-[var(--fg-subtle)] focus:border-[var(--border-strong)]"
        />
        {query ? (
          <button
            onClick={() => setQuery('')}
            aria-label="Clear search"
            className="absolute right-2 top-1/2 grid h-6 w-6 -translate-y-1/2 place-items-center rounded text-[var(--fg-subtle)] hover:text-[var(--fg)]"
          >
            <X size={13} />
          </button>
        ) : (
          <kbd className="pointer-events-none absolute right-2.5 top-1/2 hidden -translate-y-1/2 rounded border border-[var(--border)] px-1.5 font-mono text-[10px] text-[var(--fg-subtle)] md:block">
            /
          </kbd>
        )}
      </label>

      <div className="-mx-1 min-h-0 flex-1 overflow-y-auto overscroll-contain px-1 pb-6" data-lenis-prevent>
        {query ? (
          <div>
            <p className="eyebrow mb-2 flex items-center gap-2 px-2">
              {hits.length} result{hits.length === 1 ? '' : 's'}
              {loading && <Loader2 size={11} className="animate-spin" />}
            </p>
            <ul className="space-y-0.5">
              {hits.map(({ doc, snippet }) => (
                <li key={doc.slug}>
                  <Link
                    to={`/study/${doc.slug}`}
                    onClick={onNavigate}
                    className={`block rounded-lg px-2 py-2 transition-colors hover:bg-[var(--bg-inset)] ${doc.slug === active ? 'bg-[var(--bg-inset)]' : ''}`}
                  >
                    {doc.trail.length > 0 && (
                      <span className="block truncate font-mono text-[10px] uppercase tracking-wider text-[var(--fg-subtle)]">
                        {doc.trail.join(' / ')}
                      </span>
                    )}
                    <span className="block text-sm text-[var(--fg)]">{doc.title}</span>
                    {snippet && (
                      <span className="mt-0.5 line-clamp-2 block text-xs text-[var(--fg-muted)]">
                        {snippet.before}
                        <mark className="search-hit">{snippet.match}</mark>
                        {snippet.after}
                      </span>
                    )}
                  </Link>
                </li>
              ))}
            </ul>
            {!loading && hits.length === 0 && <p className="px-2 text-sm text-[var(--fg-muted)]">Nothing matches “{query}”.</p>}
          </div>
        ) : (
          <nav aria-label="Study notes" className="space-y-0.5">
            {study.tree.map((node) => (
              <TreeNode key={node.slug + node.kind} node={node} depth={0} active={active} open={open} toggle={toggle} onNavigate={onNavigate} />
            ))}
          </nav>
        )}
      </div>
    </div>
  )
}
