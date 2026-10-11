import { useEffect, useState, type RefObject } from 'react'
import { motion } from 'framer-motion'
import { useSmoothScroll } from '@/components/providers/SmoothScroll'

interface Entry {
  id: string
  text: string
  level: number
}

/** Builds an outline from the rendered h2/h3s and highlights the section in view. */
export function TableOfContents({ articleRef, deps }: { articleRef: RefObject<HTMLElement | null>; deps: unknown }) {
  const [entries, setEntries] = useState<Entry[]>([])
  const [activeId, setActiveId] = useState<string | null>(null)
  const { scrollTo } = useSmoothScroll()

  useEffect(() => {
    const frame = requestAnimationFrame(() => {
      const nodes = articleRef.current?.querySelectorAll<HTMLElement>('.md-body h2[id], .md-body h3[id]') ?? []
      setEntries(
        [...nodes].map((n) => ({
          id: n.id,
          text: n.textContent?.trim() ?? '',
          level: n.tagName === 'H2' ? 2 : 3,
        })),
      )
    })
    return () => cancelAnimationFrame(frame)
  }, [articleRef, deps])

  useEffect(() => {
    if (!entries.length) return
    const onScroll = () => {
      let current = entries[0].id
      for (const e of entries) {
        const el = document.getElementById(e.id)
        if (el && el.getBoundingClientRect().top < 140) current = e.id
      }
      setActiveId(current)
    }
    onScroll()
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [entries])

  if (entries.length < 2) return null

  return (
    <nav aria-label="On this page">
      <p className="eyebrow mb-4">On this page</p>
      <ul className="relative space-y-0.5 border-l border-[var(--border)]">
        {entries.map((e) => {
          const active = e.id === activeId
          return (
            <li key={e.id} className="relative">
              {active && (
                <motion.span
                  layoutId="toc-active"
                  className="absolute -left-px top-0 h-full w-px bg-[var(--accent)]"
                  transition={{ type: 'spring', stiffness: 400, damping: 34 }}
                />
              )}
              <a
                href={`#${e.id}`}
                onClick={(ev) => {
                  ev.preventDefault()
                  const el = document.getElementById(e.id)
                  if (el) scrollTo(el, { offset: -110 })
                  history.replaceState(null, '', `#${e.id}`)
                }}
                className={`block py-1 text-[13px] leading-snug transition-colors ${e.level === 3 ? 'pl-7' : 'pl-4'} ${
                  active ? 'text-[var(--fg)]' : 'text-[var(--fg-muted)] hover:text-[var(--fg)]'
                }`}
              >
                {e.text}
              </a>
            </li>
          )
        })}
      </ul>
    </nav>
  )
}
