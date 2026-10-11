import { useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { AnimatePresence, motion } from 'framer-motion'
import { ArrowUpRight, ExternalLink, FolderGit2 } from 'lucide-react'
import { GithubIcon } from '@/components/ui/GithubIcon'
import { PROJECTS } from '@/data/sample'
import { EASE_OUT } from '@/lib/motion'
import { toneAt, toneFor } from '@/lib/tones'
import type { Project } from '@/types'

function ProjectCard({ project, index }: { project: Project; index: number }) {
  const tone = toneAt(index)
  return (
    <article
      className="group relative flex h-full flex-col rounded-xl border border-[var(--border)] bg-[var(--bg-elevated)] p-5 transition-colors hover:border-[var(--border-strong)] sm:p-6"
      style={{ ['--c' as string]: tone }}
    >
      <span aria-hidden className="absolute inset-x-0 top-0 h-[3px] rounded-t-xl" style={{ background: tone, opacity: 0.85 }} />

      <div className="mb-4 flex items-start justify-between gap-4">
        <div className="flex items-center gap-2.5">
          <span
            className="grid h-9 w-9 place-items-center rounded-lg"
            style={{ background: `color-mix(in oklab, ${tone} 14%, transparent)`, color: tone }}
          >
            <FolderGit2 size={17} />
          </span>
          {project.featured && <span className="chip">featured</span>}
        </div>
        <div className="flex items-center gap-0.5 text-[var(--fg-muted)]">
          {project.githubUrl && (
            <a href={project.githubUrl} target="_blank" rel="noreferrer" aria-label={`${project.title} source`} className="rounded-md p-2 hover:bg-[var(--bg-inset)] hover:text-[var(--fg)]">
              <GithubIcon size={16} />
            </a>
          )}
          {project.liveUrl && (
            <a href={project.liveUrl} target="_blank" rel="noreferrer" aria-label={`${project.title} live demo`} className="rounded-md p-2 hover:bg-[var(--bg-inset)] hover:text-[var(--fg)]">
              <ExternalLink size={16} />
            </a>
          )}
        </div>
      </div>

      <Link to={`/projects/${project.slug}`} className="block">
        <h3 className="flex items-center gap-1.5 text-lg font-semibold tracking-tight">
          {project.title}
          <ArrowUpRight size={16} className="text-[var(--fg-subtle)] transition-transform group-hover:-translate-y-0.5 group-hover:translate-x-0.5 group-hover:text-[var(--fg)]" />
        </h3>
      </Link>
      <p className="mt-2 flex-1 text-[15px] leading-relaxed text-[var(--fg-muted)]">{project.description}</p>

      <div className="mt-5 flex flex-wrap gap-1.5">
        {project.tags.map((tag) => (
          <span key={tag} className="chip" style={{ ['--c' as string]: toneFor(tag) }}>
            {tag}
          </span>
        ))}
      </div>
    </article>
  )
}

export function Projects() {
  const [activeTag, setActiveTag] = useState<string | null>(null)
  const allTags = useMemo(() => Array.from(new Set(PROJECTS.flatMap((p) => p.tags))), [])
  const filtered = useMemo(
    () => (activeTag ? PROJECTS.filter((p) => p.tags.includes(activeTag)) : PROJECTS),
    [activeTag],
  )

  return (
    <section id="projects" className="px-4 py-20 sm:px-6 md:py-28">
      <div className="mx-auto max-w-7xl">
        <div className="mb-8 flex flex-wrap items-end justify-between gap-4">
          <div>
            <p className="eyebrow mb-2">// projects</p>
            <h2 className="text-3xl font-semibold tracking-tight sm:text-4xl">Things I've built</h2>
          </div>
          <span className="font-mono text-sm text-[var(--fg-subtle)]">{filtered.length}/{PROJECTS.length} shown</span>
        </div>

        <div className="-mx-4 mb-6 overflow-x-auto px-4 sm:mx-0 sm:px-0" data-lenis-prevent>
          <div className="flex w-max flex-wrap gap-1.5 sm:w-auto">
            {[null, ...allTags].map((tag) => {
              const active = activeTag === tag
              return (
                <button
                  key={tag ?? 'all'}
                  onClick={() => setActiveTag(active ? null : tag)}
                  className={`whitespace-nowrap rounded-md border px-3 py-1.5 font-mono text-xs transition-colors ${
                    active
                      ? 'border-[var(--fg)] bg-[var(--fg)] text-[var(--bg)]'
                      : 'border-[var(--border)] text-[var(--fg-muted)] hover:border-[var(--border-strong)] hover:text-[var(--fg)]'
                  }`}
                >
                  {tag ?? 'all'}
                </button>
              )
            })}
          </div>
        </div>

        <motion.div layout className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <AnimatePresence mode="popLayout">
            {filtered.map((project, i) => (
              <motion.div
                key={project.id}
                layout
                initial={{ opacity: 0, y: 16 }}
                whileInView={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, transition: { duration: 0.2 } }}
                viewport={{ once: true, margin: '-40px' }}
                transition={{ duration: 0.5, delay: (i % 2) * 0.06, ease: EASE_OUT }}
              >
                <ProjectCard project={project} index={PROJECTS.indexOf(project)} />
              </motion.div>
            ))}
          </AnimatePresence>
        </motion.div>
      </div>
    </section>
  )
}
