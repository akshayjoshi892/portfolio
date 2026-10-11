import { useRef, useState } from "react";
import { createPortal } from "react-dom";
import { Link, useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { Helmet } from "react-helmet-async";
import { AnimatePresence, LayoutGroup, motion } from "framer-motion";
import {
  ArrowLeft,
  ArrowRight,
  ArrowUpRight,
  BookOpen,
  Clock,
  FolderOpen,
  PanelLeft,
  X,
} from "lucide-react";
import { MarkdownView } from "@/components/study/MarkdownView";
import { StudySidebar } from "@/components/study/StudySidebar";
import { TableOfContents } from "@/components/study/TableOfContents";
import { SplitText } from "@/components/animations/SplitText";
import { Reveal } from "@/components/animations/Reveal";
import {
  countDocs,
  loadDoc,
  parseDoc,
  study,
  type StudyDoc,
} from "@/lib/study";
import { EASE_OUT } from "@/lib/motion";

/* ---------- Doc reader ---------- */

function DocReader({ doc }: { doc: StudyDoc }) {
  const articleRef = useRef<HTMLElement>(null);
  const {
    data: raw,
    isLoading,
    isError,
  } = useQuery({
    queryKey: ["study-doc", doc.file],
    queryFn: () => loadDoc(doc),
    staleTime: Infinity,
  });
  const parsed = raw !== undefined ? parseDoc(raw) : null;
  const title = parsed?.title ?? doc.title;

  const index = study.ordered.findIndex((d) => d.slug === doc.slug);
  const prev = study.ordered[index - 1];
  const next = study.ordered[index + 1];

  return (
    <div className="flex gap-12">
      <article ref={articleRef} className="min-w-0 max-w-[46rem] flex-1">
        <Helmet>
          <title>{title} — Study notes</title>
          {parsed?.meta.description && (
            <meta name="description" content={parsed.meta.description} />
          )}
        </Helmet>

        <nav
          aria-label="Breadcrumb"
          className="mb-6 flex flex-wrap items-center gap-1.5 font-mono text-[11px] uppercase tracking-wider text-[var(--fg-subtle)]"
        >
          <Link
            to="/study"
            className="transition-colors hover:text-[var(--fg)]"
          >
            Study
          </Link>
          {doc.trail.map((t, i) => (
            <span key={i} className="flex items-center gap-1.5">
              <span>/</span>
              <span>{t}</span>
            </span>
          ))}
        </nav>

        <motion.header
          key={doc.slug}
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, ease: EASE_OUT }}
          className="mb-10 border-b border-[var(--border)] pb-8"
        >
          <h1 className="text-[clamp(2.1rem,5vw,3.4rem)] font-medium leading-[1.02] tracking-[-0.04em]">
            {title}
          </h1>
          {parsed?.meta.description && (
            <p className="mt-4 text-lg text-[var(--fg-muted)]">
              {parsed.meta.description}
            </p>
          )}
          {parsed && (
            <p className="mt-5 flex items-center gap-4 font-mono text-xs text-[var(--fg-muted)]">
              <span className="flex items-center gap-1.5">
                <Clock size={12} /> {parsed.minutes} min read
              </span>
              <span className="truncate text-[var(--fg-subtle)]">
                {doc.file}
              </span>
            </p>
          )}
        </motion.header>

        {isLoading && (
          <div className="animate-pulse space-y-4">
            {[92, 100, 84, 96, 60].map((w, i) => (
              <div
                key={i}
                className="h-4 rounded bg-[var(--bg-inset)]"
                style={{ width: `${w}%` }}
              />
            ))}
          </div>
        )}
        {isError && (
          <p className="text-[var(--fg-muted)]">Couldn’t load this document.</p>
        )}

        {parsed && (
          <motion.div
            key={doc.slug + "-body"}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.08, ease: EASE_OUT }}
          >
            {parsed.body.trim() ? (
              <MarkdownView source={parsed.body} file={doc.file} />
            ) : (
              <p className="text-[var(--fg-muted)]">This page is empty.</p>
            )}
          </motion.div>
        )}

        {(prev || next) && (
          <nav
            aria-label="Previous and next"
            className="mt-20 grid gap-3 border-t border-[var(--border)] pt-8 sm:grid-cols-2"
          >
            {prev ? (
              <Link
                to={`/study/${prev.slug}`}
                className="group rounded-2xl border border-[var(--border)] p-5 transition-colors hover:border-[var(--border-strong)] hover:bg-[var(--bg-elevated)]"
              >
                <span className="eyebrow flex items-center gap-1.5">
                  <ArrowLeft
                    size={12}
                    className="transition-transform duration-500 group-hover:-translate-x-1"
                  />{" "}
                  Previous
                </span>
                <span className="mt-2 block font-medium">{prev.title}</span>
              </Link>
            ) : (
              <span />
            )}
            {next && (
              <Link
                to={`/study/${next.slug}`}
                className="group rounded-2xl border border-[var(--border)] p-5 text-right transition-colors hover:border-[var(--border-strong)] hover:bg-[var(--bg-elevated)]"
              >
                <span className="eyebrow flex items-center justify-end gap-1.5">
                  Next{" "}
                  <ArrowRight
                    size={12}
                    className="transition-transform duration-500 group-hover:translate-x-1"
                  />
                </span>
                <span className="mt-2 block font-medium">{next.title}</span>
              </Link>
            )}
          </nav>
        )}
      </article>

      <aside
        className="sticky top-28 hidden h-fit max-h-[calc(100vh-8rem)] w-56 shrink-0 overflow-y-auto xl:block"
        data-lenis-prevent
      >
        <TableOfContents articleRef={articleRef} deps={parsed?.body} />
      </aside>
    </div>
  );
}

/* ---------- Landing ---------- */

function StudyHome() {
  const total = study.ordered.length;
  const rootIndex = study.bySlug.get("index");
  const sections = study.tree.filter((n) => n.kind === "folder");
  const looseDocs = study.tree.filter(
    (n) => n.kind === "doc" && n.slug !== "index",
  ) as StudyDoc[];

  return (
    <div>
      <p className="eyebrow mb-5">
        Library — {total} document{total === 1 ? "" : "s"}
      </p>
      <h1 className="text-[clamp(2.8rem,8vw,6rem)] font-medium leading-[0.92] tracking-[-0.05em]">
        <SplitText text="Study" />{" "}
        <span className="serif-italic font-normal text-[var(--accent)]">
          <SplitText text="notes." delay={0.2} />
        </span>
      </h1>
      <Reveal delay={0.3}>
        <p className="mt-6 max-w-lg text-lg text-[var(--fg-muted)]">
          Everything I’ve written down while learning. Browse by section, or
          press{" "}
          <kbd className="rounded border border-[var(--border-strong)] px-1.5 font-mono text-sm">
            /
          </kbd>{" "}
          to search every page.
        </p>
      </Reveal>

      {rootIndex && (
        <Link
          to={`/study/${rootIndex.slug}`}
          className="group mt-10 inline-flex items-center gap-2 rounded-full bg-[var(--fg)] px-5 py-3 text-sm font-medium text-[var(--bg)]"
        >
          <BookOpen size={15} /> Start with the overview
          <ArrowUpRight
            size={15}
            className="transition-transform duration-500 group-hover:rotate-45"
          />
        </Link>
      )}

      {sections.length > 0 && (
        <div className="mt-14 grid gap-4 sm:grid-cols-2">
          {sections.map((section, i) => {
            if (section.kind !== "folder") return null;
            const first =
              section.index ??
              study.ordered.find((d) => d.slug.startsWith(section.slug + "/"));
            const preview = section.children.slice(0, 4);
            return (
              <motion.div
                key={section.slug}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, margin: "-40px" }}
                transition={{
                  duration: 0.8,
                  delay: (i % 2) * 0.08,
                  ease: EASE_OUT,
                }}
              >
                <Link
                  to={first ? `/study/${first.slug}` : "/study"}
                  className="group flex h-full flex-col rounded-3xl border border-[var(--border)] bg-[var(--bg-elevated)] p-6 transition-colors hover:border-[var(--border-strong)]"
                >
                  <div className="flex items-start justify-between">
                    <span className="font-mono text-xs text-[var(--fg-subtle)]">
                      {String(i + 1).padStart(2, "0")}
                    </span>
                    <span className="grid h-9 w-9 place-items-center rounded-full border border-[var(--border-strong)] transition-all duration-500 group-hover:rotate-45 group-hover:border-transparent group-hover:bg-[var(--accent)] group-hover:text-[var(--accent-fg)]">
                      <ArrowUpRight size={15} />
                    </span>
                  </div>
                  <h2 className="mt-6 text-2xl font-medium tracking-tight">
                    {section.title}
                  </h2>
                  <p className="mt-1 text-sm text-[var(--fg-muted)]">
                    {countDocs(section)} document
                    {countDocs(section) === 1 ? "" : "s"}
                  </p>
                  <ul className="mt-5 space-y-1.5 border-t border-[var(--border)] pt-4 text-sm text-[var(--fg-muted)]">
                    {preview.map((c) => (
                      <li
                        key={c.slug + c.kind}
                        className="flex items-center gap-2 truncate"
                      >
                        {c.kind === "folder" ? (
                          <FolderOpen size={13} />
                        ) : (
                          <span className="h-px w-2.5 bg-[var(--accent)]" />
                        )}
                        {c.title}
                      </li>
                    ))}
                    {section.children.length > preview.length && (
                      <li className="text-xs text-[var(--fg-subtle)]">
                        + {section.children.length - preview.length} more
                      </li>
                    )}
                  </ul>
                </Link>
              </motion.div>
            );
          })}
        </div>
      )}

      {looseDocs.length > 0 && (
        <div className="mt-14">
          <p className="eyebrow mb-4">
            {sections.length ? "Other notes" : "Notes"}
          </p>
          <ul className="divide-y divide-[var(--border)] border-y border-[var(--border)]">
            {looseDocs.map((d) => (
              <li key={d.slug}>
                <Link
                  to={`/study/${d.slug}`}
                  className="group flex items-center justify-between py-4"
                >
                  <span className="text-lg transition-transform duration-500 group-hover:translate-x-1.5">
                    {d.title}
                  </span>
                  <ArrowRight
                    size={16}
                    className="text-[var(--fg-subtle)] transition-colors group-hover:text-[var(--accent)]"
                  />
                </Link>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

function EmptyLibrary() {
  return (
    <div className="mx-auto max-w-2xl px-6 py-24">
      <p className="eyebrow mb-4">Study notes</p>
      <h1 className="text-5xl font-medium tracking-tight">No notes yet.</h1>
      <p className="mt-5 text-lg text-[var(--fg-muted)]">
        Drop Markdown files into{" "}
        <code className="rounded bg-[var(--bg-inset)] px-1.5 py-0.5 font-mono text-base">
          frontend/content/study/
        </code>{" "}
        — sub-folders become sections — and they’ll show up here automatically.
      </p>
    </div>
  );
}

/* ---------- Page ---------- */

export default function StudyMaterials() {
  const params = useParams<{ "*": string }>();
  const slug = params["*"]?.replace(/\/$/, "") || undefined;
  const doc = slug ? study.bySlug.get(slug) : undefined;
  const [drawer, setDrawer] = useState(false);

  if (study.ordered.length === 0) return <EmptyLibrary />;

  return (
    <div className="mx-auto max-w-[88rem] px-4 pb-24 pt-8 sm:px-6">
      <Helmet>
        <title>Study notes — Akshay Joshi</title>
        <meta
          name="description"
          content="Notes and study materials on engineering and AI."
        />
      </Helmet>

      {/* Mobile: open the library drawer */}
      <div className="sticky top-20 z-30 -mx-4 mb-6 px-4 lg:hidden">
        <button
          onClick={() => setDrawer(true)}
          className="glass flex w-full items-center gap-3 rounded-full border border-[var(--border)] px-4 py-2.5 text-sm"
        >
          <PanelLeft size={15} className="shrink-0 text-[var(--fg-muted)]" />
          <span className="truncate">{doc ? doc.title : "Browse notes"}</span>
          <span className="ml-auto font-mono text-[11px] text-[var(--fg-subtle)]">
            {study.ordered.length}
          </span>
        </button>
      </div>

      <div className="flex gap-10">
        <aside className="sticky top-24 hidden h-[calc(100vh-7rem)] w-64 shrink-0 lg:block">
          <LayoutGroup id="study-desktop">
            <StudySidebar active={doc?.slug} />
          </LayoutGroup>
        </aside>

        <div className="min-w-0 flex-1 lg:pt-2">
          {slug && !doc ? (
            <div className="py-20">
              <p className="eyebrow mb-3">Not found</p>
              <h1 className="text-4xl font-medium tracking-tight">
                This note doesn’t exist.
              </h1>
              <Link
                to="/study"
                className="mt-6 inline-flex items-center gap-2 text-sm text-[var(--accent)]"
              >
                <ArrowLeft size={14} /> Back to all notes
              </Link>
            </div>
          ) : doc ? (
            <DocReader doc={doc} />
          ) : (
            <StudyHome />
          )}
        </div>
      </div>

      {createPortal(
        <AnimatePresence>
          {drawer && (
            <>
              <motion.div
                key="scrim"
                className="fixed inset-0 z-[75] bg-black/50 backdrop-blur-sm lg:hidden"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                onClick={() => setDrawer(false)}
              />
              <motion.div
                key="drawer"
                role="dialog"
                aria-label="Study notes"
                className="fixed inset-y-0 left-0 z-[76] flex w-[min(20rem,86vw)] flex-col border-r border-[var(--border)] bg-[var(--bg)] p-4 pt-5 lg:hidden"
                initial={{ x: "-100%" }}
                animate={{ x: 0 }}
                exit={{ x: "-100%" }}
                transition={{ duration: 0.5, ease: EASE_OUT }}
              >
                <div className="mb-4 flex items-center justify-between">
                  <p className="eyebrow">Library</p>
                  <button
                    onClick={() => setDrawer(false)}
                    aria-label="Close library"
                    className="grid h-8 w-8 place-items-center rounded-full border border-[var(--border)]"
                  >
                    <X size={14} />
                  </button>
                </div>
                <div className="min-h-0 flex-1">
                  <LayoutGroup id="study-drawer">
                    <StudySidebar
                      active={doc?.slug}
                      onNavigate={() => setDrawer(false)}
                    />
                  </LayoutGroup>
                </div>
              </motion.div>
            </>
          )}
        </AnimatePresence>,
        document.body,
      )}
    </div>
  );
}
