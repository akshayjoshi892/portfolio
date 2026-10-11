import { motion } from "framer-motion";
import { Download, Briefcase, GraduationCap, Sparkles } from "lucide-react";
import { SKILL_GROUPS } from "@/data/sample";
import { SKILL_ICON_MAP } from "@/data/skillIcons";
import { EASE_OUT } from "@/lib/motion";
import { toneAt } from "@/lib/tones";

const STATS = [
  { value: "3+", label: "years shipping production software" },
  { value: "300+", label: "developers on my commit tracking system" },
  { value: "40%", label: "API latency cut with a Redis cache" },
  { value: "120s", label: "device boot time saved via kernel work" },
];

const NOW = [
  { icon: Briefcase, text: "Building agentic chat UIs & AI risk tooling at Boeing" },
  { icon: Sparkles, text: "Going deep on LLM agents, RAG and MCP" },
  { icon: GraduationCap, text: "Practising DSA daily on LeetCode" },
];

const rise = (delay = 0) => ({
  initial: { opacity: 0, y: 16 },
  whileInView: { opacity: 1, y: 0 },
  viewport: { once: true, margin: "-40px" },
  transition: { duration: 0.5, delay, ease: EASE_OUT },
});

export function About() {
  return (
    <section id="about" className="px-4 py-20 sm:px-6 md:py-28">
      <div className="mx-auto max-w-7xl">
        <div className="mb-10">
          <p className="eyebrow mb-2">// about</p>
          <h2 className="text-3xl font-semibold tracking-tight sm:text-4xl">
            Engineer who cares about the{" "}
            <span className="serif-italic text-[var(--accent)]">details</span>
          </h2>
        </div>

        <div className="grid gap-4 lg:grid-cols-12">
          <motion.div
            {...rise()}
            className="rounded-xl border border-[var(--border)] bg-[var(--bg-elevated)] p-6 lg:col-span-7"
          >
            <p className="text-[17px] leading-relaxed text-[var(--fg-muted)]">
              I'm a software engineer at{" "}
              <span className="text-[var(--fg)]">Boeing</span> with 3+ years of
              experience building scalable full-stack applications and, more
              recently, <span className="text-[var(--fg)]">agentic AI products</span>.
            </p>
            <p className="mt-4 text-[17px] leading-relaxed text-[var(--fg-muted)]">
              I've shipped everything from Linux kernel optimizations to
              real-time dashboards and pluggable chat engines. I care about
              performance, secure system design and interfaces that feel fast.
            </p>
            <a
              href="/resume.pdf"
              download
              className="mt-6 inline-flex items-center gap-2 rounded-lg border border-[var(--border-strong)] px-4 py-2.5 text-sm font-medium transition-colors hover:bg-[var(--bg-inset)]"
            >
              <Download size={15} /> Download résumé
            </a>
          </motion.div>

          <motion.div
            {...rise(0.06)}
            className="rounded-xl border border-[var(--border)] bg-[var(--bg-elevated)] p-6 lg:col-span-5"
          >
            <p className="mb-4 font-mono text-xs text-[var(--fg-subtle)]">$ cat now.md</p>
            <ul className="space-y-3.5">
              {NOW.map(({ icon: Icon, text }, i) => (
                <li key={text} className="flex items-start gap-3 text-[15px]">
                  <span
                    className="grid h-8 w-8 shrink-0 place-items-center rounded-lg"
                    style={{ color: toneAt(i + 1), background: `color-mix(in oklab, ${toneAt(i + 1)} 13%, transparent)` }}
                  >
                    <Icon size={15} />
                  </span>
                  <span className="pt-1.5 text-[var(--fg-muted)]">{text}</span>
                </li>
              ))}
            </ul>
          </motion.div>
        </div>

        <dl className="mt-4 grid grid-cols-2 gap-4 lg:grid-cols-4">
          {STATS.map((s, i) => (
            <motion.div
              key={s.label}
              {...rise(0.04 * i)}
              className="relative flex flex-col overflow-hidden rounded-xl border border-[var(--border)] bg-[var(--bg-elevated)] p-5"
              style={{ color: toneAt(i) }}
            >
              <span aria-hidden className="absolute inset-x-0 top-0 h-[3px] bg-current opacity-80" />
              <dt className="order-2 mt-2 text-sm text-[var(--fg-muted)]">{s.label}</dt>
              <dd className="font-mono text-3xl font-medium tracking-tight sm:text-4xl">{s.value}</dd>
            </motion.div>
          ))}
        </dl>

        <div className="mt-14">
          <p className="eyebrow mb-5">// toolkit</p>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {SKILL_GROUPS.map((group, gi) => {
              const tone = toneAt(gi);
              return (
                <motion.div
                  key={group.category}
                  {...rise(0.05 * gi)}
                  className="rounded-xl border border-[var(--border)] bg-[var(--bg-elevated)] p-5"
                >
                  <h3 className="mb-3 flex items-center gap-2 font-mono text-sm font-medium" style={{ color: tone }}>
                    <span className="h-2 w-2 rounded-full bg-current" />
                    {group.label}
                  </h3>
                  <ul className="flex flex-wrap gap-1.5">
                    {group.skills.map((skill) => {
                      const Icon = SKILL_ICON_MAP[skill.name]?.icon;
                      return (
                        <li key={skill.name} className="chip" style={{ ["--c" as string]: tone }}>
                          {Icon && <Icon size={12} />}
                          {skill.name}
                        </li>
                      );
                    })}
                  </ul>
                </motion.div>
              );
            })}
          </div>
        </div>
      </div>
    </section>
  );
}
