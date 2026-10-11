import { motion } from "framer-motion";
import { ArrowUpRight, Mail, MapPin, Briefcase } from "lucide-react";
import { useSmoothScroll } from "@/components/providers/SmoothScroll";
import { useIntroDone } from "@/components/layout/IntroContext";
import { GithubIcon } from "@/components/ui/GithubIcon";
import { LinkedinIcon } from "@/components/ui/LinkedinIcon";
import { SOCIALS } from "@/components/layout/nav";
import { StatusPill } from "./StatusPill";
import { VisitorCount } from "./VisitorCount";
import { EASE_OUT } from "@/lib/motion";
import { toneAt, toneFor } from "@/lib/tones";

const FACTS = [
  { k: "role", v: "Software Engineer @ Boeing" },
  { k: "exp", v: "3+ yrs · Zebra → Boeing" },
  { k: "focus", v: "Full-stack · Agentic AI" },
];
const STACK = ["TypeScript", "React", "Node.js", "Python", "SQL", "Docker"];

const href = (label: string) => SOCIALS.find((s) => s.label === label)?.href ?? "#";

function Terminal() {
  const lines: [string, string][] = [
    ["$ whoami", ""],
    ["", "akshay — builds web apps & AI agents that ship"],
    ["$ cat now.txt", ""],
    ["", "chat UIs for agentic systems, AI risk assessment @ Boeing"],
    ["$ ls skills/", ""],
    ["", "SKILLS"],
  ];
  return (
    <div className="card-surface overflow-hidden rounded-xl font-mono text-[13px] leading-relaxed">
      <div className="flex items-center gap-1.5 border-b border-[var(--border)] px-4 py-2.5">
        <span className="h-2.5 w-2.5 rounded-full bg-[var(--tone-clay)] opacity-80" />
        <span className="h-2.5 w-2.5 rounded-full bg-[var(--tone-amber)] opacity-80" />
        <span className="h-2.5 w-2.5 rounded-full bg-[var(--tone-sage)] opacity-80" />
        <span className="ml-3 text-[11px] text-[var(--fg-subtle)]">~/akshay — zsh</span>
      </div>
      <div className="space-y-0.5 p-4 sm:p-5">
        {lines.map(([cmd, out], i) =>
          cmd ? (
            <p key={i} className={i ? "pt-2" : ""}>
              <span className="text-[var(--accent)]">➜</span>{" "}
              <span className="text-[var(--fg)]">{cmd.slice(2)}</span>
            </p>
          ) : out === "SKILLS" ? (
            <p key={i} className="flex flex-wrap gap-x-3">
              {["react/", "node/", "typescript/", "python/", "docker/", "redis/", "llm-agents/"].map((d, j) => (
                <span key={d} style={{ color: toneAt(j) }}>{d}</span>
              ))}
            </p>
          ) : (
            <p key={i} className="text-[var(--fg-muted)]">{out}</p>
          ),
        )}
        <p className="pt-2">
          <span className="text-[var(--accent)]">➜</span>{" "}
          <span className="inline-block h-4 w-2 translate-y-0.5 animate-pulse bg-[var(--fg-muted)]" />
        </p>
      </div>
    </div>
  );
}

export function Hero() {
  const ready = useIntroDone();
  const { scrollTo } = useSmoothScroll();

  const fade = (delay: number) => ({
    initial: { opacity: 0, y: 14 },
    animate: ready ? { opacity: 1, y: 0 } : undefined,
    transition: { duration: 0.7, delay, ease: EASE_OUT },
  });

  return (
    <section id="top" className="relative px-4 pb-12 pt-28 sm:px-6 sm:pt-32 md:pb-16">
      <div className="mx-auto grid w-full max-w-7xl items-start gap-10 lg:grid-cols-12 lg:gap-12">
        <div className="lg:col-span-7">
          <motion.div {...fade(0.1)} className="mb-6 flex flex-wrap items-center gap-2">
            <StatusPill />
            <VisitorCount />
          </motion.div>

          <motion.h1
            {...fade(0.15)}
            className="text-[clamp(2.6rem,7vw,4.75rem)] font-semibold leading-[1.02] tracking-[-0.04em]"
          >
            Akshay Joshi
          </motion.h1>
          <motion.p {...fade(0.2)} className="mt-3 font-mono text-base text-[var(--accent)] sm:text-lg">
            Full-stack &amp; AI Engineer
          </motion.p>

          <motion.p
            {...fade(0.25)}
            className="mt-5 max-w-xl text-pretty text-lg leading-relaxed text-[var(--fg-muted)]"
          >
            I build production web apps end to end — APIs, data models and
            React frontends — and lately, agentic AI systems that reason over
            real enterprise data.
          </motion.p>

          <motion.dl
            {...fade(0.3)}
            className="mt-8 grid max-w-xl gap-x-6 gap-y-2 font-mono text-sm sm:grid-cols-[auto_1fr]"
          >
            {FACTS.map((f, i) => (
              <div key={f.k} className="contents">
                <dt style={{ color: toneAt(i + 1) }}>{f.k}</dt>
                <dd className="mb-2 text-[var(--fg)] sm:mb-0">{f.v}</dd>
              </div>
            ))}
            <dt style={{ color: toneAt(4) }}>stack</dt>
            <dd className="flex flex-wrap gap-1.5">
              {STACK.map((t) => (
                <span key={t} className="chip" style={{ ["--c" as string]: toneFor(t) }}>{t}</span>
              ))}
            </dd>
          </motion.dl>

          <motion.div {...fade(0.35)} className="mt-9 flex flex-wrap items-center gap-3">
            <button
              onClick={() => scrollTo("#projects")}
              className="group inline-flex items-center gap-2 rounded-lg bg-[var(--fg)] px-5 py-3 text-sm font-medium text-[var(--bg)] transition-opacity hover:opacity-90"
            >
              <Briefcase size={15} /> View projects
              <ArrowUpRight size={15} className="transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5" />
            </button>
            <button
              onClick={() => scrollTo("#contact")}
              className="inline-flex items-center gap-2 rounded-lg border border-[var(--border-strong)] px-5 py-3 text-sm font-medium transition-colors hover:bg-[var(--bg-inset)]"
            >
              <Mail size={15} /> Contact
            </button>
            <div className="ml-1 flex items-center gap-1">
              <a href={href("GitHub")} target="_blank" rel="noreferrer" aria-label="GitHub" className="rounded-md p-2.5 text-[var(--fg-muted)] hover:bg-[var(--bg-inset)] hover:text-[var(--fg)]">
                <GithubIcon size={18} />
              </a>
              <a href={href("LinkedIn")} target="_blank" rel="noreferrer" aria-label="LinkedIn" className="rounded-md p-2.5 text-[var(--fg-muted)] hover:bg-[var(--bg-inset)] hover:text-[var(--fg)]">
                <LinkedinIcon size={18} />
              </a>
              <a href={href("LeetCode")} target="_blank" rel="noreferrer" aria-label="LeetCode" className="rounded-md p-2.5 font-mono text-sm font-semibold text-[var(--fg-muted)] hover:bg-[var(--bg-inset)] hover:text-[var(--fg)]">
                LC
              </a>
            </div>
          </motion.div>
        </div>

        <motion.div {...fade(0.3)} className="lg:col-span-5 lg:pt-14">
          <Terminal />
          <p className="mt-3 flex items-center gap-1.5 font-mono text-xs text-[var(--fg-subtle)]">
            <MapPin size={12} /> India · open to interesting problems
          </p>
        </motion.div>
      </div>
    </section>
  );
}
