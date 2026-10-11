import { ArrowUp } from "lucide-react";
import { Marquee } from "@/components/animations/Marquee";
import { useSmoothScroll } from "@/components/providers/SmoothScroll";
import { useNavigateTo } from "@/hooks/useNavigateTo";
import { useLocalTime } from "@/hooks/useLocalTime";
import { RollText } from "@/components/ui/RollText";
import { NAV_ITEMS, SOCIALS } from "./nav";

export function Footer() {
  const { scrollTo } = useSmoothScroll();
  const go = useNavigateTo();
  const time = useLocalTime();

  return (
    <footer className="relative z-[2] overflow-hidden border-t border-[var(--border)] bg-[color-mix(in_oklab,var(--bg)_70%,transparent)] backdrop-blur-sm">
      <div className="mx-auto grid max-w-7xl gap-12 px-6 pb-10 pt-20 sm:grid-cols-2 lg:grid-cols-4">
        <div className="lg:col-span-2">
          <p className="eyebrow mb-4">Currently</p>
          <p className="max-w-sm text-2xl font-medium leading-snug tracking-tight">
            Building AI-native products and{" "}
            <span className="serif-italic text-[var(--accent)]">obsessing</span>{" "}
            over the details.
          </p>
        </div>
        <div>
          <p className="eyebrow mb-4">Menu</p>
          <ul className="space-y-2 text-sm">
            {NAV_ITEMS.map((item) => (
              <li key={item.href}>
                <a
                  href={item.href}
                  onClick={(e) => {
                    e.preventDefault();
                    go(item.href);
                  }}
                  className="group text-[var(--fg-muted)] transition-colors hover:text-[var(--fg)]"
                >
                  <RollText>{item.label}</RollText>
                </a>
              </li>
            ))}
          </ul>
        </div>
        <div>
          <p className="eyebrow mb-4">Elsewhere</p>
          <ul className="space-y-2 text-sm">
            {SOCIALS.map((s) => (
              <li key={s.label}>
                <a
                  href={s.href}
                  target="_blank"
                  rel="noreferrer"
                  className="group text-[var(--fg-muted)] transition-colors hover:text-[var(--fg)]"
                >
                  <RollText>{s.label}</RollText>
                </a>
              </li>
            ))}
          </ul>
        </div>
      </div>

      <Marquee speed={1.2} className="select-none py-2">
        <span className="px-6 text-[clamp(4rem,15vw,13rem)] font-semibold leading-none tracking-[-0.05em] text-[var(--fg)]">
          Akshay Joshi
        </span>
        <span className="serif-italic px-6 text-[clamp(4rem,15vw,13rem)] leading-none text-[var(--accent)]">
          ✦
        </span>
      </Marquee>

      <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-4 px-6 pb-8 pt-8 font-mono text-xs text-[var(--fg-muted)]">
        <span>© {new Date().getFullYear()} — All rights reserved</span>
        <span>Local time {time} IST</span>
        <button
          onClick={() => scrollTo(0)}
          className="group flex items-center gap-2 uppercase tracking-widest transition-colors hover:text-[var(--fg)]"
        >
          Back to top
          <span className="grid h-8 w-8 place-items-center rounded-full border border-[var(--border-strong)] transition-transform duration-500 group-hover:-translate-y-1">
            <ArrowUp size={13} />
          </span>
        </button>
      </div>
    </footer>
  );
}
