import { useEffect, useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { AnimatePresence, motion, useMotionValueEvent, useScroll } from "framer-motion";
import { Menu, Moon, Sun, X } from "lucide-react";
import { useTheme } from "@/context/ThemeContext";
import { useNavigateTo } from "@/hooks/useNavigateTo";
import { EASE_OUT } from "@/lib/motion";
import { toneAt } from "@/lib/tones";
import { NAV_ITEMS, SOCIALS } from "./nav";

/** Which home-page section is in view, for the active indicator. */
function useActiveSection(enabled: boolean) {
  const [active, setActive] = useState<string | null>(null);
  useEffect(() => {
    if (!enabled) return;
    const ids = NAV_ITEMS.filter((i) => i.href.startsWith("/#")).map((i) => i.href.slice(2));
    const els = ids.map((id) => document.getElementById(id)).filter(Boolean) as HTMLElement[];
    const io = new IntersectionObserver(
      (entries) => {
        for (const e of entries) if (e.isIntersecting) setActive(`/#${e.target.id}`);
      },
      { rootMargin: "-45% 0px -50% 0px" },
    );
    els.forEach((el) => io.observe(el));
    return () => io.disconnect();
  }, [enabled]);
  return enabled ? active : null;
}

export function Header() {
  const { theme, toggleTheme } = useTheme();
  const go = useNavigateTo();
  const { pathname } = useLocation();
  const [open, setOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const section = useActiveSection(pathname === "/");

  const { scrollY } = useScroll();
  useMotionValueEvent(scrollY, "change", (y) => setScrolled(y > 24));

  // Close the menu on route change (render-time adjustment, no effect needed).
  const [lastPath, setLastPath] = useState(pathname);
  if (pathname !== lastPath) {
    setLastPath(pathname);
    setOpen(false);
  }

  function handle(href: string) {
    setOpen(false);
    go(href);
  }

  const isActive = (href: string) =>
    href.startsWith("/#") ? section === href : pathname.startsWith(href);

  return (
    <header className="fixed inset-x-0 top-0 z-[60] flex justify-center px-4 pt-3 sm:pt-4">
      <div
        className={`relative flex w-full max-w-3xl items-center justify-between gap-2 rounded-xl border px-2 py-1.5 transition-all duration-300 ${
          scrolled || open
            ? "glass border-[var(--border)] shadow-[0_10px_30px_-15px_rgba(0,0,0,0.35)]"
            : "border-transparent"
        }`}
      >
        <Link
          to="/"
          aria-label="Akshay Joshi — home"
          className="rounded-lg px-2.5 py-1.5 font-mono text-sm font-medium transition-colors hover:bg-[var(--bg-inset)]"
        >
          <span className="text-[var(--accent)]">~/</span>akshay
        </Link>

        <nav aria-label="Primary" className="hidden md:block">
          <ul className="flex items-center gap-0.5">
            {NAV_ITEMS.map((item, i) => {
              const active = isActive(item.href);
              return (
                <li key={item.href}>
                  <a
                    href={item.href}
                    onClick={(e) => {
                      e.preventDefault();
                      handle(item.href);
                    }}
                    className={`relative flex items-center gap-1.5 rounded-lg px-3 py-1.5 font-mono text-[13px] transition-colors ${
                      active
                        ? "bg-[var(--bg-inset)] text-[var(--fg)]"
                        : "text-[var(--fg-muted)] hover:text-[var(--fg)]"
                    }`}
                  >
                    {active && (
                      <motion.span
                        layoutId="nav-dot"
                        className="h-1.5 w-1.5 rounded-full"
                        style={{ background: toneAt(i) }}
                      />
                    )}
                    {item.label.toLowerCase()}
                  </a>
                </li>
              );
            })}
          </ul>
        </nav>

        <div className="flex items-center gap-1">
          <button
            onClick={toggleTheme}
            aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
            className="grid h-8 w-8 place-items-center rounded-lg text-[var(--fg-muted)] transition-colors hover:bg-[var(--bg-inset)] hover:text-[var(--fg)]"
          >
            {theme === "dark" ? <Sun size={15} /> : <Moon size={15} />}
          </button>
          <a
            href="/#contact"
            onClick={(e) => {
              e.preventDefault();
              handle("/#contact");
            }}
            className="hidden rounded-lg bg-[var(--fg)] px-3 py-1.5 font-mono text-[13px] font-medium text-[var(--bg)] transition-opacity hover:opacity-90 sm:block"
          >
            contact
          </a>
          <button
            onClick={() => setOpen((o) => !o)}
            aria-label={open ? "Close menu" : "Open menu"}
            aria-expanded={open}
            className="grid h-8 w-8 place-items-center rounded-lg text-[var(--fg-muted)] hover:bg-[var(--bg-inset)] hover:text-[var(--fg)] md:hidden"
          >
            {open ? <X size={17} /> : <Menu size={17} />}
          </button>
        </div>

        <AnimatePresence>
          {open && (
            <motion.div
              initial={{ opacity: 0, y: -6 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -6 }}
              transition={{ duration: 0.2, ease: EASE_OUT }}
              className="glass absolute inset-x-0 top-[calc(100%+8px)] rounded-xl border border-[var(--border)] p-2 shadow-[0_10px_30px_-15px_rgba(0,0,0,0.35)] md:hidden"
            >
              <ul>
                {[...NAV_ITEMS, { label: "Contact", href: "/#contact" }].map((item, i) => (
                  <li key={item.href}>
                    <a
                      href={item.href}
                      onClick={(e) => {
                        e.preventDefault();
                        handle(item.href);
                      }}
                      className="flex items-center gap-2.5 rounded-lg px-3 py-2.5 font-mono text-sm hover:bg-[var(--bg-inset)]"
                    >
                      <span className="h-1.5 w-1.5 rounded-full" style={{ background: toneAt(i) }} />
                      {item.label.toLowerCase()}
                    </a>
                  </li>
                ))}
              </ul>
              <div className="mt-1 flex flex-wrap gap-x-4 gap-y-1 border-t border-[var(--border)] px-3 pb-1 pt-2.5 font-mono text-xs text-[var(--fg-muted)]">
                {SOCIALS.map((s) => (
                  <a key={s.label} href={s.href} target="_blank" rel="noreferrer" className="hover:text-[var(--fg)]">
                    {s.label.toLowerCase()}
                  </a>
                ))}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </header>
  );
}
