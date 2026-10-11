import { useEffect, useState } from "react";
import { AnimatePresence, motion, animate } from "framer-motion";
import { EASE_IN_OUT, EASE_OUT, prefersReducedMotion } from "@/lib/motion";

const WORDS = ["Design", "Engineer", "Ship"];

function shouldShow() {
  try {
    // Only greet visitors landing on the home page itself, never deep links.
    const landingHome =
      window.location.pathname === "/" && !window.location.hash;
    return (
      landingHome &&
      !sessionStorage.getItem("intro-seen") &&
      !prefersReducedMotion()
    );
  } catch {
    return false;
  }
}

/** First-visit intro: a counter runs to 100 while words cycle, then the curtain lifts. */
export function Preloader({ onDone }: { onDone: () => void }) {
  const [visible, setVisible] = useState(shouldShow);
  const [count, setCount] = useState(0);

  useEffect(() => {
    if (!visible) {
      onDone();
      return;
    }
    document.documentElement.style.overflow = "hidden";
    const controls = animate(0, 100, {
      duration: 1.9,
      ease: [0.65, 0, 0.35, 1],
      onUpdate: (v) => setCount(Math.round(v)),
      onComplete: () => {
        try {
          sessionStorage.setItem("intro-seen", "1");
        } catch {
          /* private mode */
        }
        setTimeout(() => setVisible(false), 250);
      },
    });
    return () => controls.stop();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const word =
    WORDS[Math.min(WORDS.length - 1, Math.floor((count / 100) * WORDS.length))];

  return (
    <AnimatePresence
      onExitComplete={() => {
        document.documentElement.style.overflow = "";
        onDone();
      }}
    >
      {visible && (
        <motion.div
          key="preloader"
          className="fixed inset-0 z-[200] flex flex-col justify-between bg-[#0b0b0a] p-6 text-[#ecebe6] sm:p-10"
          exit={{ clipPath: "inset(0 0 100% 0)" }}
          initial={{ clipPath: "inset(0 0 0% 0)" }}
          transition={{ duration: 0.9, ease: EASE_IN_OUT }}
        >
          <div className="flex items-center justify-between font-mono text-xs uppercase tracking-[0.2em] text-[#9a9890]">
            <span>Akshay Joshi</span>
            <span>Portfolio ©{new Date().getFullYear()}</span>
          </div>

          <div className="overflow-hidden">
            <AnimatePresence mode="wait">
              <motion.p
                key={word}
                initial={{ y: "110%" }}
                animate={{ y: 0 }}
                exit={{ y: "-110%" }}
                transition={{ duration: 0.45, ease: EASE_OUT }}
                className="serif-italic text-5xl sm:text-7xl"
              >
                {word}.
              </motion.p>
            </AnimatePresence>
          </div>

          <div className="flex items-end justify-between gap-6">
            <div className="h-px flex-1 bg-[#242421]">
              <div
                className="h-px bg-[#d4ff3f]"
                style={{ width: `${count}%` }}
              />
            </div>
            <span className="font-sans text-[clamp(4rem,14vw,11rem)] font-medium leading-[0.8] tracking-tighter tabular-nums">
              {count}
            </span>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
