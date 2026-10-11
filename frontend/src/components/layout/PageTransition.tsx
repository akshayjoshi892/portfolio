import type { ReactNode } from 'react'
import { motion } from 'framer-motion'
import { EASE_IN_OUT, EASE_OUT } from '@/lib/motion'

/** Content fades up while a curtain sweeps across on route change. */
export function PageTransition({ children }: { children: ReactNode }) {
  return (
    <>
      <motion.div
        initial={{ opacity: 0, y: 24 }}
        animate={{ opacity: 1, y: 0, transition: { duration: 0.8, delay: 0.35, ease: EASE_OUT } }}
        exit={{ opacity: 0, y: -16, transition: { duration: 0.35, ease: EASE_IN_OUT } }}
      >
        {children}
      </motion.div>
      <motion.div
        aria-hidden
        className="pointer-events-none fixed inset-0 z-[90] origin-top bg-[var(--fg)]"
        initial={{ scaleY: 1 }}
        animate={{ scaleY: 0, transition: { duration: 0.7, ease: EASE_IN_OUT } }}
        exit={{ scaleY: 0 }}
        style={{ originY: 0 }}
      />
      <motion.div
        aria-hidden
        className="pointer-events-none fixed inset-0 z-[90] bg-[var(--fg)]"
        initial={{ scaleY: 0 }}
        animate={{ scaleY: 0 }}
        exit={{ scaleY: 1, transition: { duration: 0.45, ease: EASE_IN_OUT } }}
        style={{ originY: 1 }}
      />
    </>
  )
}
