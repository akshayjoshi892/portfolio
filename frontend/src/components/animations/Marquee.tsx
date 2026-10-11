import { useRef, type ReactNode } from 'react'
import {
  motion,
  useAnimationFrame,
  useMotionValue,
  useScroll,
  useSpring,
  useTransform,
  useVelocity,
  wrap,
} from 'framer-motion'
import { prefersReducedMotion } from '@/lib/motion'

interface MarqueeProps {
  children: ReactNode
  /** Base speed in % of one copy per second. Negative runs right. */
  speed?: number
  className?: string
}

/**
 * Infinite ticker that speeds up with scroll velocity and flips direction
 * when the page scrolls the other way.
 */
export function Marquee({ children, speed = 2.5, className }: MarqueeProps) {
  const baseX = useMotionValue(0)
  const { scrollY } = useScroll()
  const velocity = useSpring(useVelocity(scrollY), { damping: 50, stiffness: 400 })
  const factor = useTransform(velocity, [-1500, 0, 1500], [-4, 0, 4], { clamp: false })
  const direction = useRef(1)
  const reduced = useRef(prefersReducedMotion())

  const x = useTransform(baseX, (v) => `${wrap(-25, 0, v)}%`)

  useAnimationFrame((_, delta) => {
    if (reduced.current) return
    const f = factor.get()
    if (f < 0) direction.current = -1
    else if (f > 0) direction.current = 1
    const move = direction.current * speed * (delta / 1000) * (1 + Math.abs(f))
    baseX.set(baseX.get() - move)
  })

  return (
    <div className={`flex overflow-hidden whitespace-nowrap ${className ?? ''}`}>
      <motion.div className="flex shrink-0 flex-nowrap" style={{ x }}>
        {[0, 1, 2, 3].map((i) => (
          <div key={i} aria-hidden={i > 0} className="flex shrink-0 items-center">
            {children}
          </div>
        ))}
      </motion.div>
    </div>
  )
}
