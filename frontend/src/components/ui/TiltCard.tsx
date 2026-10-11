import { useRef, type ReactNode, type PointerEvent } from 'react'
import { motion, useMotionValue, useSpring, useTransform } from 'framer-motion'
import { isFinePointer } from '@/lib/motion'

/** Card that tilts toward the pointer and lights its border with a spotlight. */
export function TiltCard({
  children,
  className,
  max = 7,
  ...rest
}: {
  children: ReactNode
  className?: string
  max?: number
} & Omit<React.HTMLAttributes<HTMLDivElement>, 'onPointerMove' | 'onPointerLeave'>) {
  const ref = useRef<HTMLDivElement>(null)
  const px = useMotionValue(0.5)
  const py = useMotionValue(0.5)
  const spring = { stiffness: 220, damping: 22 }
  const rotateX = useSpring(useTransform(py, [0, 1], [max, -max]), spring)
  const rotateY = useSpring(useTransform(px, [0, 1], [-max, max]), spring)
  const fine = useRef(isFinePointer())

  function onMove(e: PointerEvent<HTMLDivElement>) {
    const el = ref.current
    if (!el) return
    const r = el.getBoundingClientRect()
    const x = e.clientX - r.left
    const y = e.clientY - r.top
    el.style.setProperty('--mx', `${x}px`)
    el.style.setProperty('--my', `${y}px`)
    if (!fine.current) return
    px.set(x / r.width)
    py.set(y / r.height)
  }

  function onLeave() {
    px.set(0.5)
    py.set(0.5)
  }

  return (
    <div style={{ perspective: 1100 }} className="h-full">
      <motion.div
        ref={ref}
        onPointerMove={onMove}
        onPointerLeave={onLeave}
        style={{ rotateX, rotateY, transformStyle: 'preserve-3d' }}
        className={`spotlight ${className ?? ''}`}
        {...(rest as object)}
      >
        {children}
      </motion.div>
    </div>
  )
}
