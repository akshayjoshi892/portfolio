import { useRef, type ReactNode, type MouseEvent } from 'react'
import { motion, useMotionValue, useSpring } from 'framer-motion'

interface MagneticButtonProps {
  children: ReactNode
  className?: string
  onClick?: () => void
  strength?: number
  type?: 'button' | 'submit' | 'reset'
  disabled?: boolean
  href?: string
  'aria-label'?: string
}

const SPRING = { stiffness: 180, damping: 14, mass: 0.35 }

/** Pulls toward the pointer; the label inside drifts further for a layered feel. */
export function MagneticButton({
  children,
  className,
  onClick,
  strength = 0.35,
  type = 'button',
  disabled,
  href,
  'aria-label': ariaLabel,
}: MagneticButtonProps) {
  const ref = useRef<HTMLElement>(null)
  const x = useMotionValue(0)
  const y = useMotionValue(0)
  const sx = useSpring(x, SPRING)
  const sy = useSpring(y, SPRING)
  const ix = useSpring(useMotionValue(0), SPRING)
  const iy = useSpring(useMotionValue(0), SPRING)

  function handleMouseMove(e: MouseEvent<HTMLElement>) {
    const rect = ref.current?.getBoundingClientRect()
    if (!rect) return
    const relX = e.clientX - rect.left - rect.width / 2
    const relY = e.clientY - rect.top - rect.height / 2
    x.set(relX * strength)
    y.set(relY * strength)
    ix.set(relX * strength * 0.45)
    iy.set(relY * strength * 0.45)
  }

  function reset() {
    x.set(0)
    y.set(0)
    ix.set(0)
    iy.set(0)
  }

  const inner = (
    <motion.span className="relative z-10 flex items-center justify-center gap-2" style={{ x: ix, y: iy }}>
      {children}
    </motion.span>
  )

  const shared = {
    onMouseMove: handleMouseMove,
    onMouseLeave: reset,
    style: { x: sx, y: sy },
    whileTap: { scale: 0.95 },
    className,
    'aria-label': ariaLabel,
  }

  if (href) {
    const external = href.startsWith('http')
    return (
      <motion.a
        ref={ref as React.Ref<HTMLAnchorElement>}
        href={href}
        target={external ? '_blank' : undefined}
        rel={external ? 'noreferrer' : undefined}
        {...shared}
      >
        {inner}
      </motion.a>
    )
  }

  return (
    <motion.button ref={ref as React.Ref<HTMLButtonElement>} type={type} disabled={disabled} onClick={onClick} {...shared}>
      {inner}
    </motion.button>
  )
}
