import { motion } from 'framer-motion'
import { EASE_OUT } from '@/lib/motion'

interface SplitTextProps {
  text: string
  className?: string
  delay?: number
  stagger?: number
  /** Animate when mounted (`true`) or when scrolled into view. */
  play?: boolean
  inView?: boolean
}

/** Characters slide up from behind a mask, one after another. */
export function SplitText({ text, className, delay = 0, stagger = 0.035, play = true, inView = false }: SplitTextProps) {
  const words = text.split(' ')
  let index = 0

  const trigger = inView
    ? { initial: 'hidden', whileInView: 'show', viewport: { once: true, margin: '-10%' } }
    : { initial: 'hidden', animate: play ? 'show' : 'hidden' }

  return (
    <motion.span className={className} {...trigger}>
      <span className="sr-only">{text}</span>
      {words.map((word, wi) => (
        <span key={wi} aria-hidden className="inline-block whitespace-nowrap">
          {word.split('').map((char) => {
            const i = index++
            return (
              <span
                key={i}
                className="inline-block pb-[0.08em] align-bottom"
                style={{ clipPath: 'inset(-50% -40% -0.22em -40%)' }}
              >
                <motion.span
                  className="inline-block will-change-transform"
                  variants={{
                    hidden: { y: '135%', rotate: 8 },
                    show: { y: 0, rotate: 0, transition: { duration: 0.9, ease: EASE_OUT, delay: delay + i * stagger } },
                  }}
                >
                  {char}
                </motion.span>
              </span>
            )
          })}
          {wi < words.length - 1 && <span className="inline-block">&nbsp;</span>}
        </span>
      ))}
    </motion.span>
  )
}

/** Whole lines slide up from a mask — for paragraphs and headings. */
export function LineReveal({
  children,
  className,
  delay = 0,
}: {
  children: React.ReactNode
  className?: string
  delay?: number
}) {
  return (
    <span className={`block overflow-hidden pb-[0.1em] ${className ?? ''}`}>
      <motion.span
        className="block will-change-transform"
        initial={{ y: '105%' }}
        whileInView={{ y: 0 }}
        viewport={{ once: true, margin: '-8%' }}
        transition={{ duration: 1, ease: EASE_OUT, delay }}
      >
        {children}
      </motion.span>
    </span>
  )
}
