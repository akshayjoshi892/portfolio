import type { ReactNode } from 'react'
import { LineReveal } from '@/components/animations/SplitText'

export function SectionHeading({
  index,
  label,
  children,
  aside,
}: {
  index: string
  label: string
  children: ReactNode
  aside?: ReactNode
}) {
  return (
    <div className="mb-14 grid gap-6 border-t border-[var(--border)] pt-6 md:mb-20 md:grid-cols-12">
      <p className="eyebrow md:col-span-3">
        ({index}) {label}
      </p>
      <h2 className="text-[clamp(2.5rem,6.5vw,5.5rem)] font-medium leading-[0.95] tracking-[-0.045em] md:col-span-7">
        <LineReveal>{children}</LineReveal>
      </h2>
      {aside && <div className="self-end md:col-span-2 md:text-right">{aside}</div>}
    </div>
  )
}
