/** Text that rolls up to reveal a copy of itself on hover of the nearest `.group`. */
export function RollText({ children, className }: { children: string; className?: string }) {
  return (
    <span className={`relative inline-flex overflow-hidden ${className ?? ''}`}>
      <span className="block transition-transform duration-500 ease-[var(--ease-out-expo)] group-hover:-translate-y-full">
        {children}
      </span>
      <span
        aria-hidden
        className="absolute inset-0 block translate-y-full transition-transform duration-500 ease-[var(--ease-out-expo)] group-hover:translate-y-0"
      >
        {children}
      </span>
    </span>
  )
}
