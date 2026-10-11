export function PageSkeleton() {
  return (
    <div className="mx-auto max-w-5xl animate-pulse space-y-6 px-6 py-10">
      <div className="h-8 w-1/3 rounded-lg bg-[var(--bg-elevated)]" />
      <div className="h-4 w-2/3 rounded-lg bg-[var(--bg-elevated)]" />
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        {Array.from({ length: 4 }).map((_, i) => (
          <div key={i} className="h-40 rounded-2xl bg-[var(--bg-elevated)]" />
        ))}
      </div>
    </div>
  )
}

export function CardSkeleton() {
  return <div className="h-56 w-full animate-pulse rounded-2xl bg-[var(--bg-elevated)]" />
}
