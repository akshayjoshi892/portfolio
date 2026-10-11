import { Link } from 'react-router-dom'
import { Helmet } from 'react-helmet-async'
import { ArrowLeft } from 'lucide-react'
import { SplitText } from '@/components/animations/SplitText'
import { Reveal } from '@/components/animations/Reveal'

export default function NotFound() {
  return (
    <div className="mx-auto flex min-h-[80vh] max-w-7xl flex-col justify-center px-6 pt-24">
      <Helmet>
        <title>404 — Not Found</title>
      </Helmet>
      <p className="eyebrow mb-6">Error 404</p>
      <h1 className="text-[clamp(4rem,16vw,14rem)] font-semibold leading-[0.85] tracking-[-0.06em]">
        <SplitText text="Lost in" />{' '}
        <span className="serif-italic font-normal text-[var(--accent)]">
          <SplitText text="space." delay={0.25} />
        </span>
      </h1>
      <Reveal delay={0.4} className="mt-10 flex flex-wrap items-center gap-6">
        <p className="max-w-sm text-[var(--fg-muted)]">The page you’re looking for doesn’t exist or has moved.</p>
        <Link
          to="/"
          className="group inline-flex items-center gap-2 rounded-full bg-[var(--fg)] px-6 py-3.5 text-sm font-medium text-[var(--bg)]"
        >
          <ArrowLeft size={16} className="transition-transform duration-500 group-hover:-translate-x-1" />
          Back home
        </Link>
      </Reveal>
    </div>
  )
}
