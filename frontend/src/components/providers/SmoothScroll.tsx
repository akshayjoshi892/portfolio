import { createContext, useCallback, useContext, useEffect, useRef, type ReactNode } from 'react'
import { useLocation } from 'react-router-dom'
import Lenis from 'lenis'
import { prefersReducedMotion } from '@/lib/motion'

type ScrollTo = (target: string | number | HTMLElement, opts?: { offset?: number; immediate?: boolean }) => void

const SmoothScrollContext = createContext<{ scrollTo: ScrollTo; lenis: () => Lenis | null }>({
  scrollTo: () => {},
  lenis: () => null,
})

export function SmoothScrollProvider({ children }: { children: ReactNode }) {
  const lenisRef = useRef<Lenis | null>(null)
  const { pathname, hash } = useLocation()

  useEffect(() => {
    if (prefersReducedMotion()) return
    const lenis = new Lenis({ duration: 1.15, easing: (t) => 1 - Math.pow(1 - t, 4), wheelMultiplier: 0.95 })
    lenisRef.current = lenis
    let frame = requestAnimationFrame(function raf(time) {
      lenis.raf(time)
      frame = requestAnimationFrame(raf)
    })
    return () => {
      cancelAnimationFrame(frame)
      lenis.destroy()
      lenisRef.current = null
    }
  }, [])

  const scrollTo = useCallback<ScrollTo>((target, opts = {}) => {
    const el = typeof target === 'string' ? document.querySelector<HTMLElement>(target) : target
    if (el === null) return
    const lenis = lenisRef.current
    if (lenis) {
      lenis.scrollTo(el, { offset: opts.offset ?? 0, immediate: opts.immediate, duration: 1.4 })
    } else if (typeof el === 'number') {
      window.scrollTo({ top: el, behavior: opts.immediate ? 'auto' : 'smooth' })
    } else {
      el.scrollIntoView({ behavior: opts.immediate ? 'auto' : 'smooth' })
    }
  }, [])

  // New route: jump to top, or to the hash target once the page has mounted.
  useEffect(() => {
    if (hash) {
      // The page is lazy-loaded, so poll until the target section exists.
      let tries = 0
      let t: ReturnType<typeof setTimeout>
      const attempt = () => {
        if (document.querySelector(hash)) t = setTimeout(() => scrollTo(hash), 400)
        else if (tries++ < 40) t = setTimeout(attempt, 100)
      }
      attempt()
      return () => clearTimeout(t)
    }
    scrollTo(0, { immediate: true })
  }, [pathname, hash, scrollTo])

  return (
    <SmoothScrollContext.Provider value={{ scrollTo, lenis: () => lenisRef.current }}>
      {children}
    </SmoothScrollContext.Provider>
  )
}

export function useSmoothScroll() {
  return useContext(SmoothScrollContext)
}
