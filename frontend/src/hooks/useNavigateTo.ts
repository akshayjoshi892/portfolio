import { useCallback } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { useSmoothScroll } from '@/components/providers/SmoothScroll'

/** Handles both route links and `/#section` anchors with smooth scrolling. */
export function useNavigateTo() {
  const navigate = useNavigate()
  const { pathname } = useLocation()
  const { scrollTo } = useSmoothScroll()

  return useCallback(
    (href: string) => {
      if (href.startsWith('/#')) {
        const id = href.slice(1)
        if (pathname === '/') scrollTo(id)
        else navigate(href)
        return
      }
      navigate(href)
    },
    [navigate, pathname, scrollTo],
  )
}
