import { useState, type ReactNode } from 'react'
import { useLocation } from 'react-router-dom'
import { Header } from './Header'
import { Footer } from './Footer'
import { Preloader } from './Preloader'
import { IntroContext } from './IntroContext'
import { GrainOverlay } from '@/components/textures/GrainOverlay'
import { SmoothScrollProvider } from '@/components/providers/SmoothScroll'
import { PresenceTracker } from '@/components/providers/PresenceTracker'

export function AppShell({ children }: { children: ReactNode }) {
  const [introDone, setIntroDone] = useState(false)
  const { pathname } = useLocation()

  return (
    <SmoothScrollProvider>
      <IntroContext.Provider value={introDone}>
        <Preloader onDone={() => setIntroDone(true)} />
        <GrainOverlay />
        <PresenceTracker />
        <div className="relative flex min-h-dvh flex-col">
          <Header />
          <main className={`relative z-[2] flex-1 ${pathname === '/' ? '' : 'pt-20'}`}>{children}</main>
          {pathname === '/' && <Footer />}
        </div>
      </IntroContext.Provider>
    </SmoothScrollProvider>
  )
}
