import { createContext, useContext } from 'react'

/** `true` once the preloader has lifted, so entrance animations can start in sync. */
export const IntroContext = createContext(true)

export function useIntroDone() {
  return useContext(IntroContext)
}
