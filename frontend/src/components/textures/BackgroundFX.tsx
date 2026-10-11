/**
 * Ambient page background: a fine dot grid that fades toward the edges, two
 * soft accent glows and a vignette. Purely CSS (see .bg-fx in index.css), fixed
 * to the viewport so it costs nothing on scroll.
 */
export function BackgroundFX() {
  return (
    <div aria-hidden className="bg-fx pointer-events-none fixed inset-0 z-0 overflow-hidden">
      <div className="bg-fx__glow bg-fx__glow--a" />
      <div className="bg-fx__glow bg-fx__glow--b" />
      <div className="bg-fx__dots" />
      <div className="bg-fx__vignette" />
    </div>
  )
}
