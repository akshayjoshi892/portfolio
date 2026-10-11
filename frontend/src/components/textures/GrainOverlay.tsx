/**
 * Full-viewport film grain. Uses a tiled SVG noise image as a CSS background,
 * which the browser rasterises once — far cheaper than a live SVG filter layer.
 */
const NOISE = `url("data:image/svg+xml;utf8,${encodeURIComponent(
  `<svg xmlns='http://www.w3.org/2000/svg' width='180' height='180'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='3' stitchTiles='stitch'/><feColorMatrix type='saturate' values='0'/></filter><rect width='100%' height='100%' filter='url(%23n)' opacity='0.5'/></svg>`,
).replace(/%2523/g, '%23')}")`

export function GrainOverlay() {
  return (
    <div
      aria-hidden
      className="pointer-events-none fixed inset-0 z-[1] mix-blend-overlay"
      style={{ opacity: 'var(--grain-opacity)', backgroundImage: NOISE, backgroundSize: '180px 180px' }}
    />
  )
}
