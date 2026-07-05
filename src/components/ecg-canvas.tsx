'use client'

import { useEffect, useRef } from 'react'

/**
 * ECG Canvas Monitor — Simple Full-Redraw Version
 *
 * On each animation frame:
 * 1. Clear the canvas
 * 2. Draw the ECG paper grid
 * 3. Draw all visible data points (ECG trace + reconstruction)
 * 4. Draw a green dot at the latest data position
 *
 * The dot sweeps left-to-right as new data arrives. When the buffer
 * fills up, old data scrolls off the left edge naturally (the buffer
 * is a rolling window of the most recent samples).
 */

export type EcgPoint = {
  t: number
  value: number
  prediction: number
  anomaly: boolean
}

type Props = {
  data: EcgPoint[]
  active: boolean
  height?: number
  pixelsPerSample?: number
}

// Visual constants
const GRID_SMALL = 10       // small grid square (px)
const GRID_LARGE = 50       // large grid square (px)
const COLOR_GRID_SMALL = '#fde2e2'
const COLOR_GRID_LARGE = '#f9c5c5'
const COLOR_TRACE = '#0f172a'       // black
const COLOR_RECON = '#0891b2'       // cyan
const COLOR_DOT = '#22c55e'         // green
const COLOR_ANOMALY = '#fecaca'     // red tint

function drawGrid(ctx: CanvasRenderingContext2D, w: number, h: number) {
  // Background
  ctx.fillStyle = '#ffffff'
  ctx.fillRect(0, 0, w, h)

  // Small grid
  ctx.strokeStyle = COLOR_GRID_SMALL
  ctx.lineWidth = 0.5
  ctx.beginPath()
  for (let x = 0; x <= w; x += GRID_SMALL) { ctx.moveTo(x, 0); ctx.lineTo(x, h) }
  for (let y = 0; y <= h; y += GRID_SMALL) { ctx.moveTo(0, y); ctx.lineTo(w, y) }
  ctx.stroke()

  // Large grid
  ctx.strokeStyle = COLOR_GRID_LARGE
  ctx.lineWidth = 1
  ctx.beginPath()
  for (let x = 0; x <= w; x += GRID_LARGE) { ctx.moveTo(x, 0); ctx.lineTo(x, h) }
  for (let y = 0; y <= h; y += GRID_LARGE) { ctx.moveTo(0, y); ctx.lineTo(w, y) }
  ctx.stroke()
}

function valueToY(value: number, h: number): number {
  return h / 2 - value * (h / 12)
}

export function EcgCanvas({
  data,
  active,
  height = 300,
  pixelsPerSample = 3,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const dataRef = useRef<EcgPoint[]>(data)
  const activeRef = useRef<boolean>(active)

  useEffect(() => { dataRef.current = data }, [data])
  useEffect(() => { activeRef.current = active }, [active])

  // Set up canvas size
  useEffect(() => {
    const canvas = canvasRef.current
    const container = containerRef.current
    if (!canvas || !container) return

    const resize = () => {
      const w = container.clientWidth
      const dpr = window.devicePixelRatio || 1
      canvas.width = w * dpr
      canvas.height = height * dpr
      canvas.style.width = `${w}px`
      canvas.style.height = `${height}px`
      const ctx = canvas.getContext('2d')
      if (ctx) ctx.scale(dpr, dpr)
    }
    resize()
    window.addEventListener('resize', resize)
    return () => window.removeEventListener('resize', resize)
  }, [height])

  // Animation loop — full redraw every frame
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    let raf: number

    const render = () => {
      const dpr = window.devicePixelRatio || 1
      const w = canvas.width / dpr
      const h = height
      const points = dataRef.current

      // 1. Clear + draw grid
      drawGrid(ctx, w, h)

      if (points.length > 1) {
        // How many points fit on screen
        const maxPoints = Math.floor(w / pixelsPerSample)
        // Take the most recent points (rolling window)
        const visible = points.slice(-maxPoints)
        // Start x so the latest point is near the right edge
        const startX = w - visible.length * pixelsPerSample

        // 2. Draw anomaly region highlights
        ctx.fillStyle = COLOR_ANOMALY
        ctx.globalAlpha = 0.3
        for (let i = 0; i < visible.length; i++) {
          if (visible[i].anomaly) {
            const x = startX + i * pixelsPerSample
            ctx.fillRect(x, 0, pixelsPerSample, h)
          }
        }
        ctx.globalAlpha = 1

        // 3. Draw reconstruction (cyan dashed)
        ctx.strokeStyle = COLOR_RECON
        ctx.lineWidth = 1
        ctx.setLineDash([4, 3])
        ctx.beginPath()
        for (let i = 0; i < visible.length; i++) {
          const x = startX + i * pixelsPerSample
          const y = valueToY(visible[i].prediction, h)
          if (i === 0) ctx.moveTo(x, y)
          else ctx.lineTo(x, y)
        }
        ctx.stroke()
        ctx.setLineDash([])

        // 4. Draw ECG trace (black)
        ctx.strokeStyle = COLOR_TRACE
        ctx.lineWidth = 1.8
        ctx.lineJoin = 'round'
        ctx.lineCap = 'round'
        ctx.beginPath()
        for (let i = 0; i < visible.length; i++) {
          const x = startX + i * pixelsPerSample
          const y = valueToY(visible[i].value, h)
          if (i === 0) ctx.moveTo(x, y)
          else ctx.lineTo(x, y)
        }
        ctx.stroke()

        // 5. Draw the green sweep dot at the latest point
        if (activeRef.current && visible.length > 0) {
          const last = visible[visible.length - 1]
          const dotX = startX + (visible.length - 1) * pixelsPerSample
          const dotY = valueToY(last.value, h)

          // Glow
          ctx.shadowColor = COLOR_DOT
          ctx.shadowBlur = 12
          ctx.fillStyle = COLOR_DOT
          ctx.beginPath()
          ctx.arc(dotX, dotY, 4, 0, Math.PI * 2)
          ctx.fill()
          ctx.shadowBlur = 0

          // Inner white dot for contrast
          ctx.fillStyle = '#ffffff'
          ctx.beginPath()
          ctx.arc(dotX, dotY, 1.5, 0, Math.PI * 2)
          ctx.fill()
        }
      } else if (!activeRef.current) {
        // Idle message
        ctx.fillStyle = '#94a3b8'
        ctx.font = '14px sans-serif'
        ctx.textAlign = 'center'
        ctx.fillText('Idle — click Start to begin', w / 2, h / 2)
      }

      raf = requestAnimationFrame(render)
    }

    raf = requestAnimationFrame(render)
    return () => cancelAnimationFrame(raf)
  }, [height, pixelsPerSample])

  return (
    <div ref={containerRef} className="w-full">
      <canvas
        ref={canvasRef}
        className="w-full rounded-lg border border-slate-200"
        style={{ height: `${height}px` }}
      />
    </div>
  )
}
