import type { PosePoint } from '../types'

/** MediaPipe Pose (33) connections — subset for clear stage overlay. */
export const MP_POSE_CONNECTIONS: [number, number][] = [
  [0, 1],
  [1, 2],
  [2, 3],
  [3, 7],
  [0, 4],
  [4, 5],
  [5, 6],
  [6, 8],
  [9, 10],
  [11, 12],
  [11, 13],
  [13, 15],
  [15, 17],
  [15, 19],
  [15, 21],
  [12, 14],
  [14, 16],
  [16, 18],
  [16, 20],
  [16, 22],
  [11, 23],
  [12, 24],
  [23, 24],
  [23, 25],
  [25, 27],
  [27, 29],
  [27, 31],
  [24, 26],
  [26, 28],
  [28, 30],
  [28, 32],
]

/** COCO-17 (YOLO-Pose) connections */
export const COCO_POSE_CONNECTIONS: [number, number][] = [
  [0, 1],
  [0, 2],
  [1, 3],
  [2, 4],
  [5, 6],
  [5, 7],
  [7, 9],
  [6, 8],
  [8, 10],
  [5, 11],
  [6, 12],
  [11, 12],
  [11, 13],
  [13, 15],
  [12, 14],
  [14, 16],
]

/** @deprecated use MP_POSE_CONNECTIONS or resolveConnections */
export const POSE_CONNECTIONS = MP_POSE_CONNECTIONS

const VIS_MIN = 0.3

export function resolveConnections(count: number): [number, number][] {
  return count <= 17 ? COCO_POSE_CONNECTIONS : MP_POSE_CONNECTIONS
}

export function drawSkeleton(
  ctx: CanvasRenderingContext2D,
  landmarks: PosePoint[],
  width: number,
  height: number,
  color = 'rgba(255, 236, 179, 0.95)',
  glow = 'rgba(232, 197, 110, 0.35)',
) {
  const connections = resolveConnections(landmarks.length)

  ctx.save()
  ctx.lineCap = 'round'
  ctx.lineJoin = 'round'

  ctx.strokeStyle = glow
  ctx.lineWidth = 6
  for (const [a, b] of connections) {
    const pa = landmarks[a]
    const pb = landmarks[b]
    if (!pa || !pb) continue
    if ((pa.visibility ?? 1) < VIS_MIN || (pb.visibility ?? 1) < VIS_MIN) continue
    ctx.beginPath()
    ctx.moveTo(pa.x * width, pa.y * height)
    ctx.lineTo(pb.x * width, pb.y * height)
    ctx.stroke()
  }

  ctx.strokeStyle = color
  ctx.lineWidth = 2.2
  for (const [a, b] of connections) {
    const pa = landmarks[a]
    const pb = landmarks[b]
    if (!pa || !pb) continue
    if ((pa.visibility ?? 1) < VIS_MIN || (pb.visibility ?? 1) < VIS_MIN) continue
    ctx.beginPath()
    ctx.moveTo(pa.x * width, pa.y * height)
    ctx.lineTo(pb.x * width, pb.y * height)
    ctx.stroke()
  }

  for (let i = 0; i < landmarks.length; i++) {
    const p = landmarks[i]
    if (!p || (p.visibility ?? 1) < VIS_MIN) continue
    const x = p.x * width
    const y = p.y * height
    ctx.beginPath()
    ctx.fillStyle = 'rgba(255, 248, 220, 0.95)'
    ctx.arc(x, y, 3.2, 0, Math.PI * 2)
    ctx.fill()
  }

  ctx.restore()
}

export function drawSkeletons(
  ctx: CanvasRenderingContext2D,
  people: PosePoint[][],
  width: number,
  height: number,
  colors?: string[],
) {
  const palette = colors ?? [
    'rgba(255, 236, 179, 0.95)',
    'rgba(196, 92, 74, 0.95)',
    'rgba(120, 180, 160, 0.95)',
    'rgba(166, 124, 82, 0.95)',
  ]
  people.forEach((lm, i) => {
    if (!lm.length) return
    const c = palette[i % palette.length]
    drawSkeleton(ctx, lm, width, height, c, c.replace(/[\d.]+\)$/, '0.35)'))
  })
}
