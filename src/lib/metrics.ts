import type { FramePose, MetricSample, PosePoint } from '../types'

const KEY_INDICES = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

function dist(a: PosePoint, b: PosePoint) {
  const dx = a.x - b.x
  const dy = a.y - b.y
  const dz = (a.z ?? 0) - (b.z ?? 0)
  return Math.hypot(dx, dy, dz)
}

/** Feature vector: torso-normalized key landmark XY. */
export function poseFeature(landmarks: PosePoint[]): number[] {
  const lHip = landmarks[23]
  const rHip = landmarks[24]
  const lShoulder = landmarks[11]
  const rShoulder = landmarks[12]
  if (!lHip || !rHip || !lShoulder || !rShoulder) return []

  const cx = (lHip.x + rHip.x + lShoulder.x + rShoulder.x) / 4
  const cy = (lHip.y + rHip.y + lShoulder.y + rShoulder.y) / 4
  const scale =
    Math.hypot(lShoulder.x - rShoulder.x, lShoulder.y - rShoulder.y) || 1

  const out: number[] = []
  for (const i of KEY_INDICES) {
    const p = landmarks[i]
    if (!p) {
      out.push(0, 0)
      continue
    }
    out.push((p.x - cx) / scale, (p.y - cy) / scale)
  }
  return out
}

export function cosineSimilarity(a: number[], b: number[]): number {
  if (!a.length || a.length !== b.length) return 0
  let dot = 0
  let na = 0
  let nb = 0
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i]
    na += a[i] * a[i]
    nb += b[i] * b[i]
  }
  if (na === 0 || nb === 0) return 0
  return dot / (Math.sqrt(na) * Math.sqrt(nb))
}

export function buildMetrics(poses: FramePose[]): MetricSample[] {
  if (poses.length < 2) return []

  const samples: MetricSample[] = []
  let maxEnergy = 1e-6
  let maxSpeed = 1e-6

  for (let i = 1; i < poses.length; i++) {
    const prev = poses[i - 1]
    const cur = poses[i]
    const dt = Math.max(cur.time - prev.time, 1 / 60)

    let energy = 0
    let speedSum = 0
    let speedCount = 0

    const n = Math.min(prev.landmarks.length, cur.landmarks.length)
    for (let j = 0; j < n; j++) {
      const a = prev.landmarks[j]
      const b = cur.landmarks[j]
      if (!a || !b) continue
      if ((a.visibility ?? 1) < 0.4 || (b.visibility ?? 1) < 0.4) continue
      const v = dist(a, b) / dt
      energy += v * v
      if (KEY_INDICES.includes(j)) {
        speedSum += v
        speedCount++
      }
    }

    const speed = speedCount ? speedSum / speedCount : 0
    maxEnergy = Math.max(maxEnergy, energy)
    maxSpeed = Math.max(maxSpeed, speed)
    samples.push({ time: cur.time, energy, speed })
  }

  return samples.map((s) => ({
    time: s.time,
    energy: s.energy / maxEnergy,
    speed: s.speed / maxSpeed,
  }))
}
