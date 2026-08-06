import { cosineSimilarity, poseFeature } from './metrics'
import type { FramePose, RepeatMark } from '../types'

function uid() {
  return `m_${Math.random().toString(36).slice(2, 10)}`
}

/**
 * Find repeating movement windows via sliding-window pose similarity.
 * MVP: compare windows of ~0.8s; cluster high-similarity pairs.
 */
export function detectRepeats(
  poses: FramePose[],
  duration: number,
): RepeatMark[] {
  if (poses.length < 30 || duration < 2) return []

  const fps = poses.length / duration
  const win = Math.max(8, Math.round(fps * 0.8))
  const step = Math.max(2, Math.round(win / 3))

  const features: number[][] = poses.map((p) => poseFeature(p.landmarks))
  const windows: { startIdx: number; endIdx: number; feat: number[] }[] = []

  for (let i = 0; i + win < features.length; i += step) {
    const slice = features.slice(i, i + win)
    if (slice.some((f) => f.length === 0)) continue
    const avg = new Array(slice[0].length).fill(0)
    for (const f of slice) {
      for (let k = 0; k < f.length; k++) avg[k] += f[k]
    }
    for (let k = 0; k < avg.length; k++) avg[k] /= slice.length
    windows.push({ startIdx: i, endIdx: i + win, feat: avg })
  }

  type Pair = { a: number; b: number; score: number }
  const pairs: Pair[] = []
  const minGap = Math.round(fps * 1.2)

  for (let i = 0; i < windows.length; i++) {
    for (let j = i + 1; j < windows.length; j++) {
      if (windows[j].startIdx - windows[i].endIdx < minGap) continue
      const score = cosineSimilarity(windows[i].feat, windows[j].feat)
      if (score >= 0.92) pairs.push({ a: i, b: j, score })
    }
  }

  pairs.sort((x, y) => y.score - x.score)

  const used = new Set<number>()
  const marks: RepeatMark[] = []
  let group = 1

  for (const p of pairs) {
    if (used.has(p.a) || used.has(p.b)) continue
    used.add(p.a)
    used.add(p.b)
    const groupId = `g${group++}`
    const wa = windows[p.a]
    const wb = windows[p.b]
    marks.push({
      id: uid(),
      start: poses[wa.startIdx].time,
      end: poses[wa.endIdx - 1].time,
      label: `重複 ${group}`,
      source: 'auto',
      groupId,
    })
    marks.push({
      id: uid(),
      start: poses[wb.startIdx].time,
      end: poses[wb.endIdx - 1].time,
      label: `重複 ${group - 1}`,
      source: 'auto',
      groupId,
    })
    if (marks.length >= 12) break
  }

  return marks.sort((a, b) => a.start - b.start)
}

export function createManualMark(
  start: number,
  end: number,
  label?: string,
): RepeatMark {
  const s = Math.min(start, end)
  const e = Math.max(start, end)
  return {
    id: uid(),
    start: s,
    end: e,
    label: label?.trim() || '手動標記',
    source: 'manual',
  }
}
