import { useCallback, useEffect, useRef, useState } from 'react'
import { getPoseLandmarker, landmarksFromResult } from '../lib/pose'
import { drawSkeleton, drawSkeletons } from '../lib/skeleton'
import type { FramePose, PosePoint } from '../types'

const ACTOR_COLORS = [
  'rgba(255, 236, 179, 0.95)',
  'rgba(196, 92, 74, 0.95)',
  'rgba(120, 180, 160, 0.95)',
  'rgba(166, 124, 82, 0.95)',
]

export function useLiveSkeleton(
  videoRef: React.RefObject<HTMLVideoElement | null>,
  canvasRef: React.RefObject<HTMLCanvasElement | null>,
  enabled: boolean,
  recordedPoses: FramePose[],
  /** When multiple actors selected, pass parallel pose series */
  multiPoses?: FramePose[][],
) {
  const [ready, setReady] = useState(false)
  const rafRef = useRef(0)

  useEffect(() => {
    let cancelled = false
    getPoseLandmarker()
      .then(() => {
        if (!cancelled) setReady(true)
      })
      .catch(console.error)
    return () => {
      cancelled = true
    }
  }, [])

  const findNearestPose = useCallback(
    (poses: FramePose[], time: number): PosePoint[] | null => {
      if (!poses.length) return null
      let best = poses[0]
      let bestD = Math.abs(best.time - time)
      for (const p of poses) {
        const d = Math.abs(p.time - time)
        if (d < bestD) {
          best = p
          bestD = d
        }
      }
      // Skip if too far from sampled frame
      if (bestD > 0.35) return null
      return best.landmarks
    },
    [],
  )

  useEffect(() => {
    if (!enabled) return
    const hasRecorded =
      recordedPoses.length > 0 || (multiPoses && multiPoses.some((p) => p.length > 0))
    if (!hasRecorded && !ready) return

    const tick = async () => {
      const video = videoRef.current
      const canvas = canvasRef.current
      if (!video || !canvas || video.readyState < 2) {
        rafRef.current = requestAnimationFrame(tick)
        return
      }

      const w = video.videoWidth
      const h = video.videoHeight
      if (w && h && (canvas.width !== w || canvas.height !== h)) {
        canvas.width = w
        canvas.height = h
      }

      const ctx = canvas.getContext('2d')
      if (!ctx) {
        rafRef.current = requestAnimationFrame(tick)
        return
      }

      ctx.clearRect(0, 0, canvas.width, canvas.height)

      if (multiPoses && multiPoses.length > 0) {
        const people = multiPoses
          .map((poses) => findNearestPose(poses, video.currentTime))
          .filter((p): p is PosePoint[] => !!p && p.length > 0)
        if (people.length) {
          drawSkeletons(ctx, people, canvas.width, canvas.height, ACTOR_COLORS)
        }
      } else if (recordedPoses.length) {
        const landmarks = findNearestPose(recordedPoses, video.currentTime)
        if (landmarks) {
          drawSkeleton(ctx, landmarks, canvas.width, canvas.height)
        }
      } else if (!video.paused && !video.ended && ready) {
        try {
          const landmarker = await getPoseLandmarker()
          const result = landmarker.detectForVideo(video, performance.now())
          if (result.landmarks[0]) {
            drawSkeleton(
              ctx,
              landmarksFromResult(result.landmarks[0]),
              canvas.width,
              canvas.height,
            )
          }
        } catch {
          /* ignore frame errors */
        }
      }

      rafRef.current = requestAnimationFrame(tick)
    }

    rafRef.current = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(rafRef.current)
  }, [
    enabled,
    ready,
    videoRef,
    canvasRef,
    recordedPoses,
    multiPoses,
    findNearestPose,
  ])

  return { ready }
}
