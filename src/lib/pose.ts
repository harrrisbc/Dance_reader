import { FilesetResolver, PoseLandmarker } from '@mediapipe/tasks-vision'
import type { FramePose, PosePoint } from '../types'

let landmarkerPromise: Promise<PoseLandmarker> | null = null

export async function getPoseLandmarker(): Promise<PoseLandmarker> {
  if (!landmarkerPromise) {
    landmarkerPromise = (async () => {
      const vision = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm',
      )
      return PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath:
            'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
          delegate: 'GPU',
        },
        runningMode: 'VIDEO',
        numPoses: 1,
        minPoseDetectionConfidence: 0.4,
        minPosePresenceConfidence: 0.4,
        minTrackingConfidence: 0.4,
      })
    })()
  }
  return landmarkerPromise
}

export function landmarksFromResult(
  landmarks: { x: number; y: number; z: number; visibility?: number }[],
): PosePoint[] {
  return landmarks.map((l) => ({
    x: l.x,
    y: l.y,
    z: l.z,
    visibility: l.visibility,
  }))
}

export async function analyzeVideoPoses(
  video: HTMLVideoElement,
  onProgress?: (pct: number) => void,
  sampleFps = 12,
): Promise<FramePose[]> {
  const landmarker = await getPoseLandmarker()
  const duration = video.duration
  if (!Number.isFinite(duration) || duration <= 0) return []

  const interval = 1 / sampleFps
  const poses: FramePose[] = []
  const wasPaused = video.paused
  const prevTime = video.currentTime

  video.pause()

  for (let t = 0; t < duration; t += interval) {
    await seekVideo(video, t)
    const result = landmarker.detectForVideo(video, performance.now())
    const lm = result.landmarks[0]
    if (lm?.length) {
      poses.push({
        time: t,
        landmarks: landmarksFromResult(lm),
      })
    }
    onProgress?.(Math.min(1, t / duration))
  }

  await seekVideo(video, prevTime)
  if (!wasPaused) void video.play()
  onProgress?.(1)
  return poses
}

function seekVideo(video: HTMLVideoElement, time: number) {
  return new Promise<void>((resolve) => {
    const onSeeked = () => {
      video.removeEventListener('seeked', onSeeked)
      resolve()
    }
    video.addEventListener('seeked', onSeeked)
    video.currentTime = Math.min(time, video.duration || time)
  })
}
