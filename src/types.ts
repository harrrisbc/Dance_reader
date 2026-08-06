export type PosePoint = {
  x: number
  y: number
  z: number
  visibility?: number
}

export type FramePose = {
  time: number
  landmarks: PosePoint[]
}

export type MetricSample = {
  time: number
  energy: number
  speed: number
}

export type RepeatMark = {
  id: string
  start: number
  end: number
  label: string
  source: 'auto' | 'manual'
  groupId?: string
  actorId?: string
  color?: string
  motifTypeId?: string
}

export type AnalysisState = {
  poses: FramePose[]
  metrics: MetricSample[]
  repeats: RepeatMark[]
  duration: number
}

/** Backend analysis.json contract (version 1) */
export type AnalysisKeypoint = [number, number, number] // x, y, conf (0–1)

export type AnalysisPoseFrame = {
  time: number
  keypoints: AnalysisKeypoint[]
}

export type AnalysisActor = {
  id: string
  label: string
  metrics: MetricSample[]
  poses: AnalysisPoseFrame[]
}

export type MotifOccurrence = {
  id: string
  actorId: string
  start: number
  end: number
}

export type MotifType = {
  id: string
  name: string
  color: string
  occurrences: MotifOccurrence[]
}

export type ManualMark = {
  id: string
  start: number
  end: number
  label: string
  actorId?: string | null
  motifTypeId?: string | null
}

export type AnalysisResult = {
  version: 1
  duration: number
  fps: number
  actors: AnalysisActor[]
  motifTypes: MotifType[]
  manualMarks: ManualMark[]
}

export type JobStatus = {
  id: string
  status: 'queued' | 'running' | 'done' | 'error'
  progress: number
  message: string
  result: AnalysisResult | null
  error: string | null
  overlayAvailable: boolean
}

/** Convert backend keypoints → FramePose landmarks for skeleton drawing. */
export function keypointsToLandmarks(
  keypoints: AnalysisKeypoint[],
): PosePoint[] {
  return keypoints.map(([x, y, conf]) => ({
    x,
    y,
    z: 0,
    visibility: conf,
  }))
}

export function actorPosesToFramePoses(
  actor: AnalysisActor | undefined,
): FramePose[] {
  if (!actor) return []
  return actor.poses.map((p) => ({
    time: p.time,
    landmarks: keypointsToLandmarks(p.keypoints),
  }))
}
