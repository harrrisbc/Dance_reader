import type {
  AnalysisResult,
  JobStatus,
  MotifType,
} from '../types'

const POLL_MS = 800

export async function startAnalyze(
  file: Blob,
  filename = 'video.mp4',
  opts?: { stride?: number; writeOverlay?: boolean },
): Promise<JobStatus> {
  const form = new FormData()
  form.append('file', file, filename)
  form.append('stride', String(opts?.stride ?? 2))
  form.append('write_overlay', String(opts?.writeOverlay ?? true))

  const res = await fetch('/api/analyze', {
    method: 'POST',
    body: form,
  })
  if (!res.ok) {
    const detail = await res.text().catch(() => '')
    throw new Error(detail || `分析請求失敗（${res.status}）`)
  }
  return (await res.json()) as JobStatus
}

export async function getJob(jobId: string): Promise<JobStatus> {
  const res = await fetch(`/api/jobs/${jobId}`)
  if (!res.ok) throw new Error(`無法取得工作狀態（${res.status}）`)
  return (await res.json()) as JobStatus
}

export async function pollJob(
  jobId: string,
  onProgress?: (progress: number, message: string) => void,
  signal?: AbortSignal,
): Promise<AnalysisResult> {
  while (true) {
    if (signal?.aborted) throw new Error('已取消分析')
    const job = await getJob(jobId)
    onProgress?.(job.progress, job.message)
    if (job.status === 'done' && job.result) return job.result
    if (job.status === 'error') {
      throw new Error(job.error || '分析失敗')
    }
    await sleep(POLL_MS, signal)
  }
}

export async function renameMotifType(
  jobId: string,
  typeId: string,
  name: string,
): Promise<JobStatus> {
  const res = await fetch(`/api/jobs/${jobId}/motif-types/${typeId}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name }),
  })
  if (!res.ok) throw new Error('重新命名失敗')
  return (await res.json()) as JobStatus
}

export function motifTypesToMarks(types: MotifType[]) {
  return types.flatMap((t) =>
    t.occurrences.map((o) => ({
      id: o.id,
      start: o.start,
      end: o.end,
      label: t.name,
      source: 'auto' as const,
      groupId: t.id,
      actorId: o.actorId,
      color: t.color,
      motifTypeId: t.id,
    })),
  )
}

function sleep(ms: number, signal?: AbortSignal) {
  return new Promise<void>((resolve, reject) => {
    const t = setTimeout(() => resolve(), ms)
    signal?.addEventListener(
      'abort',
      () => {
        clearTimeout(t)
        reject(new Error('已取消分析'))
      },
      { once: true },
    )
  })
}
