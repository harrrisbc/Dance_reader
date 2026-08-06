import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Controls } from './components/Controls'
import { ImportPanel } from './components/ImportPanel'
import { Stage } from './components/Stage'
import { Timeline } from './components/Timeline'
import { useLiveSkeleton } from './hooks/useLiveSkeleton'
import { motifTypesToMarks, pollJob, renameMotifType, startAnalyze } from './lib/api'
import { createManualMark } from './lib/repeats'
import type {
  AnalysisResult,
  FramePose,
  MetricSample,
  MotifType,
  RepeatMark,
} from './types'
import { actorPosesToFramePoses } from './types'
import './App.css'

export default function App() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const objectUrlRef = useRef<string | null>(null)
  const mediaBlobRef = useRef<Blob | null>(null)
  const mediaNameRef = useRef('video.mp4')
  const abortRef = useRef<AbortController | null>(null)

  const [src, setSrc] = useState<string | null>(null)
  const [duration, setDuration] = useState(0)
  const [currentTime, setCurrentTime] = useState(0)
  const [playing, setPlaying] = useState(false)
  const [skeletonOn, setSkeletonOn] = useState(true)
  const [poses, setPoses] = useState<FramePose[]>([])
  const [multiPoses, setMultiPoses] = useState<FramePose[][]>([])
  const [metrics, setMetrics] = useState<MetricSample[]>([])
  const [marks, setMarks] = useState<RepeatMark[]>([])
  const [motifTypes, setMotifTypes] = useState<MotifType[]>([])
  const [actors, setActors] = useState<{ id: string; label: string }[]>([])
  const [selectedActorId, setSelectedActorId] = useState<string>('all')
  const [jobId, setJobId] = useState<string | null>(null)
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null)
  const [analyzing, setAnalyzing] = useState(false)
  const [progress, setProgress] = useState(0)
  const [progressLabel, setProgressLabel] = useState('分析中')
  const [markMode, setMarkMode] = useState(false)
  const [draftStart, setDraftStart] = useState<number | null>(null)
  const [selectedMarkId, setSelectedMarkId] = useState<string | null>(null)
  const [youtubeUrl, setYoutubeUrl] = useState('')
  const [youtubeBusy, setYoutubeBusy] = useState(false)
  const [youtubeError, setYoutubeError] = useState<string | null>(null)

  useLiveSkeleton(
    videoRef,
    canvasRef,
    skeletonOn && !!src,
    poses,
    selectedActorId === 'all' && multiPoses.length > 1 ? multiPoses : undefined,
  )

  const revokeSrc = useCallback(() => {
    if (objectUrlRef.current) {
      URL.revokeObjectURL(objectUrlRef.current)
      objectUrlRef.current = null
    }
  }, [])

  useEffect(() => () => {
    revokeSrc()
    abortRef.current?.abort()
  }, [revokeSrc])

  const applyAnalysis = useCallback(
    (result: AnalysisResult, keepManual = true) => {
      setAnalysis(result)
      setActors(result.actors.map((a) => ({ id: a.id, label: a.label })))
      setMotifTypes(result.motifTypes)

      const primary =
        selectedActorId !== 'all' &&
        result.actors.some((a) => a.id === selectedActorId)
          ? selectedActorId
          : result.actors[0]?.id

      if (selectedActorId === 'all') {
        setMultiPoses(result.actors.map((a) => actorPosesToFramePoses(a)))
        setPoses(actorPosesToFramePoses(result.actors[0]))
        // Average / first actor metrics for chart when "all"
        setMetrics(result.actors[0]?.metrics ?? [])
      } else {
        const actor = result.actors.find((a) => a.id === primary) ?? result.actors[0]
        setMultiPoses([])
        setPoses(actorPosesToFramePoses(actor))
        setMetrics(actor?.metrics ?? [])
        if (actor && selectedActorId !== actor.id) {
          setSelectedActorId(actor.id)
        }
      }

      const autoMarks = motifTypesToMarks(result.motifTypes)
      setMarks((prev) => {
        const manual = keepManual ? prev.filter((m) => m.source === 'manual') : []
        return [...autoMarks, ...manual].sort((a, b) => a.start - b.start)
      })

      if (result.duration > 0) {
        setDuration((d) => (d > 0 ? d : result.duration))
      }
    },
    [selectedActorId],
  )

  // Re-derive poses/metrics when actor selection changes
  useEffect(() => {
    if (!analysis) return
    if (selectedActorId === 'all') {
      setMultiPoses(analysis.actors.map((a) => actorPosesToFramePoses(a)))
      setPoses(actorPosesToFramePoses(analysis.actors[0]))
      setMetrics(analysis.actors[0]?.metrics ?? [])
    } else {
      const actor = analysis.actors.find((a) => a.id === selectedActorId)
      setMultiPoses([])
      setPoses(actorPosesToFramePoses(actor))
      setMetrics(actor?.metrics ?? [])
    }
  }, [selectedActorId, analysis])

  const visibleMarks = useMemo(() => {
    if (selectedActorId === 'all') return marks
    return marks.filter(
      (m) => m.source === 'manual' || !m.actorId || m.actorId === selectedActorId,
    )
  }, [marks, selectedActorId])

  const resetAnalysis = () => {
    abortRef.current?.abort()
    abortRef.current = null
    setPoses([])
    setMultiPoses([])
    setMetrics([])
    setMarks([])
    setMotifTypes([])
    setActors([])
    setSelectedActorId('all')
    setAnalysis(null)
    setJobId(null)
    setSelectedMarkId(null)
    setDraftStart(null)
    setMarkMode(false)
    setProgress(0)
    setProgressLabel('分析中')
  }

  const loadBlob = (blob: Blob, name = 'video.mp4') => {
    revokeSrc()
    const url = URL.createObjectURL(blob)
    objectUrlRef.current = url
    mediaBlobRef.current = blob
    mediaNameRef.current = name
    setSrc(url)
    resetAnalysis()
    setPlaying(false)
  }

  const onLocalFile = (file: File) => {
    setYoutubeError(null)
    loadBlob(file, file.name || 'video.mp4')
  }

  const onYoutubeSubmit = async () => {
    const url = youtubeUrl.trim()
    if (!url) return
    setYoutubeBusy(true)
    setYoutubeError(null)
    try {
      const res = await fetch('/api/youtube', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url }),
      })
      if (!res.ok) {
        const data = (await res.json().catch(() => null)) as {
          error?: string
        } | null
        throw new Error(data?.error || `匯入失敗（${res.status}）`)
      }
      const blob = await res.blob()
      loadBlob(blob, 'youtube.mp4')
    } catch (err) {
      const msg =
        err instanceof Error
          ? err.message
          : '無法匯入 YouTube。請確認本機已安裝 yt-dlp，或改用本機影片。'
      setYoutubeError(msg)
    } finally {
      setYoutubeBusy(false)
    }
  }

  const onLoadedMetadata = () => {
    const v = videoRef.current
    if (!v) return
    setDuration(v.duration || 0)
  }

  const onTimeUpdate = () => {
    const v = videoRef.current
    if (!v) return
    setCurrentTime(v.currentTime)
    setPlaying(!v.paused)
  }

  const seek = (time: number) => {
    const v = videoRef.current
    if (!v) return
    v.currentTime = Math.max(0, Math.min(duration || v.duration || 0, time))
    setCurrentTime(v.currentTime)
  }

  const togglePlay = async () => {
    const v = videoRef.current
    if (!v) return
    if (v.paused) {
      await v.play()
      setPlaying(true)
    } else {
      v.pause()
      setPlaying(false)
    }
  }

  const onAnalyze = async () => {
    const blob = mediaBlobRef.current
    if (!blob) {
      setYoutubeError('請先開啟影片再分析。')
      return
    }
    abortRef.current?.abort()
    const ac = new AbortController()
    abortRef.current = ac

    setAnalyzing(true)
    setProgress(0)
    setProgressLabel('上傳中')
    setYoutubeError(null)
    try {
      const job = await startAnalyze(blob, mediaNameRef.current, {
        stride: 2,
        writeOverlay: true,
      })
      setJobId(job.id)
      setProgressLabel(job.message || '分析中')
      const result = await pollJob(
        job.id,
        (p, msg) => {
          setProgress(p)
          if (msg) setProgressLabel(msg)
        },
        ac.signal,
      )
      applyAnalysis(result, true)
    } catch (err) {
      console.error(err)
      if (err instanceof Error && err.message === '已取消分析') return
      setYoutubeError(
        err instanceof Error
          ? err.message
          : '分析失敗。請確認後端已啟動（埠 8000），或換一支較短的影片。',
      )
    } finally {
      setAnalyzing(false)
    }
  }

  const onRenameType = async (typeId: string, currentName: string) => {
    const next = window.prompt('重新命名動作類型', currentName)
    if (next == null) return
    const name = next.trim()
    if (!name) return

    setMotifTypes((prev) =>
      prev.map((t) => (t.id === typeId ? { ...t, name } : t)),
    )
    setMarks((prev) =>
      prev.map((m) =>
        m.motifTypeId === typeId || m.groupId === typeId ? { ...m, label: name } : m,
      ),
    )
    setAnalysis((prev) =>
      prev
        ? {
            ...prev,
            motifTypes: prev.motifTypes.map((t) =>
              t.id === typeId ? { ...t, name } : t,
            ),
          }
        : prev,
    )

    if (jobId) {
      try {
        await renameMotifType(jobId, typeId, name)
      } catch (err) {
        console.error(err)
      }
    }
  }

  const onTimelineClick = (time: number) => {
    if (!markMode) return
    if (draftStart == null) {
      setDraftStart(time)
      return
    }
    const label = window.prompt('標記名稱（可留空）', '') ?? ''
    const assign = motifTypes.length
      ? window.prompt(
          `指定類型 ID（可留空新建）。現有：${motifTypes.map((t) => t.id).join(', ')}`,
          '',
        )
      : ''
    const mark = createManualMark(draftStart, time, label || undefined)
    if (assign?.trim()) {
      const mt = motifTypes.find((t) => t.id === assign.trim())
      if (mt) {
        mark.motifTypeId = mt.id
        mark.groupId = mt.id
        mark.color = mt.color
        mark.label = label.trim() || mt.name
      }
    }
    if (selectedActorId !== 'all') mark.actorId = selectedActorId
    setMarks((prev) => [...prev, mark].sort((a, b) => a.start - b.start))
    setSelectedMarkId(mark.id)
    setDraftStart(null)
  }

  return (
    <div className="app">
      <header className="masthead">
        <p className="brand">Dance Analyser</p>
        <p className="tagline">舞台上讀舞——骨架、能量、重複動作。</p>
      </header>

      <ImportPanel
        onLocalFile={onLocalFile}
        youtubeUrl={youtubeUrl}
        onYoutubeUrlChange={setYoutubeUrl}
        onYoutubeSubmit={onYoutubeSubmit}
        youtubeBusy={youtubeBusy}
        youtubeError={youtubeError}
      />

      <Stage
        videoRef={videoRef}
        canvasRef={canvasRef}
        src={src}
        hasMedia={!!src}
        onLoadedMetadata={onLoadedMetadata}
        onTimeUpdate={onTimeUpdate}
      />

      <Controls
        hasMedia={!!src}
        playing={playing}
        analyzing={analyzing}
        progress={progress}
        progressLabel={progressLabel}
        skeletonOn={skeletonOn}
        markMode={markMode}
        selectedMarkId={selectedMarkId}
        actors={actors}
        selectedActorId={selectedActorId}
        onSelectActor={setSelectedActorId}
        onTogglePlay={togglePlay}
        onAnalyze={onAnalyze}
        onToggleSkeleton={() => setSkeletonOn((s) => !s)}
        onToggleMarkMode={() => {
          setMarkMode((m) => !m)
          setDraftStart(null)
        }}
        onDeleteMark={() => {
          if (!selectedMarkId) return
          setMarks((prev) => prev.filter((m) => m.id !== selectedMarkId))
          setSelectedMarkId(null)
        }}
        onSeekRelative={(d) => seek(currentTime + d)}
      />

      <Timeline
        duration={duration}
        currentTime={currentTime}
        metrics={metrics}
        marks={visibleMarks}
        markMode={markMode}
        draftStart={draftStart}
        onSeek={seek}
        onTimelineClick={onTimelineClick}
        onSelectMark={setSelectedMarkId}
        selectedMarkId={selectedMarkId}
      />

      {motifTypes.length > 0 && (
        <ul className="mark-list" aria-label="動作類型">
          {motifTypes.map((t) => (
            <li key={t.id} className="motif-row">
              <button
                type="button"
                className={
                  visibleMarks.some(
                    (m) => m.motifTypeId === t.id && m.id === selectedMarkId,
                  )
                    ? 'active'
                    : ''
                }
                onClick={() => {
                  const first = t.occurrences[0]
                  if (first) {
                    setSelectedMarkId(first.id)
                    seek(first.start)
                  }
                }}
              >
                <span className="type-row">
                  <span
                    className="type-swatch"
                    style={{ background: t.color }}
                    aria-hidden
                  />
                  {t.name} · {t.occurrences.length} 次
                </span>
              </button>
              <button
                type="button"
                className="linkish rename-btn"
                onClick={() => void onRenameType(t.id, t.name)}
              >
                重新命名
              </button>
            </li>
          ))}
        </ul>
      )}

      {visibleMarks.filter((m) => m.source === 'manual').length > 0 && (
        <ul className="mark-list" aria-label="手動標記">
          {visibleMarks
            .filter((m) => m.source === 'manual')
            .map((m) => (
              <li key={m.id}>
                <button
                  type="button"
                  className={m.id === selectedMarkId ? 'active' : ''}
                  onClick={() => {
                    setSelectedMarkId(m.id)
                    seek(m.start)
                  }}
                >
                  <span>{m.label}</span>
                  <span className="meta">
                    手動 · {m.start.toFixed(1)}s–{m.end.toFixed(1)}s
                  </span>
                </button>
              </li>
            ))}
        </ul>
      )}
    </div>
  )
}
