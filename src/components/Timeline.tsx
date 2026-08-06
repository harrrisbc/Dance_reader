import type { MetricSample, RepeatMark } from '../types'

type Props = {
  duration: number
  currentTime: number
  metrics: MetricSample[]
  marks: RepeatMark[]
  markMode: boolean
  draftStart: number | null
  onSeek: (time: number) => void
  onTimelineClick: (time: number) => void
  onSelectMark: (id: string) => void
  selectedMarkId: string | null
}

function formatTime(t: number) {
  if (!Number.isFinite(t) || t < 0) return '0:00'
  const m = Math.floor(t / 60)
  const s = Math.floor(t % 60)
  return `${m}:${s.toString().padStart(2, '0')}`
}

function hexToRgba(hex: string, alpha: number) {
  const h = hex.replace('#', '')
  if (h.length !== 6) return `rgba(196, 92, 74, ${alpha})`
  const r = parseInt(h.slice(0, 2), 16)
  const g = parseInt(h.slice(2, 4), 16)
  const b = parseInt(h.slice(4, 6), 16)
  return `rgba(${r}, ${g}, ${b}, ${alpha})`
}

export function Timeline({
  duration,
  currentTime,
  metrics,
  marks,
  markMode,
  draftStart,
  onSeek,
  onTimelineClick,
  onSelectMark,
  selectedMarkId,
}: Props) {
  const width = 1000
  const height = 120
  const safeDur = duration > 0 ? duration : 1

  const energyPath = toPath(metrics, safeDur, width, height, 'energy', 0.15)
  const speedPath = toPath(metrics, safeDur, width, height, 'speed', 0.55)

  const playX = (currentTime / safeDur) * width

  const handleClick = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect()
    const ratio = (e.clientX - rect.left) / rect.width
    const time = Math.max(0, Math.min(duration, ratio * duration))
    if (markMode) onTimelineClick(time)
    else onSeek(time)
  }

  return (
    <section className="timeline" aria-label="時間軸">
      <div className="timeline-head">
        <h2>時間軸</h2>
        <span className="timeline-time">
          {formatTime(currentTime)} / {formatTime(duration)}
        </span>
        {markMode && (
          <span className="timeline-mode">
            {draftStart == null
              ? '點選起點'
              : `起點 ${formatTime(draftStart)} · 再點終點`}
          </span>
        )}
      </div>

      <svg
        className="timeline-svg"
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="none"
        onClick={handleClick}
        role="img"
        aria-label="能量與速度曲線"
      >
        <defs>
          <linearGradient id="energyFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="rgba(232, 197, 110, 0.45)" />
            <stop offset="100%" stopColor="rgba(232, 197, 110, 0)" />
          </linearGradient>
          <linearGradient id="speedStroke" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="#c45c4a" />
            <stop offset="100%" stopColor="#e8c56e" />
          </linearGradient>
        </defs>

        <rect x="0" y="0" width={width} height={height} fill="transparent" />

        {energyPath.fill && (
          <path d={energyPath.fill} fill="url(#energyFill)" />
        )}
        {energyPath.line && (
          <path
            d={energyPath.line}
            fill="none"
            stroke="rgba(232, 197, 110, 0.9)"
            strokeWidth="2"
          />
        )}
        {speedPath.line && (
          <path
            d={speedPath.line}
            fill="none"
            stroke="url(#speedStroke)"
            strokeWidth="2.2"
          />
        )}

        {marks.map((m) => {
          const x1 = (m.start / safeDur) * width
          const x2 = (m.end / safeDur) * width
          const selected = m.id === selectedMarkId
          const base =
            m.color ||
            (m.source === 'auto' ? '#c45c4a' : '#78b4a0')
          return (
            <g
              key={m.id}
              className="mark-band"
              onClick={(ev) => {
                ev.stopPropagation()
                onSelectMark(m.id)
                onSeek(m.start)
              }}
            >
              <rect
                x={x1}
                y={8}
                width={Math.max(4, x2 - x1)}
                height={height - 16}
                fill={hexToRgba(base, selected ? 0.45 : 0.22)}
                stroke={hexToRgba(base, selected ? 0.95 : 0.8)}
                strokeWidth={selected ? 2 : 1}
                rx="3"
              />
            </g>
          )
        })}

        {draftStart != null && (
          <line
            x1={(draftStart / safeDur) * width}
            x2={(draftStart / safeDur) * width}
            y1="0"
            y2={height}
            stroke="rgba(120, 180, 160, 0.9)"
            strokeWidth="2"
            strokeDasharray="4 3"
          />
        )}

        <line
          x1={playX}
          x2={playX}
          y1="0"
          y2={height}
          stroke="rgba(255, 248, 220, 0.95)"
          strokeWidth="2"
        />
      </svg>

      <div className="timeline-legend">
        <span className="leg energy">能量</span>
        <span className="leg speed">速度</span>
        <span className="leg auto">動作類型</span>
        <span className="leg manual">手動標記</span>
      </div>
    </section>
  )
}

function toPath(
  metrics: MetricSample[],
  duration: number,
  width: number,
  height: number,
  key: 'energy' | 'speed',
  bandTop: number,
) {
  if (!metrics.length) return { line: '', fill: '' }
  const bandH = height * 0.35
  const top = height * bandTop

  const pts = metrics.map((m) => {
    const x = (m.time / duration) * width
    const y = top + bandH * (1 - Math.min(1, m[key]))
    return { x, y }
  })

  const line = pts
    .map((p, i) => `${i === 0 ? 'M' : 'L'}${p.x.toFixed(1)},${p.y.toFixed(1)}`)
    .join(' ')

  const fill = `${line} L${pts[pts.length - 1].x.toFixed(1)},${(top + bandH).toFixed(1)} L${pts[0].x.toFixed(1)},${(top + bandH).toFixed(1)} Z`

  return { line, fill }
}
