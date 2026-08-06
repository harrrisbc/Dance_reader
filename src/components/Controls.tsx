type Props = {
  hasMedia: boolean
  playing: boolean
  analyzing: boolean
  progress: number
  progressLabel?: string
  skeletonOn: boolean
  markMode: boolean
  selectedMarkId: string | null
  actors: { id: string; label: string }[]
  selectedActorId: string // 'all' | track id
  onSelectActor: (id: string) => void
  onTogglePlay: () => void
  onAnalyze: () => void
  onToggleSkeleton: () => void
  onToggleMarkMode: () => void
  onDeleteMark: () => void
  onSeekRelative: (delta: number) => void
}

export function Controls({
  hasMedia,
  playing,
  analyzing,
  progress,
  progressLabel,
  skeletonOn,
  markMode,
  selectedMarkId,
  actors,
  selectedActorId,
  onSelectActor,
  onTogglePlay,
  onAnalyze,
  onToggleSkeleton,
  onToggleMarkMode,
  onDeleteMark,
  onSeekRelative,
}: Props) {
  return (
    <section className="controls" aria-label="操控">
      <div className="controls-row">
        <button
          className="btn btn-ghost"
          type="button"
          disabled={!hasMedia}
          onClick={() => onSeekRelative(-2)}
        >
          −2 秒
        </button>
        <button
          className="btn btn-primary"
          type="button"
          disabled={!hasMedia}
          onClick={onTogglePlay}
        >
          {playing ? '暫停' : '播放'}
        </button>
        <button
          className="btn btn-ghost"
          type="button"
          disabled={!hasMedia}
          onClick={() => onSeekRelative(2)}
        >
          +2 秒
        </button>
      </div>

      <div className="controls-row">
        <button
          className="btn btn-accent"
          type="button"
          disabled={!hasMedia || analyzing}
          onClick={onAnalyze}
        >
          {analyzing
            ? `${progressLabel || '分析中'} ${Math.round(progress * 100)}%`
            : '分析骨架／重複動作'}
        </button>
        <button
          className={`btn ${skeletonOn ? 'btn-primary' : 'btn-ghost'}`}
          type="button"
          disabled={!hasMedia}
          onClick={onToggleSkeleton}
        >
          骨架 {skeletonOn ? '開' : '關'}
        </button>
        <button
          className={`btn ${markMode ? 'btn-primary' : 'btn-ghost'}`}
          type="button"
          disabled={!hasMedia}
          onClick={onToggleMarkMode}
        >
          {markMode ? '結束標記' : '手動標記'}
        </button>
        <button
          className="btn btn-ghost"
          type="button"
          disabled={!selectedMarkId}
          onClick={onDeleteMark}
        >
          刪除標記
        </button>
      </div>

      {actors.length > 0 && (
        <div className="controls-row actor-row" role="group" aria-label="舞者選擇">
          <span className="actor-label">舞者</span>
          <button
            type="button"
            className={`btn btn-sm ${selectedActorId === 'all' ? 'btn-primary' : 'btn-ghost'}`}
            onClick={() => onSelectActor('all')}
          >
            全部
          </button>
          {actors.map((a) => (
            <button
              key={a.id}
              type="button"
              className={`btn btn-sm ${selectedActorId === a.id ? 'btn-primary' : 'btn-ghost'}`}
              onClick={() => onSelectActor(a.id)}
            >
              {a.label}
            </button>
          ))}
        </div>
      )}
    </section>
  )
}
