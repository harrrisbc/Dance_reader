type Props = {
  videoRef: React.RefObject<HTMLVideoElement | null>
  canvasRef: React.RefObject<HTMLCanvasElement | null>
  src: string | null
  onLoadedMetadata: () => void
  onTimeUpdate: () => void
  hasMedia: boolean
}

export function Stage({
  videoRef,
  canvasRef,
  src,
  onLoadedMetadata,
  onTimeUpdate,
  hasMedia,
}: Props) {
  return (
    <section className="stage" aria-label="舞台">
      <div className="stage-frame">
        <div className="stage-glow" aria-hidden />
        {hasMedia && src ? (
          <>
            <video
              ref={videoRef}
              className="stage-video"
              src={src}
              playsInline
              onLoadedMetadata={onLoadedMetadata}
              onTimeUpdate={onTimeUpdate}
            />
            <canvas ref={canvasRef} className="stage-skeleton" />
          </>
        ) : (
          <div className="stage-empty">
            <p className="stage-empty-brand">Dance Analyser</p>
            <p className="stage-empty-hint">
              開啟本機影片，或貼上 YouTube 連結，燈光就會亮起。
            </p>
          </div>
        )}
        <div className="stage-vignette" aria-hidden />
      </div>
    </section>
  )
}
