type Props = {
  onLocalFile: (file: File) => void
  youtubeUrl: string
  onYoutubeUrlChange: (v: string) => void
  onYoutubeSubmit: () => void
  youtubeBusy: boolean
  youtubeError: string | null
}

export function ImportPanel({
  onLocalFile,
  youtubeUrl,
  onYoutubeUrlChange,
  onYoutubeSubmit,
  youtubeBusy,
  youtubeError,
}: Props) {
  return (
    <section className="import-panel" aria-label="匯入">
      <label className="btn btn-primary file-btn">
        開啟本機影片
        <input
          type="file"
          accept="video/*"
          hidden
          onChange={(e) => {
            const f = e.target.files?.[0]
            if (f) onLocalFile(f)
            e.target.value = ''
          }}
        />
      </label>

      <form
        className="yt-form"
        onSubmit={(e) => {
          e.preventDefault()
          onYoutubeSubmit()
        }}
      >
        <input
          className="yt-input"
          type="url"
          placeholder="貼上 YouTube 連結…"
          value={youtubeUrl}
          onChange={(e) => onYoutubeUrlChange(e.target.value)}
          disabled={youtubeBusy}
        />
        <button className="btn btn-ghost" type="submit" disabled={youtubeBusy}>
          {youtubeBusy ? '下載中…' : '匯入'}
        </button>
      </form>

      {youtubeError && <p className="yt-error">{youtubeError}</p>}
    </section>
  )
}
