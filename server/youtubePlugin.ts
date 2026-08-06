import { spawn } from 'node:child_process'
import { createReadStream, existsSync, mkdirSync, readdirSync, unlinkSync } from 'node:fs'
import { tmpdir } from 'node:os'
import path from 'node:path'
import type { Plugin } from 'vite'

const CACHE = path.join(tmpdir(), 'dance-analyser-yt')

function isYouTubeUrl(url: string) {
  try {
    const u = new URL(url)
    return (
      u.hostname.includes('youtube.com') ||
      u.hostname.includes('youtu.be') ||
      u.hostname.includes('youtube-nocookie.com')
    )
  } catch {
    return false
  }
}

function runYtDlp(url: string, outTemplate: string) {
  return new Promise<void>((resolve, reject) => {
    const args = [
      '-f',
      'bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b',
      '--merge-output-format',
      'mp4',
      '-o',
      outTemplate,
      '--no-playlist',
      '--max-filesize',
      '200M',
      url,
    ]
    const child = spawn('yt-dlp', args, { shell: true })
    let err = ''
    child.stderr.on('data', (d) => {
      err += String(d)
    })
    child.on('error', (e) => {
      reject(
        new Error(
          '找不到 yt-dlp。請先安裝：https://github.com/yt-dlp/yt-dlp ，然後再試。',
        ),
      )
      void e
    })
    child.on('close', (code) => {
      if (code === 0) resolve()
      else
        reject(
          new Error(
            err.trim() ||
              'YouTube 下載失敗。請檢查連結，或改用本機影片。',
          ),
        )
    })
  })
}

export function youtubeApiPlugin(): Plugin {
  return {
    name: 'dance-analyser-youtube-api',
    configureServer(server) {
      server.middlewares.use(async (req, res, next) => {
        if (req.method !== 'POST' || req.url !== '/api/youtube') {
          next()
          return
        }

        try {
          const chunks: Buffer[] = []
          for await (const chunk of req) {
            chunks.push(Buffer.from(chunk))
          }
          const body = JSON.parse(Buffer.concat(chunks).toString('utf8')) as {
            url?: string
          }
          const url = body.url?.trim()
          if (!url || !isYouTubeUrl(url)) {
            res.statusCode = 400
            res.setHeader('Content-Type', 'application/json')
            res.end(JSON.stringify({ error: '請提供有效的 YouTube 連結。' }))
            return
          }

          if (!existsSync(CACHE)) mkdirSync(CACHE, { recursive: true })
          const id = `yt_${Date.now()}`
          const outTemplate = path.join(CACHE, `${id}.%(ext)s`)
          await runYtDlp(url, outTemplate)

          const file = readdirSync(CACHE).find((f) => f.startsWith(id))
          if (!file) {
            res.statusCode = 500
            res.setHeader('Content-Type', 'application/json')
            res.end(JSON.stringify({ error: '下載完成但找不到檔案。' }))
            return
          }

          const full = path.join(CACHE, file)
          res.statusCode = 200
          res.setHeader('Content-Type', 'video/mp4')
          res.setHeader(
            'Content-Disposition',
            `inline; filename="${encodeURIComponent(file)}"`,
          )
          const stream = createReadStream(full)
          stream.pipe(res)
          stream.on('close', () => {
            try {
              unlinkSync(full)
            } catch {
              /* ignore */
            }
          })
        } catch (err) {
          res.statusCode = 500
          res.setHeader('Content-Type', 'application/json')
          res.end(
            JSON.stringify({
              error:
                err instanceof Error
                  ? err.message
                  : 'YouTube 匯入失敗。',
            }),
          )
        }
      })
    },
  }
}
