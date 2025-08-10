import { useEffect, useRef, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import '@tensorflow/tfjs-backend-webgl'
import labels from './labels.json'
import './App.css'

function App() {
  const fileInputRef = useRef<HTMLInputElement>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const detectorRef = useRef<tf.GraphModel | null>(null)
  const rafRef = useRef<number | null>(null)
  const lastVideoTimeRef = useRef<number>(-1)
  const [modelInputShape, setModelInputShape] = useState<number[]>([1, 0, 0, 3])
  const [frameAspect, setFrameAspect] = useState<string>('16 / 9')
  const [modelLoading, setModelLoading] = useState<boolean>(true)
  const [modelProgress, setModelProgress] = useState<number>(0)
  const [videoUrl, setVideoUrl] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [hasSelected, setHasSelected] = useState(false)

  const handleSelectClick = () => {
    fileInputRef.current?.click()
  }

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    setHasSelected(true)
    if (videoUrl) {
      URL.revokeObjectURL(videoUrl)
      setVideoUrl(null)
    }
    setIsLoading(true)
    const url = URL.createObjectURL(file)
    setVideoUrl(url)
  }

  useEffect(() => {
    return () => {
      if (videoUrl) URL.revokeObjectURL(videoUrl)
    }
  }, [videoUrl])

  // Initialize YOLOv8 TFJS model once
  useEffect(() => {
    let isCancelled = false
    const init = async () => {
      if (detectorRef.current) return
      try {
        await tf.ready()
        const modelUrl = `${window.location.origin}/yolov8n_web_model/model.json`
        const net = await tf.loadGraphModel(modelUrl, {
          onProgress: (fractions) => {
            setModelLoading(true)
            setModelProgress(fractions ?? 0)
          }
        })
        if (isCancelled) return
        // Warmup
        const inputShape = net.inputs[0].shape as number[]
        const dummy = tf.ones(inputShape)
        const warm = net.execute(dummy)
        setModelInputShape(inputShape)
        setModelLoading(false)
        setModelProgress(1)
        detectorRef.current = net
        tf.dispose([dummy as tf.Tensor, warm as tf.Tensor | tf.Tensor[]])
      } catch (err) {
        // eslint-disable-next-line no-console
        console.error('Failed to initialize YOLOv8 model', err)
      }
    }
    init()
    return () => {
      isCancelled = true
    }
  }, [])

  // When a video is available, start detection loop
  useEffect(() => {
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas) return

      const startWhenReady = () => {
      if (!detectorRef.current) return
      // Ensure canvas matches element size (CSS pixels)
      const resizeCanvas = () => {
        const baseW = (modelInputShape[1] as number) || 640
        const baseH = (modelInputShape[2] as number) || 640
        if (canvas.width !== baseW) canvas.width = baseW
        if (canvas.height !== baseH) canvas.height = baseH
      }
      resizeCanvas()

      const ctx = canvas.getContext('2d')!

      const drawDetections = (
        boxesData: Float32Array,
        scoresData: Float32Array,
        classesData: Float32Array,
        xRatio: number,
        yRatio: number
      ) => {
        ctx.clearRect(0, 0, canvas.width, canvas.height)

        for (let i = 0; i < scoresData.length; i++) {
          const y1 = boxesData[i * 4 + 0] * yRatio
          const x1 = boxesData[i * 4 + 1] * xRatio
          const y2 = boxesData[i * 4 + 2] * yRatio
          const x2 = boxesData[i * 4 + 3] * xRatio
          const w = Math.round(x2 - x1)
          const h = Math.round(y2 - y1)
          const x = Math.round(x1)
          const y = Math.round(y1)

          // Neon box
          ctx.lineWidth = 2
          ctx.strokeStyle = '#00fff0'
          ctx.shadowColor = 'rgba(0,255,240,0.6)'
          ctx.shadowBlur = 12
          ctx.strokeRect(x, y, w, h)

          // Corner accents
          ctx.shadowBlur = 0
          ctx.strokeStyle = '#ff00e6'
          const corner = 12
          // TL
          ctx.beginPath(); ctx.moveTo(x, y + corner); ctx.lineTo(x, y); ctx.lineTo(x + corner, y); ctx.stroke()
          // TR
          ctx.beginPath(); ctx.moveTo(x + w - corner, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + corner); ctx.stroke()
          // BL
          ctx.beginPath(); ctx.moveTo(x, y + h - corner); ctx.lineTo(x, y + h); ctx.lineTo(x + corner, y + h); ctx.stroke()
          // BR
          ctx.beginPath(); ctx.moveTo(x + w - corner, y + h); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w, y + h - corner); ctx.stroke()

          // Label
          const klassIndex = classesData[i]
          const score = scoresData[i]
          if (klassIndex !== undefined) {
            const className = labels[Math.floor(klassIndex)] ?? 'obj'
            const txt = `${className} ${(score * 100).toFixed(0)}%`
            ctx.font = '600 12px Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial'
            const paddingX = 8
            const paddingY = 4
            const textWidth = ctx.measureText(txt).width
            const boxW = textWidth + paddingX * 2
            const boxH = 18
            const bx = Math.max(0, Math.min(canvas.width - boxW, x))
            const by = Math.max(0, y - boxH - 6)
            ctx.fillStyle = 'rgba(8,12,17,0.92)'
            ctx.strokeStyle = 'rgba(0,255,240,0.55)'
            ctx.lineWidth = 1
            ctx.shadowColor = 'rgba(0,255,240,0.35)'
            ctx.shadowBlur = 8
            ctx.fillRect(bx, by, boxW, boxH)
            ctx.strokeRect(bx, by, boxW, boxH)
            ctx.shadowBlur = 0
            ctx.fillStyle = '#e6faff'
            ctx.fillText(txt, bx + paddingX, by + boxH - paddingY)
          }
        }
      }

      const loop = async () => {
        if (!detectorRef.current) return
        if (!video.paused && !video.ended) {
          // Only process when time advances
          if (video.currentTime !== lastVideoTimeRef.current) {
            lastVideoTimeRef.current = video.currentTime
            try {
              // Prepare input as in the reference implementation
              const [modelW, modelH] = [modelInputShape[1], modelInputShape[2]]
              if (!modelW || !modelH) return

              tf.engine().startScope()
              const img = tf.browser.fromPixels(video)
              const h = img.shape[0]
              const w = img.shape[1]
              const maxSize = Math.max(w, h)
              const pad = img.pad([[0, maxSize - h], [0, maxSize - w], [0, 0]]) as tf.Tensor3D
              const input = tf.image.resizeBilinear(pad, [modelW, modelH]).div(255).expandDims(0) as tf.Tensor4D
              const res = detectorRef.current.execute(input) as tf.Tensor
              const trans = (res as tf.Tensor).transpose([0, 2, 1]) as tf.Tensor3D
              const boxes = tf.tidy(() => {
                const ww = trans.slice([0, 0, 2], [-1, -1, 1]) as tf.Tensor3D
                const hh = trans.slice([0, 0, 3], [-1, -1, 1]) as tf.Tensor3D
                const x1 = tf.sub(trans.slice([0, 0, 0], [-1, -1, 1]) as tf.Tensor, tf.div(ww, 2))
                const y1 = tf.sub(trans.slice([0, 0, 1], [-1, -1, 1]) as tf.Tensor, tf.div(hh, 2))
                return tf
                  .concat([y1, x1, tf.add(y1, hh), tf.add(x1, ww)], 2)
                  .squeeze() as tf.Tensor2D
              })
              // rawScores shape [num, numClasses]
              const rawScores = trans.slice([0, 0, 4], [-1, -1, -1]).squeeze([0]) as tf.Tensor2D
              const scores = rawScores.max(1) as tf.Tensor1D
              const classes = rawScores.argMax(1) as tf.Tensor1D
              const nms = await tf.image.nonMaxSuppressionAsync(boxes as tf.Tensor2D, scores as tf.Tensor1D, 500, 0.45, 0.2)
              const boxesData = (boxes as tf.Tensor2D).gather(nms, 0).dataSync() as Float32Array
              const scoresData = (scores as tf.Tensor1D).gather(nms, 0).dataSync() as Float32Array
              const classesData = (classes as tf.Tensor1D).gather(nms, 0).dataSync() as Float32Array
              const xRatio = Math.max(w, h) / w
              const yRatio = Math.max(w, h) / h
              drawDetections(boxesData, scoresData, classesData, xRatio, yRatio)
              tf.dispose([res, trans, boxes, scores, classes, nms, input, pad, img])
              tf.engine().endScope()
            } catch (e) {
              // eslint-disable-next-line no-console
              console.warn('detectForVideo failed', e)
            }
          }
        }
        rafRef.current = requestAnimationFrame(loop)
      }

      if (!rafRef.current) rafRef.current = requestAnimationFrame(loop)

      const onResize = () => {
        resizeCanvas()
      }
      window.addEventListener('resize', onResize)

      return () => {
        window.removeEventListener('resize', onResize)
      }
    }

    let cleanupResize: (() => void) | undefined

    if (videoUrl) {
      // Start once metadata is loaded (videoWidth/Height available)
      const onLoaded = () => {
        // Match stage aspect to the actual video to avoid letterboxing offsets
        if (video.videoWidth && video.videoHeight) {
          setFrameAspect(`${video.videoWidth} / ${video.videoHeight}`)
        }
        const cleanup = startWhenReady()
        if (cleanup) cleanupResize = cleanup
      }
      if (video.readyState >= 1) {
        const cleanup = startWhenReady()
        if (cleanup) cleanupResize = cleanup
      } else {
        video.addEventListener('loadedmetadata', onLoaded, { once: true })
      }
    }

    return () => {
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current)
        rafRef.current = null
      }
      lastVideoTimeRef.current = -1
      // Clear overlay
      const ctx = canvas.getContext('2d')
      ctx?.clearRect(0, 0, canvas.width, canvas.height)
      if (cleanupResize) cleanupResize()
    }
  }, [videoUrl])

  // Cleanup detector on unmount
  useEffect(() => {
    return () => {
      if (detectorRef.current) {
        detectorRef.current = null
      }
    }
  }, [])

  return (
    <div className="neo-app">
      {modelLoading && (
        <div className="model-loader" role="status" aria-live="polite">
          <div className="spinner" />
          <span>Loading model... {(modelProgress * 100).toFixed(0)}%</span>
      </div>
      )}
      <div className="bg-grid" aria-hidden />

      <header className="neo-header">
        <h1 className="title">
          <span className="accent">NEON</span> PLAYER
        </h1>
        <p className="subtitle">Load a video from your system and play it in a cyberpunk frame.</p>
      </header>

      <div className="controls">
        <button className="neo-button" onClick={handleSelectClick} aria-label="Select video">
          <span className="icon" aria-hidden>
            <svg viewBox="0 0 24 24" fill="none">
              <path
                d="M4 5h16a1 1 0 0 1 1 1v12a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1ZM4 9h16M4 15h16M8 5v14M16 5v14"
                stroke="currentColor"
                strokeWidth="1.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </span>
          <span>Select Video</span>
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept="video/*"
          className="hidden-input"
          onChange={handleFileChange}
        />
      </div>

      <section className="stage">
        {!hasSelected && (
          <div className="placeholder">
            <p>Awaiting input...</p>
            <p className="hint">Press Select Video to boot the feed</p>
          </div>
        )}

        {hasSelected && (
          <div className={`video-frame ${isLoading ? 'loading' : ''}`} style={{ ['--frame-aspect' as any]: frameAspect }}>
            {videoUrl && (
              <video
                ref={videoRef}
                src={videoUrl}
                className="video"
                controls
                onLoadedData={() => setIsLoading(false)}
              />
            )}
            <canvas ref={canvasRef} className="overlay-canvas" aria-hidden />
            <div className="scanlines" aria-hidden />
            {isLoading && (
              <div className="loader">
                <div className="spinner" />
                <span>Booting video...</span>
              </div>
            )}
            <div className="corners" aria-hidden>
              <span />
              <span />
              <span />
              <span />
            </div>
          </div>
        )}
      </section>
    </div>
  )
}

export default App
