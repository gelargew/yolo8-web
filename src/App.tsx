import { useEffect, useRef, useState } from 'react'
import labels from './labels.json'
import { Upload } from 'lucide-react'
import PlabsLogo from './plabs'
import { loadYoloModel, startVideoDetection } from './detection/yolo'
const PERSON_INDEX = labels.indexOf('person')
import './App.css'

function App() {
  const fileInputRef = useRef<HTMLInputElement>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const counterRef = useRef<HTMLDivElement>(null)
  const modelRef = useRef<{ net: any, inputShape: number[] } | null>(null)
  const rafRef = useRef<number | null>(null)
  const lastVideoTimeRef = useRef<number>(-1)
  const [modelInputShape, setModelInputShape] = useState<number[]>([1, 0, 0, 3])
  const [frameAspect, setFrameAspect] = useState<string>('16 / 9')
  const [modelLoading, setModelLoading] = useState<boolean>(true)
  const [modelProgress, setModelProgress] = useState<number>(0)
  const [videoUrl, setVideoUrl] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [hasSelected, setHasSelected] = useState(false)
  const [demoSelected, setDemoSelected] = useState<string>('')

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

  const handleDemoChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const val = e.target.value
    setDemoSelected(val)
    if (!val) return
    setHasSelected(true)
    setIsLoading(true)
    // demos are served from public/demo
    setVideoUrl(`/demo/${val}`)
  }

  const togglePlay = () => {
    const v = videoRef.current
    if (!v) return
    if (v.paused) v.play()
    else v.pause()
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
      if (modelRef.current) return
      try {
        const modelUrl = `${window.location.origin}/yolov8n_web_model/model.json`
        const loaded = await loadYoloModel(modelUrl, (f) => {
          setModelLoading(true)
          setModelProgress(f ?? 0)
        })
        if (isCancelled) return
        setModelInputShape(loaded.inputShape)
        setModelLoading(false)
        setModelProgress(1)
        modelRef.current = loaded
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
      if (!modelRef.current) return
      const stop = startVideoDetection({
        video,
        canvas,
        model: modelRef.current,
        classFilterIndex: PERSON_INDEX,
        onFrameCount: (count) => {
          if (counterRef.current) counterRef.current.textContent = String(count)
        }
      })
      return () => stop()
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
      const ctx = canvas.getContext('2d')
      ctx?.clearRect(0, 0, canvas.width, canvas.height)
      if (cleanupResize) cleanupResize()
    }
  }, [videoUrl])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      modelRef.current = null
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

      <div className="main-wrap">
        <div className="neo-grid">
        <div className="left-pane">
          <div className="left-controls">
            <button className="neo-button" onClick={handleSelectClick} aria-label="Select video">
              <span className="icon" aria-hidden>
                <Upload size={20} />
              </span>
              <span>Upload Video</span>
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept="video/*"
              className="hidden-input"
              onChange={handleFileChange}
            />

            <div className="select-wrap">
              <select className="neo-select" value={demoSelected} onChange={handleDemoChange} aria-label="Select demo video">
                <option value="">Select demo</option>
              <option value="mall.mp4">Mall</option>
              <option value="mall2.mp4">Mall 2</option>
              <option value="hall.mp4">Hall</option>
              <option value="street.mp4">Street</option>
              </select>
              <span className="chevron" aria-hidden>
                <svg viewBox="0 0 24 24" fill="none">
                  <path d="M6 9l6 6 6-6" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </span>
            </div>
          </div>

          <section className="stage">
            {!hasSelected && (
              <div className="placeholder">
                <p>Awaiting input...</p>
                <p className="hint">Upload or choose a demo to start</p>
              </div>
            )}

            {hasSelected && (
              <div className={`video-frame ${isLoading ? 'loading' : ''}`} style={{ ['--frame-aspect' as any]: frameAspect }}>
                {videoUrl && (
                  <video
                    ref={videoRef}
                    src={videoUrl}
                    className="video"
                    playsInline
                    muted
                    autoPlay
                    onClick={togglePlay}
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

        <aside className="right-pane">
          <div className="brand">
            <h2 className="brand-title">Realtime crowd detection model</h2>
          </div>
          <div className="hud-counter large" aria-live="polite">
            <span className="label">count</span>
            <span className="value" ref={counterRef}>0</span>
          </div>
        </aside>
        </div>
      </div>

    </div>
  )
}

export default App
