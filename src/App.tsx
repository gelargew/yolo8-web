import { useEffect, useRef, useState } from 'react'
import labels from './labels.json'
import { Upload } from 'lucide-react'
import PlabsLogo from './plabs'
import { loadPoseModel, startPoseVideoDetection } from './detection/pose'
const PERSON_INDEX = labels.indexOf('person')
import './App.css'


const MODELS = {
  object: '/yolov8n_web_model/model.json'
,
pose: '/pose_model/model.json'
}

const DEBUG = false

function App() {
  const fileInputRef = useRef<HTMLInputElement>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const counterRef = useRef<HTMLDivElement>(null)
  const workingRef = useRef<HTMLSpanElement>(null)
  const idleRef = useRef<HTMLSpanElement>(null)
  const modelRef = useRef<{ net: any, inputShape: number[] } | null>(null)
  const rafRef = useRef<number | null>(null)
  const lastVideoTimeRef = useRef<number>(-1)
  const webcamStreamRef = useRef<MediaStream | null>(null)
  const [modelInputShape, setModelInputShape] = useState<number[]>([1, 0, 0, 3])
  const [frameAspect, setFrameAspect] = useState<string>('16 / 9')
  const [modelLoading, setModelLoading] = useState<boolean>(true)
  const [modelProgress, setModelProgress] = useState<number>(0)
  const [videoUrl, setVideoUrl] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [hasSelected, setHasSelected] = useState(false)
  const [demoSelected, setDemoSelected] = useState<string>('')
  const [webcamActive, setWebcamActive] = useState<boolean>(false)

  const handleSelectClick = () => fileInputRef.current?.click()

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    setHasSelected(true)
    if (webcamActive) stopWebcam()
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
    if (webcamActive) stopWebcam()
    setHasSelected(true)
    setIsLoading(true)
    // demos are served from public/demo
    setVideoUrl(`/demo/${val}`)
  }

  const startWebcam = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false })
      webcamStreamRef.current = stream
      setWebcamActive(true)
      const v = videoRef.current
      if (v) {
        ;(v as HTMLVideoElement & { srcObject: MediaStream | null }).srcObject = stream
        const onLoaded = () => {
          if (v.videoWidth && v.videoHeight) {
            setFrameAspect(`${v.videoWidth} / ${v.videoHeight}`)
          }
          setIsLoading(false)
          v.play()
        }
        if (v.readyState >= 1) onLoaded()
        else v.addEventListener('loadedmetadata', onLoaded, { once: true })
      } else {
        // video ref will attach in effect
      }
    } catch (err) {
      // eslint-disable-next-line no-console
      console.error('webcam error', err)
      setIsLoading(false)
      setWebcamActive(false)
    }
  }

  const stopWebcam = () => {
    const stream = webcamStreamRef.current
    const v = videoRef.current
    if (stream) {
      stream.getTracks().forEach(t => t.stop())
      webcamStreamRef.current = null
    }
    if (v) (v as HTMLVideoElement & { srcObject: MediaStream | null }).srcObject = null
    setWebcamActive(false)
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

  // Initialize Pose TFJS model once
  useEffect(() => {
    let isCancelled = false
    const init = async () => {
      if (modelRef.current) return
      try {
        const modelUrl = `${window.location.origin}${MODELS.pose}`
        const loaded = await loadPoseModel(modelUrl, (f) => {
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
        console.error('Failed to initialize Pose model', err)
      }
    }
    init()
    return () => {
      isCancelled = true
    }
  }, [])

  // When a source is available, start detection loop
  useEffect(() => {
    let cleanupDetection: (() => void) | undefined
    let cleanupVideoListeners: (() => void) | undefined
    let retryId: number | undefined

    const tryInit = () => {
      const video = videoRef.current
      const canvas = canvasRef.current
      if (!video || !canvas) {
        retryId = window.setTimeout(tryInit, 50)
        return
      }

      const startWhenReady = () => {
        if (!modelRef.current) return
        const stop = startPoseVideoDetection({
          video,
          canvas,
          model: modelRef.current,
          onFrameCount: (count) => {
            if (counterRef.current) counterRef.current.textContent = String(count)
          },
          onFrameStatusCounts: (working, idle) => {
            if (workingRef.current) workingRef.current.textContent = String(working)
            if (idleRef.current) idleRef.current.textContent = String(idle)
          }
        })
        return () => stop()
      }

      if (videoUrl || webcamActive) {
        const onLoaded = () => {
          if (video.videoWidth && video.videoHeight) {
            setFrameAspect(`${video.videoWidth} / ${video.videoHeight}`)
          }
          setIsLoading(false)
          cleanupDetection = startWhenReady()
        }
        if (video.readyState >= 1) {
          cleanupDetection = startWhenReady()
        } else {
          video.addEventListener('loadedmetadata', onLoaded, { once: true })
        }

        const onError = (ev: Event) => {
          console.error('[video] error', ev)
        }
        const onStalled = () => {}
        const onWaiting = () => {}
        const onCanPlay = () => {}
        const onPlaying = () => {
          setIsLoading(false)
        }
        video.addEventListener('error', onError)
        video.addEventListener('stalled', onStalled)
        video.addEventListener('waiting', onWaiting)
        video.addEventListener('canplay', onCanPlay)
        video.addEventListener('playing', onPlaying)
        cleanupVideoListeners = (() => {
          video.removeEventListener('error', onError)
          video.removeEventListener('stalled', onStalled)
          video.removeEventListener('waiting', onWaiting)
          video.removeEventListener('canplay', onCanPlay)
          video.removeEventListener('playing', onPlaying)
        })
      }
    }

    tryInit()

    return () => {
      if (retryId) window.clearTimeout(retryId)
      const video = videoRef.current
      const canvas = canvasRef.current
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current)
        rafRef.current = null
      }
      lastVideoTimeRef.current = -1
      if (canvas) {
        const ctx = canvas.getContext('2d')
        ctx?.clearRect(0, 0, canvas.width, canvas.height)
      }
      if (cleanupDetection) cleanupDetection()
      if (cleanupVideoListeners) cleanupVideoListeners()
      if (DEBUG) console.log('[detect] cleanup complete')
    }
  }, [videoUrl, webcamActive])

  // Ensure stream attaches if video element ref becomes available slightly later
  useEffect(() => {
    if (!webcamActive) return
    const v = videoRef.current
    const stream = webcamStreamRef.current
    if (v && stream) {
      if (!(v as any).srcObject) {
        ;(v as HTMLVideoElement & { srcObject: MediaStream | null }).srcObject = stream
        if (DEBUG) console.log('[webcam] srcObject attached in effect', v.readyState)
      }
      const onLoaded = () => {
        if (DEBUG) console.log('[webcam] (effect) loadedmetadata', v.videoWidth, v.videoHeight)
        if (v.videoWidth && v.videoHeight) setFrameAspect(`${v.videoWidth} / ${v.videoHeight}`)
        setIsLoading(false)
        v.play().catch(() => {/* ignore */})
      }
      if (v.readyState >= 1) onLoaded()
      else v.addEventListener('loadedmetadata', onLoaded, { once: true })
      return () => v.removeEventListener('loadedmetadata', onLoaded)
    }
  }, [webcamActive])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      modelRef.current = null
      stopWebcam()
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

            <div className="left-actions-row">
              <button
                className={`neo-button ${webcamActive ? 'active' : ''}`}
                onClick={() => {
                  if (webcamActive) {
                    stopWebcam()
                    setHasSelected(false)
                  } else {
                    setHasSelected(true)
                    setIsLoading(true)
                    if (videoUrl) {
                      URL.revokeObjectURL(videoUrl)
                      setVideoUrl(null)
                    }
                    startWebcam()
                  }
                }}
                aria-label={webcamActive ? 'Stop webcam' : 'Start webcam'}
              >
                {webcamActive ? 'Stop Webcam' : 'Start Webcam'}
              </button>

              <div className="select-wrap">
                <select className="neo-select" value={demoSelected} onChange={handleDemoChange} aria-label="Select demo video">
                  <option value="">Select demo</option>
                  <option value="idle.mp4">Idle</option>
                  <option value="working.mp4">Working</option>
                </select>
                <span className="chevron" aria-hidden>
                  <svg viewBox="0 0 24 24" fill="none">
                    <path d="M6 9l6 6 6-6" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                </span>
              </div>
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
                <video
                  ref={videoRef}
                  src={webcamActive ? undefined : videoUrl ?? undefined}
                  className="video"
                  playsInline
                  muted
                  autoPlay
                  onClick={togglePlay}
                  onLoadedData={() => setIsLoading(false)}
                />
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
            <h2 className="brand-title">Behavior detection model</h2>
          </div>
          <div className="hud-counter" aria-live="polite">
            <span className="label">total</span>
            <span className="value" ref={counterRef}>0</span>
          </div>
          <div className="hud-counter" aria-live="polite">
            <span className="label">working</span>
            <span className="value" ref={workingRef}>0</span>
          </div>
          <div className="hud-counter" aria-live="polite">
            <span className="label">idle</span>
            <span className="value" ref={idleRef}>0</span>
          </div>
          <p className="tagline"><span>copyright ©2025 by</span>
          <PlabsLogo className="logo" /></p>
        </aside>
        </div>
      </div>

    </div>
  )
}

export default App
