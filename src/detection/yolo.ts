import * as tf from '@tensorflow/tfjs'
import '@tensorflow/tfjs-backend-webgl'
import labels from '../labels.json'

export type LoadedModel = {
  net: tf.GraphModel
  inputShape: number[]
}

export async function loadYoloModel(modelUrl: string, onProgress?: (p: number) => void): Promise<LoadedModel> {
  await tf.ready()
  const net = await tf.loadGraphModel(modelUrl, {
    onProgress: f => onProgress?.(f ?? 0)
  })
  const inputShape = net.inputs[0].shape as number[]
  // Warmup
  const dummy = tf.ones(inputShape)
  const warm = net.execute(dummy)
  tf.dispose([dummy as tf.Tensor, warm as tf.Tensor | tf.Tensor[]])
  return { net, inputShape }
}

export type StartVideoDetectionOptions = {
  video: HTMLVideoElement
  canvas: HTMLCanvasElement
  model: LoadedModel
  classFilterIndex?: number
  onFrameCount?: (count: number) => void
}

export function startVideoDetection(options: StartVideoDetectionOptions): () => void {
  const { video, canvas, model, classFilterIndex, onFrameCount } = options
  const ctx = canvas.getContext('2d')!

  // Fix canvas to model input dimensions to mirror preprocessing scale
  const modelW = (model.inputShape[1] as number) || 640
  const modelH = (model.inputShape[2] as number) || 640
  if (canvas.width !== modelW) canvas.width = modelW
  if (canvas.height !== modelH) canvas.height = modelH

  let rafId: number | null = null
  let lastVideoTime = -1

  const loop = async () => {
    if (video.paused || video.ended) {
      rafId = requestAnimationFrame(loop)
      return
    }

    if (video.currentTime !== lastVideoTime) {
      lastVideoTime = video.currentTime
      try {
        tf.engine().startScope()
        const img = tf.browser.fromPixels(video)
        const h = img.shape[0]
        const w = img.shape[1]
        const maxSize = Math.max(w, h)
        const pad = img.pad([[0, maxSize - h], [0, maxSize - w], [0, 0]]) as tf.Tensor3D
        const input = tf.image.resizeBilinear(pad, [modelW, modelH]).div(255).expandDims(0) as tf.Tensor4D
        const res = model.net.execute(input) as tf.Tensor
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
        const rawScores = trans.slice([0, 0, 4], [-1, -1, -1]).squeeze([0]) as tf.Tensor2D
        const scores = rawScores.max(1) as tf.Tensor1D
        const classes = rawScores.argMax(1) as tf.Tensor1D
        const nms = await tf.image.nonMaxSuppressionAsync(boxes as tf.Tensor2D, scores as tf.Tensor1D, 500, 0.45, 0.2)
        const boxesData = (boxes as tf.Tensor2D).gather(nms, 0).dataSync() as Float32Array
        const scoresData = (scores as tf.Tensor1D).gather(nms, 0).dataSync() as Float32Array
        const classesData = (classes as tf.Tensor1D).gather(nms, 0).dataSync() as Float32Array

        // Convert padded-square coords back to original pixels
        const xRatio = Math.max(w, h) / w
        const yRatio = Math.max(w, h) / h

        draw(ctx, canvas, boxesData, scoresData, classesData, xRatio, yRatio, classFilterIndex)

        if (onFrameCount) {
          // Count only filtered class if provided, else total
          let count = 0
          for (let i = 0; i < scoresData.length; i++) {
            if (classFilterIndex === undefined || Math.floor(classesData[i]) === classFilterIndex) count++
          }
          onFrameCount(count)
        }

        tf.dispose([res, trans, boxes, scores, classes, nms, input, pad, img])
        tf.engine().endScope()
      } catch (e) {
        // eslint-disable-next-line no-console
        console.warn('yolo loop error', e)
      }
    }

    rafId = requestAnimationFrame(loop)
  }

  if (!rafId) rafId = requestAnimationFrame(loop)

  return () => {
    if (rafId) cancelAnimationFrame(rafId)
    rafId = null
    ctx.clearRect(0, 0, canvas.width, canvas.height)
  }
}

function draw(
  ctx: CanvasRenderingContext2D,
  canvas: HTMLCanvasElement,
  boxesData: Float32Array,
  scoresData: Float32Array,
  classesData: Float32Array,
  xRatio: number,
  yRatio: number,
  classFilterIndex?: number
) {
  ctx.clearRect(0, 0, canvas.width, canvas.height)

  for (let i = 0; i < scoresData.length; i++) {
    const klassIndex = Math.floor(classesData[i])
    if (classFilterIndex !== undefined && klassIndex !== classFilterIndex) continue

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
    ctx.beginPath(); ctx.moveTo(x, y + corner); ctx.lineTo(x, y); ctx.lineTo(x + corner, y); ctx.stroke()
    ctx.beginPath(); ctx.moveTo(x + w - corner, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + corner); ctx.stroke()
    ctx.beginPath(); ctx.moveTo(x, y + h - corner); ctx.lineTo(x, y + h); ctx.lineTo(x + corner, y + h); ctx.stroke()
    ctx.beginPath(); ctx.moveTo(x + w - corner, y + h); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w, y + h - corner); ctx.stroke()

    // Label
    const score = scoresData[i]
    const className = labels[klassIndex] ?? 'obj'
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


