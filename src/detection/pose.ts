import * as tf from '@tensorflow/tfjs'
import '@tensorflow/tfjs-backend-webgl'

export type LoadedModel = {
  net: tf.GraphModel
  inputShape: number[]
}

// Reuse generic loader; pose model is a standard TFJS GraphModel
export async function loadPoseModel(modelUrl: string, onProgress?: (p: number) => void): Promise<LoadedModel> {
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

export type StartPoseVideoDetectionOptions = {
  video: HTMLVideoElement
  canvas: HTMLCanvasElement
  model: LoadedModel
  onFrameCount?: (count: number) => void
  onFrameStatusCounts?: (working: number, idle: number) => void
}

// YOLOv8-pose for COCO has 17 keypoints with (x, y, conf)
const NUM_KEYPOINTS = 17
const KPT_STRIDE = 3 // x, y, conf
const NUM_CLASSES = 1 // pose model here is trained for 'person' only

export function startPoseVideoDetection(options: StartPoseVideoDetectionOptions): () => void {
  const { video, canvas, model, onFrameCount } = options
  const ctx = canvas.getContext('2d')!


  const modelW = (model.inputShape[1] as number) || 640
  const modelH = (model.inputShape[2] as number) || 640
  if (canvas.width !== modelW) canvas.width = modelW
  if (canvas.height !== modelH) canvas.height = modelH

  let rafId: number | null = null
  let lastVideoTime = -1

  type WristPoint = [number, number]
  type TrackSample = { t: number; lw?: WristPoint; rw?: WristPoint }
  type Track = {
    id: number
    lastBox: { x1: number; y1: number; x2: number; y2: number }
    lastCenter: [number, number]
    lastUpdateTime: number
    lastSampleTime: number
    samples: TrackSample[]
    status: 'working' | 'idle'
  }
  const tracks = new Map<number, Track>()
  let nextTrackId = 1

  const SAMPLE_INTERVAL_S = 0.5
  const WINDOW_S = 3.0
  const KP_CONF_THRESHOLD = 0.4

  const loop = async () => {
    if (video.paused || video.ended) {
      // eslint-disable-next-line no-console
      // console.log('[pose] paused/ended, scheduling next frame')
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
        const input = tf.image
          .resizeBilinear(pad, [modelW, modelH])
          .div(255)
          .expandDims(0) as tf.Tensor4D

        const res = model.net.execute(input) as tf.Tensor
        const trans = (res as tf.Tensor).transpose([0, 2, 1]) as tf.Tensor3D // [1, num_preds, D]

        // Boxes (xywh -> yxyx)
        const boxes = tf.tidy(() => {
          const ww = trans.slice([0, 0, 2], [-1, -1, 1]) as tf.Tensor3D
          const hh = trans.slice([0, 0, 3], [-1, -1, 1]) as tf.Tensor3D
          const x1 = tf.sub(trans.slice([0, 0, 0], [-1, -1, 1]) as tf.Tensor, tf.div(ww, 2))
          const y1 = tf.sub(trans.slice([0, 0, 1], [-1, -1, 1]) as tf.Tensor, tf.div(hh, 2))
          return tf
            .concat([y1, x1, tf.add(y1, hh), tf.add(x1, ww)], 2)
            .squeeze() as tf.Tensor2D // [num_preds, 4]
        })

        // Class scores limited to NUM_CLASSES, exclude keypoints block
        const classStart = 4
        const kptStart = classStart + NUM_CLASSES
        const rawClass = trans.slice([0, 0, classStart], [-1, -1, NUM_CLASSES]).squeeze([0]) as tf.Tensor2D // [num_preds, C]
        const scores = rawClass.max(1) as tf.Tensor1D
        const classes = rawClass.argMax(1) as tf.Tensor1D

        // Keypoints block [num_preds, 51]
        const kptsAll = trans
          .slice([0, 0, kptStart], [-1, -1, NUM_KEYPOINTS * KPT_STRIDE])
          .squeeze([0]) as tf.Tensor2D

        // NMS on boxes using class score
        const nms = await tf.image.nonMaxSuppressionAsync(
          boxes as tf.Tensor2D,
          scores as tf.Tensor1D,
          300,
          0.45,
          0.25
        )

        const boxesTensor = (boxes as tf.Tensor2D).gather(nms, 0)
        const scoresTensor = (scores as tf.Tensor1D).gather(nms, 0)
        const classesTensor = (classes as tf.Tensor1D).gather(nms, 0)
        const kptsTensor = (kptsAll as tf.Tensor2D).gather(nms, 0)

        const boxesData = boxesTensor.dataSync() as Float32Array
        const scoresData = scoresTensor.dataSync() as Float32Array
        const classesData = classesTensor.dataSync() as Float32Array
        const kptsData = kptsTensor.dataSync() as Float32Array


        const xRatio = Math.max(w, h) / w
        const yRatio = Math.max(w, h) / h

        // Build detection objects in canvas space
        type Det = {
          idx: number
          box: { x1: number; y1: number; x2: number; y2: number }
          center: [number, number]
          score: number
          lw?: WristPoint
          rw?: WristPoint
        }
        const dets: Det[] = []
        const num = scoresData.length
        for (let i = 0; i < num; i++) {
          const y1 = boxesData[i * 4 + 0] * yRatio
          const x1 = boxesData[i * 4 + 1] * xRatio
          const y2 = boxesData[i * 4 + 2] * yRatio
          const x2 = boxesData[i * 4 + 3] * xRatio
          const cx = (x1 + x2) / 2
          const cy = (y1 + y2) / 2
          // wrists kpt indices: 9 (left), 10 (right)
          const base = i * NUM_KEYPOINTS * KPT_STRIDE
          const lx = kptsData[base + 9 * KPT_STRIDE + 0] * xRatio
          const ly = kptsData[base + 9 * KPT_STRIDE + 1] * yRatio
          const lc = kptsData[base + 9 * KPT_STRIDE + 2]
          const rx = kptsData[base + 10 * KPT_STRIDE + 0] * xRatio
          const ry = kptsData[base + 10 * KPT_STRIDE + 1] * yRatio
          const rc = kptsData[base + 10 * KPT_STRIDE + 2]
          const det: Det = {
            idx: i,
            box: { x1, y1, x2, y2 },
            center: [cx, cy],
            score: scoresData[i]
          }
          if (lc >= KP_CONF_THRESHOLD) det.lw = [lx, ly]
          if (rc >= KP_CONF_THRESHOLD) det.rw = [rx, ry]
          dets.push(det)
        }

        // Expire old tracks
        const nowT = video.currentTime
        for (const [id, tr] of tracks) {
          if (nowT - tr.lastUpdateTime > 2.0) {
            tracks.delete(id)
          }
        }

        // Greedy matching by center distance
        const assignedTrackByDet: number[] = new Array(num).fill(-1)
        const usedTrackIds = new Set<number>()
        for (let i = 0; i < dets.length; i++) {
          const d = dets[i]
          let bestTrackId = -1
          let bestDist = Infinity
          for (const [id, tr] of tracks) {
            if (usedTrackIds.has(id)) continue
            // Only consider recently updated tracks
            if (nowT - tr.lastUpdateTime > 1.5) continue
            const dx = d.center[0] - tr.lastCenter[0]
            const dy = d.center[1] - tr.lastCenter[1]
            const dist = Math.hypot(dx, dy)
            // threshold proportional to box size
            const bw = tr.lastBox.x2 - tr.lastBox.x1
            const bh = tr.lastBox.y2 - tr.lastBox.y1
            const thresh = Math.max(0.5 * Math.hypot(bw, bh), 80)
            if (dist < thresh && dist < bestDist) {
              bestDist = dist
              bestTrackId = id
            }
          }
          if (bestTrackId !== -1) {
            assignedTrackByDet[i] = bestTrackId
            usedTrackIds.add(bestTrackId)
          }
        }

        // Create tracks for unassigned detections
        for (let i = 0; i < dets.length; i++) {
          if (assignedTrackByDet[i] !== -1) continue
          const d = dets[i]
          const id = nextTrackId++
          tracks.set(id, {
            id,
            lastBox: { ...d.box },
            lastCenter: [...d.center],
            lastUpdateTime: nowT,
            lastSampleTime: -Infinity,
            samples: [],
            status: 'idle'
          })
          assignedTrackByDet[i] = id
        }

        // Update tracks and sample wrists
        for (let i = 0; i < dets.length; i++) {
          const trackId = assignedTrackByDet[i]
          if (trackId === -1) continue
          const tr = tracks.get(trackId)!
          const d = dets[i]
          tr.lastBox = { ...d.box }
          tr.lastCenter = [...d.center]
          tr.lastUpdateTime = nowT

          // sample every SAMPLE_INTERVAL_S
          if (nowT - tr.lastSampleTime >= SAMPLE_INTERVAL_S) {
            const sample: TrackSample = { t: nowT }
            if (d.lw) sample.lw = d.lw
            if (d.rw) sample.rw = d.rw
            tr.samples.push(sample)
            tr.lastSampleTime = nowT
            // keep only last WINDOW_S seconds
            const cutoff = nowT - WINDOW_S
            tr.samples = tr.samples.filter(s => s.t >= cutoff)
          }

          // compute motion over window and classify
          if (tr.samples.length >= 2) {
            let totalDist = 0
            let countPairs = 0
            for (let k = 1; k < tr.samples.length; k++) {
              const a = tr.samples[k - 1]
              const b = tr.samples[k]
              const addDist = (p1?: WristPoint, p2?: WristPoint) => {
                if (!p1 || !p2) return 0
                return Math.hypot(p2[0] - p1[0], p2[1] - p1[1])
              }
              const dL = addDist(a.lw, b.lw)
              const dR = addDist(a.rw, b.rw)
              if (dL > 0) totalDist += dL
              if (dR > 0) totalDist += dR
              if (dL > 0 || dR > 0) countPairs++
            }
            // normalize by box height to be scale-invariant
            const boxH = Math.max(1, tr.lastBox.y2 - tr.lastBox.y1)
            const normMotion = countPairs > 0 ? (totalDist / countPairs) / boxH : 0
            // threshold tuned empirically; ~0.1 of box height per 0.5s indicates activity
            tr.status = normMotion >= 0.1 ? 'working' : 'idle'
          } else {
            tr.status = 'idle'
          }
        }

        // Build statuses aligned with detections order
        const statuses: Array<'working' | 'idle'> = dets.map((_, i) => {
          const id = assignedTrackByDet[i]
          if (id === -1) return 'idle'
          const tr = tracks.get(id)
          return tr ? tr.status : 'idle'
        })

        drawPoses(ctx, canvas, boxesData, scoresData, classesData, kptsData, xRatio, yRatio, statuses)

        if (onFrameCount) {
          let count = 0
          for (let i = 0; i < scoresData.length; i++) {
            if (scoresData[i] > 0) count++
          }
          onFrameCount(count)
        }

        if (options.onFrameStatusCounts) {
          let working = 0
          let idle = 0
          for (let i = 0; i < statuses.length; i++) {
            if (statuses[i] === 'working') working++
            else idle++
          }
          options.onFrameStatusCounts(working, idle)
        }

        tf.dispose([res, trans, boxes, scores, classes, nms, kptsAll, input, pad, img, boxesTensor, scoresTensor, classesTensor, kptsTensor])
        tf.engine().endScope()
      } catch (e) {
        // eslint-disable-next-line no-console
        console.warn('pose loop error', e)
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

function drawPoses(
  ctx: CanvasRenderingContext2D,
  canvas: HTMLCanvasElement,
  boxesData: Float32Array,
  scoresData: Float32Array,
  classesData: Float32Array,
  kptsData: Float32Array,
  xRatio: number,
  yRatio: number,
  statuses: Array<'working' | 'idle'>
) {
  ctx.clearRect(0, 0, canvas.width, canvas.height)

  const kpThreshold = 0.4
  const colorPrimary = '#00fff0'
  const colorSecondary = '#ff00e6'

  // Skeleton edges in COCO format indices
  const edges: Array<[number, number]> = [
    [0, 1], [0, 2], [1, 3], [2, 4], // head
    [5, 6], [5, 7], [7, 9], [6, 8], [8, 10], // arms
    [5, 11], [6, 12], [11, 12], // torso
    [11, 13], [13, 15], [12, 14], [14, 16] // legs
  ]

  for (let i = 0; i < scoresData.length; i++) {
    const y1 = boxesData[i * 4 + 0] * yRatio
    const x1 = boxesData[i * 4 + 1] * xRatio
    const y2 = boxesData[i * 4 + 2] * yRatio
    const x2 = boxesData[i * 4 + 3] * xRatio
    const w = Math.round(x2 - x1)
    const h = Math.round(y2 - y1)
    const x = Math.round(x1)
    const y = Math.round(y1)

    // Box (optional aesthetic)
    ctx.lineWidth = 2
    ctx.strokeStyle = colorPrimary
    ctx.shadowColor = 'rgba(0,255,240,0.6)'
    ctx.shadowBlur = 12
    ctx.strokeRect(x, y, w, h)
    ctx.shadowBlur = 0

    // Keypoints for this detection
    const base = i * NUM_KEYPOINTS * KPT_STRIDE
    const points: Array<[number, number, number]> = []
    for (let k = 0; k < NUM_KEYPOINTS; k++) {
      const kx = kptsData[base + k * KPT_STRIDE + 0] * xRatio
      const ky = kptsData[base + k * KPT_STRIDE + 1] * yRatio
      const kc = kptsData[base + k * KPT_STRIDE + 2]
      points.push([kx, ky, kc])
    }

    // Skeleton lines
    ctx.lineWidth = 3
    ctx.strokeStyle = colorSecondary
    ctx.shadowColor = 'rgba(255,0,230,0.5)'
    ctx.shadowBlur = 8
    for (const [a, b] of edges) {
      const [xA, yA, cA] = points[a]
      const [xB, yB, cB] = points[b]
      if (cA >= kpThreshold && cB >= kpThreshold) {
        ctx.beginPath()
        ctx.moveTo(xA, yA)
        ctx.lineTo(xB, yB)
        ctx.stroke()
      }
    }
    ctx.shadowBlur = 0

    // Joints
    for (const [px, py, pc] of points) {
      if (pc < kpThreshold) continue
      ctx.beginPath()
      ctx.fillStyle = '#e6faff'
      ctx.arc(px, py, 3.5, 0, Math.PI * 2)
      ctx.fill()
    }

    // Status label
    const status = statuses[i] ?? 'idle'
    const label = status === 'working' ? 'working' : 'idle'
    const bg = status === 'working' ? 'rgba(0, 200, 120, 0.9)' : 'rgba(180, 180, 40, 0.9)'
    const txtColor = '#ffffff'
    const paddingX = 8
    const paddingY = 4
    const fontSize = 13
    ctx.font = `600 ${fontSize}px Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial`
    const textWidth = ctx.measureText(label).width
    const boxW = textWidth + paddingX * 2
    const boxH = fontSize + paddingY * 2 - 2
    const bx = Math.max(0, Math.min(canvas.width - boxW - 2, x))
    const by = Math.max(0, y - boxH - 8)
    ctx.shadowColor = 'rgba(0,0,0,0.25)'
    ctx.shadowBlur = 8
    ctx.fillStyle = bg
    ctx.fillRect(bx, by, boxW, boxH)
    ctx.shadowBlur = 0
    ctx.fillStyle = txtColor
    ctx.fillText(label, bx + paddingX, by + boxH - paddingY - 2)
  }
}


