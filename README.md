# Pose-based Behavior Detection (Web, TFJS)

A client‑side app that detects people and classifies their activity as working or idle in real time using a YOLOv8‑pose model running in the browser (TensorFlow.js). Upload a video, use the webcam, or try built‑in demos; the app overlays a skeleton and shows live counters.

## Features
- Real‑time pose detection (no server)
- Working vs Idle classification per person
  - Based on wrist motion over a 3‑second sliding window (sampled every 0.5s)
  - Scale‑normalized thresholding, lightweight tracking for stability
- Webcam support via a single button
- Built‑in demos in `public/demo/` (`idle.mp4`, `working.mp4`)
- Neon UI with canvas overlay (skeleton, joints, labels) and live counters

## Quick start
Using Bun:

```bash
bun install
bun dev
```

Node/NPM alternative:

```bash
npm install
npm run dev
```

Open the local URL shown in the terminal.

## How it works
- Model: YOLOv8‑pose TFJS graph model served from `public/pose_model/model.json` (already included). See `public/pose_model/metadata.yaml`.
- Pipeline: frame → square pad → resize → normalize → model forward → transpose → decode boxes + keypoints → NMS → draw skeleton.
- Behavior: Greedy track matching by center, sample left/right wrist positions every 0.5s, compute normalized motion over 3s → classify as "working" or "idle" per track.

## Key files
- `src/detection/pose.ts` – pose model loader, detection loop, tracking, and working/idle logic
- `src/App.tsx` – UI (upload, webcam button, demo selector, video + canvas overlay, counters)
- `src/labels.json` – COCO class labels (used for display)

## Usage
- Click "Start Webcam" or choose a demo (Idle/Working), or click "Upload Video".
- Counters on the right show total, working, and idle.
- Click the video to play/pause.

## Notes
- Uses `@tensorflow/tfjs` (WebGL). Chrome recommended.
- Webcam requires camera permission; if the video appears stuck, click the video once to trigger playback (autoplay policies may apply).
- Icons via `lucide-react`.

## License
MIT for app code. Review model weight licenses separately.
