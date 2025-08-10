# Crowd Count (YOLOv8, Web)

A simple, client‑side crowd counter that detects people in videos using YOLOv8 running in the browser (TensorFlow.js). Upload your own video or pick a built‑in demo; the app draws neon boxes and shows a live person count.

## Features
- Real‑time person detection on the client (no server)
- Neon cyberpunk UI with overlay canvas and HUD counter
- Upload video or choose from demos in `public/demo/`
- Click video to play/pause (native controls hidden)

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
- Model: YOLOv8 TFJS graph model served from `public/yolov8n_web_model/model.json` (already included).
- Pipeline: frame → square pad → resize → normalize → model forward → transpose → decode boxes → NMS → draw.
- Counting: Filters detections to the COCO `person` class and updates a HUD counter each frame (no React state for performance).

## Key files
- `src/detection/yolo.ts` – model loader and detection loop
- `src/App.tsx` – UI (upload, demo selector, video + canvas overlay, counter)
- `src/labels.json` – COCO class labels

## Notes
- Uses `@tensorflow/tfjs` (WebGL). Chrome recommended.
- Icons via `lucide-react`.

## License
MIT for app code. Review model weight licenses separately.
