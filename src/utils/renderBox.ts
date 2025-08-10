import labels from '../labels.json'

export const renderBoxes = (
  canvas: HTMLCanvasElement,
  boxesData: Float32Array,
  scoresData: Float32Array,
  classesData: Float32Array,
  ratios: [number, number]
) => {
  const ctx = canvas.getContext('2d')!
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height)

  const colors = new Colors()

  const fontSize = Math.max(
    Math.round(Math.max(ctx.canvas.width, ctx.canvas.height) / 40),
    14
  )
  const font = `${fontSize}px Arial`
  ctx.font = font
  ctx.textBaseline = 'top'

  for (let i = 0; i < scoresData.length; ++i) {
    const klass = labels[Math.floor(classesData[i])] ?? 'obj'
    const color = colors.get(classesData[i])
    const score = (scoresData[i] * 100).toFixed(1)

    let y1 = boxesData[i * 4 + 0]
    let x1 = boxesData[i * 4 + 1]
    let y2 = boxesData[i * 4 + 2]
    let x2 = boxesData[i * 4 + 3]

    x1 *= ratios[0]
    x2 *= ratios[0]
    y1 *= ratios[1]
    y2 *= ratios[1]

    const width = x2 - x1
    const height = y2 - y1

    ctx.fillStyle = Colors.hexToRgba(color, 0.2)!
    ctx.fillRect(x1, y1, width, height)

    ctx.strokeStyle = color
    ctx.lineWidth = Math.max(Math.min(ctx.canvas.width, ctx.canvas.height) / 200, 2.5)
    ctx.strokeRect(x1, y1, width, height)

    ctx.fillStyle = color
    const text = `${klass} - ${score}%`
    const textWidth = ctx.measureText(text).width
    const textHeight = parseInt(font, 10)
    const yText = y1 - (textHeight + ctx.lineWidth)
    ctx.fillRect(
      x1 - 1,
      yText < 0 ? 0 : yText,
      textWidth + ctx.lineWidth,
      textHeight + ctx.lineWidth
    )

    ctx.fillStyle = '#ffffff'
    ctx.fillText(text, x1 - 1, yText < 0 ? 0 : yText)
  }
}

class Colors {
  private palette: string[]
  public n: number
  constructor() {
    this.palette = [
      '#FF3838',
      '#FF9D97',
      '#FF701F',
      '#FFB21D',
      '#CFD231',
      '#48F90A',
      '#92CC17',
      '#3DDB86',
      '#1A9334',
      '#00D4BB',
      '#2C99A8',
      '#00C2FF',
      '#344593',
      '#6473FF',
      '#0018EC',
      '#8438FF',
      '#520085',
      '#CB38FF',
      '#FF95C8',
      '#FF37C7'
    ]
    this.n = this.palette.length
  }

  get = (i: number) => this.palette[Math.floor(i) % this.n]

  static hexToRgba = (hex: string, alpha: number) => {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex)
    return result
      ? `rgba(${[parseInt(result[1], 16), parseInt(result[2], 16), parseInt(result[3], 16)].join(
          ', '
        )}, ${alpha})`
      : null
  }
}


