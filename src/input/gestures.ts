/**
 * 手势识别：把平板 Pointer Events 转成沙画笔迹段。
 *
 * 自动识别规则（结合接触宽度、速度、压感、停留时间）：
 *   - 静止轻点（位移小、时间短）            -> 指尖点孔 Tap
 *   - 接触面积大且横向移动（手掌侧面）       -> 侧扫 Sweep
 *   - 两点快速反向 / 工具栏翻转 / 摇平板     -> 翻转（由 App 处理惯性）
 *   - 普通慢速移动                          -> 推沙 Push
 *   - 双指或长按停顿后移动                  -> 撒沙 Sprinkle
 * 老师也可以在工具栏手动锁定工具，锁定后不再自动判别。
 */
import { Tool, type StrokeSegment } from '../sim/constants';
import { clamp } from '../sim/math';

export type ManualTool = 'auto' | 'push' | 'tap' | 'sweep' | 'sprinkle';

export interface PointerSample {
  x: number;
  y: number;
  t: number;
  pressure: number;
  width: number;
  height: number;
}

export interface StrokeConfig {
  /** 模拟纹理尺寸 */
  simWidth: number;
  simHeight: number;
  /** 画布上灯箱视口（CSS 像素） */
  viewport: { x: number; y: number; w: number; h: number };
  manualTool: ManualTool;
}

const TAP_MAX_MOVE = 18;       // CSS px，小于该位移视为点
const TAP_MAX_MS = 260;        // 按住时长
const SWEEP_WIDTH_RATIO = 26;  // 接触宽度超过手指基线的倍率阈值
const SPRINKLE_HOLD_MS = 320;  // 长按停顿后转撒沙

export class GestureRecognizer {
  private active = new Map<number, PointerSample>();
  private start: PointerSample | null = null;
  private last: PointerSample | null = null;
  private heldStill = false;
  private decided: 'tap' | 'moving' | null = null;
  private seedCounter = 0;
  cfg: StrokeConfig;

  constructor(cfg: StrokeConfig) {
    this.cfg = cfg;
  }

  /** 客户端坐标 -> 模拟纹理坐标 */
  toSim(clientX: number, clientY: number): { x: number; y: number } {
    const v = this.cfg.viewport;
    const nx = clamp((clientX - v.x) / v.w, 0, 1);
    const ny = clamp((clientY - v.y) / v.h, 0, 1);
    return { x: nx * this.cfg.simWidth, y: ny * this.cfg.simHeight };
  }

  private sample(e: PointerEvent): PointerSample {
    const p = this.toSim(e.clientX, e.clientY);
    return {
      x: p.x,
      y: p.y,
      t: performance.now(),
      pressure: e.pressure > 0 ? e.pressure : 0.5,
      width: Math.max(1, e.width),
      height: Math.max(1, e.height),
    };
  }

  /** 接触点数（用于区分双指撒沙） */
  get pointerCount(): number {
    return this.active.size;
  }

  down(e: PointerEvent): void {
    const s = this.sample(e);
    this.active.set(e.pointerId, s);
    if (this.active.size === 1) {
      this.start = s;
      this.last = s;
      this.decided = null;
      this.heldStill = false;
    }
  }

  /**
   * 移动事件返回 0..N 个笔迹段（快速移动一帧可能拆成多段，上限由引擎 MAX_SEGMENTS 控制）。
   * 同时返回当前识别出的工具，供 UI 高亮。
   */
  move(e: PointerEvent): { segments: StrokeSegment[]; tool: ManualTool } {
    if (!this.active.has(e.pointerId) || !this.start || !this.last) {
      return { segments: [], tool: this.cfg.manualTool };
    }
    const s = this.sample(e);
    this.active.set(e.pointerId, s);

    const fromStart = Math.hypot(s.x - this.start.x, s.y - this.start.y);
    const dt = Math.max(1, s.t - this.last.t);
    const speed = Math.hypot(s.x - this.last.x, s.y - this.last.y) / dt; // 纹理像素/ms

    if (this.decided === null) {
      if (fromStart < TAP_MAX_MOVE * (this.cfg.simWidth / this.cfg.viewport.w)) {
        if (s.t - this.start.t > SPRINKLE_HOLD_MS) this.heldStill = true;
        this.last = s;
        return { segments: [], tool: this.cfg.manualTool };
      }
      this.decided = 'moving';
    }

    const tool = this.resolveTool(e, s, speed);
    const seg = this.buildSegment(this.last, s, tool, speed);
    this.last = s;
    return { segments: seg ? [seg] : [], tool };
  }

  private resolveTool(e: PointerEvent, s: PointerSample, speed: number): ManualTool {
    if (this.cfg.manualTool !== 'auto') return this.cfg.manualTool;
    // 双指或长按停顿 -> 撒沙
    if (this.active.size >= 2 || this.heldStill) return 'sprinkle';
    // 手掌侧面：接触面积明显变大
    const contactBaseline = this.cfg.viewport.w / SWEEP_WIDTH_RATIO;
    if (s.width * (this.cfg.simWidth / this.cfg.viewport.w) > contactBaseline * 1.4) return 'sweep';
    void e;
    void speed;
    return 'push';
  }

  private buildSegment(a: PointerSample, b: PointerSample, tool: ManualTool, speed: number): StrokeSegment | null {
    const pressure = clamp((a.pressure + b.pressure) * 0.5, 0.05, 1);
    const scale = this.cfg.simWidth / this.cfg.viewport.w; // 纹理像素 / CSS 像素
    switch (tool) {
      case 'tap':
      case 'push': {
        const radius = (14 + pressure * 16) * scale;
        return {
          x0: a.x, y0: a.y, x1: b.x, y1: b.y,
          radius, pressure, tool: Tool.Push,
          pushDist: clamp(12 + speed * 90, 8, 60) * scale * 0.6,
          pickup: 0.85,
          seed: this.seedCounter++,
        };
      }
      case 'sweep': {
        const radius = (46 + pressure * 30) * scale;
        return {
          x0: a.x, y0: a.y, x1: b.x, y1: b.y,
          radius, pressure, tool: Tool.Sweep,
          pushDist: clamp(20 + speed * 120, 12, 80) * scale * 0.5,
          pickup: 0.7,
          seed: this.seedCounter++,
        };
      }
      case 'sprinkle': {
        const radius = (10 + pressure * 14) * scale;
        return {
          x0: a.x, y0: a.y, x1: b.x, y1: b.y,
          radius, pressure, tool: Tool.Sprinkle,
          pushDist: 0, pickup: 0,
          seed: (this.seedCounter++ * 2654435761) >>> 0,
        };
      }
      default:
        return null;
    }
  }

  /** 抬手：若全程几乎没动且时间短，产出一个点孔段 */
  up(e: PointerEvent): StrokeSegment | null {
    const wasStart = this.start;
    this.active.delete(e.pointerId);
    if (!wasStart || this.active.size > 0) {
      if (this.active.size === 0) this.reset();
      return null;
    }
    const end = this.sample(e);
    const moveDist = Math.hypot(end.x - wasStart.x, end.y - wasStart.y);
    const heldMs = end.t - wasStart.t;
    this.reset();
    const scale = this.cfg.simWidth / this.cfg.viewport.w;
    const isTap =
      this.cfg.manualTool === 'tap' ||
      (this.cfg.manualTool === 'auto' && moveDist < TAP_MAX_MOVE * scale && heldMs < TAP_MAX_MS);
    if (!isTap) return null;
    const pressure = end.pressure > 0 ? end.pressure : 0.5;
    // 孔半径上限 7 纹理像素：点孔由单个 16x16 workgroup（含孔沿 r+1）处理
    const radius = Math.min((12 + pressure * 10) * scale, 7);
    return {
      x0: end.x, y0: end.y, x1: end.x, y1: end.y,
      radius,
      pressure,
      tool: Tool.Tap,
      pushDist: 0,
      pickup: 1,
      seed: this.seedCounter++,
    };
  }

  private reset(): void {
    this.start = null;
    this.last = null;
    this.decided = null;
    this.heldStill = false;
  }
}
