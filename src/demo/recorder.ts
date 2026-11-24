/**
 * 老师示范：录制笔迹时间线、回放、描红提示。
 * 录制只存归一化坐标（0..1）与工具/压感，换分辨率也能回放；
 * 回放时按当前模拟尺寸还原成 StrokeSegment 喂给引擎。
 */
import { Tool, type StrokeSegment } from '../sim/constants';
import type { ManualTool } from '../input/gestures';

export interface RecordedPoint {
  t: number;       // 相对录制开始的毫秒
  x: number;       // 归一化 0..1
  y: number;
  pressure: number;
  tool: ManualTool;
  phase: 'down' | 'move' | 'up';
}

export interface Demo {
  name: string;
  createdAt: number;
  durationMs: number;
  points: RecordedPoint[];
}

const DEMO_STORAGE_KEY = 'sand-painting-demos-v1';

export class DemoRecorder {
  private recording: RecordedPoint[] = [];
  private startAt = 0;
  private recorded: Demo[] = [];

  constructor() {
    this.recorded = this.loadFromStorage();
  }

  get isRecording(): boolean {
    return this.startAt > 0;
  }

  start(name: string): void {
    this.recording = [];
    this.startAt = performance.now();
    this.currentName = name;
  }

  private currentName = '未命名示范';

  addPoint(nx: number, ny: number, pressure: number, tool: ManualTool, phase: 'down' | 'move' | 'up'): void {
    if (!this.isRecording) return;
    this.recording.push({
      t: performance.now() - this.startAt,
      x: nx,
      y: ny,
      pressure,
      tool,
      phase,
    });
  }

  stop(): Demo | null {
    if (!this.isRecording || this.recording.length === 0) {
      this.startAt = 0;
      return null;
    }
    const demo: Demo = {
      name: this.currentName,
      createdAt: Date.now(),
      durationMs: this.recording[this.recording.length - 1].t,
      points: this.recording,
    };
    this.recorded.unshift(demo);
    this.startAt = 0;
    this.recording = [];
    this.saveToStorage();
    return demo;
  }

  list(): Demo[] {
    return this.recorded;
  }

  remove(index: number): void {
    this.recorded.splice(index, 1);
    this.saveToStorage();
  }

  /** 描红用：取示范的中心线（move/down 点），归一化坐标 */
  guidePath(demo: Demo): Array<{ x: number; y: number }> {
    return demo.points.filter((p) => p.phase !== 'up').map((p) => ({ x: p.x, y: p.y }));
  }

  /**
   * 把示范在给定时间窗内的点还原成引擎笔迹段。
   * 调用方每帧传入当前回放时间，返回该帧应执行的段。
   */
  segmentsUntil(demo: Demo, fromMs: number, toMs: number, simWidth: number, simHeight: number): StrokeSegment[] {
    const window = demo.points.filter((p) => p.t >= fromMs && p.t < toMs && p.phase !== 'up');
    const segs: StrokeSegment[] = [];
    for (let i = 1; i < window.length; i++) {
      const a = window[i - 1];
      const b = window[i];
      const tool = b.tool === 'sweep' ? Tool.Sweep : b.tool === 'sprinkle' ? Tool.Sprinkle : Tool.Push;
      const scale = simWidth / 1024; // 基准半径按 1024 宽设计
      segs.push({
        x0: a.x * simWidth,
        y0: a.y * simHeight,
        x1: b.x * simWidth,
        y1: b.y * simHeight,
        radius: (tool === Tool.Sweep ? 60 : tool === Tool.Sprinkle ? 22 : 26) * scale,
        pressure: b.pressure,
        tool,
        pushDist: tool === Tool.Push ? 26 * scale : tool === Tool.Sweep ? 34 * scale : 0,
        pickup: tool === Tool.Sprinkle ? 0 : 0.8,
        seed: i,
      });
    }
    // 点孔（半径 7，适配 16x16 单 workgroup 环槽）
    const tap = demo.points.find((p) => p.t >= fromMs && p.t < toMs && p.tool === 'tap');
    if (tap) {
      segs.push({
        x0: tap.x * simWidth, y0: tap.y * simHeight,
        x1: tap.x * simWidth, y1: tap.y * simHeight,
        radius: 7, pressure: tap.pressure,
        tool: Tool.Tap, pushDist: 0, pickup: 1, seed: 9999,
      });
    }
    return segs.slice(0, 8);
  }

  private loadFromStorage(): Demo[] {
    try {
      const raw = localStorage.getItem(DEMO_STORAGE_KEY);
      return raw ? (JSON.parse(raw) as Demo[]) : [];
    } catch {
      return [];
    }
  }

  private saveToStorage(): void {
    try {
      localStorage.setItem(DEMO_STORAGE_KEY, JSON.stringify(this.recorded.slice(0, 20)));
    } catch {
      // 存储满或被禁用时静默降级，本次会话仍可用
    }
  }
}
