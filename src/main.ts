import './ui/styles.css';
import { GpuSandEngine } from './gpu/GpuSandEngine';
import { GestureRecognizer, type ManualTool } from './input/gestures';
import { PracticeChecker, LESSONS, type CheckResult, type LessonId } from './practice/checker';
import { DemoRecorder, type Demo } from './demo/recorder';
import { UiBuilder, type AppMode } from './ui/UiBuilder';
import { PARAM_PRESETS, RESOLUTION_OPTIONS, cloneParams, type Preset } from './sim/presets';
import { SIM_WIDTH, SIM_HEIGHT, type SimParams, type StrokeSegment } from './sim/constants';

class SandApp {
  private canvas: HTMLCanvasElement;
  private overlay: SVGSVGElement;
  private engine!: GpuSandEngine;
  private ui!: UiBuilder;
  private gestures!: GestureRecognizer;
  private checker!: PracticeChecker;
  private recorder = new DemoRecorder();
  private params: SimParams = cloneParams(PARAM_PRESETS[0].params);
  private mode: AppMode = 'practice';
  private manualTool: ManualTool = 'auto';
  private activeLesson: LessonId = 'push';
  private presetIndex = 0;
  private resolutionIndex = 0;
  private lastResult: CheckResult | null = null;
  private pendingSegments: StrokeSegment[] = [];
  private strokeActive = false;
  private seed = 1;
  private demoPlayback: { demo: Demo; startAt: number } | null = null;
  private demoLastT = 0;
  private rafId = 0;

  constructor() {
    this.canvas = document.getElementById('stage') as HTMLCanvasElement;
    this.overlay = document.getElementById('guide-overlay') as unknown as SVGSVGElement;
  }

  async start(): Promise<void> {
    try {
      this.resizeCanvas();
      const res = RESOLUTION_OPTIONS[this.resolutionIndex];
      this.engine = await GpuSandEngine.create(this.canvas, res.width, res.height, this.params);
    } catch (err) {
      this.showUnsupported(err);
      return;
    }

    this.gestures = new GestureRecognizer({
      simWidth: this.engine.width,
      simHeight: this.engine.height,
      viewport: this.currentViewport(),
      manualTool: this.manualTool,
    });
    this.checker = new PracticeChecker(this.engine);

    this.ui = new UiBuilder(document.getElementById('ui-root') as HTMLDivElement, {
      onMode: (m) => this.setMode(m),
      onTool: (t) => this.setTool(t),
      onClear: () => this.engine.clearAll(),
      onFlip: () => this.engine.flip(6),
      onUndo: () => this.undo(),
      onEvaluate: () => void this.evaluate(),
      onSelectLesson: (id) => this.selectLesson(id),
      onRecordToggle: () => this.toggleRecord(),
      onPlayDemo: (demo) => this.playDemo(demo),
      onGuideToggle: () => this.toggleGuide(),
      onDeleteDemo: (i) => {
        this.recorder.remove(i);
        this.refreshPanel();
      },
      onParamsChange: (p) => this.applyParams(p),
      onPreset: (preset) => this.applyPreset(preset),
      onResolution: (i) => void this.changeResolution(i),
    });
    this.ui.setPanelParams(this.params);
    this.refreshPanel();
    this.bindPointer();
    window.addEventListener('resize', () => {
      this.resizeCanvas();
      this.gestures.cfg.viewport = this.currentViewport();
      this.drawGuide();
    });

    // 摇一摇翻转（平板内置加速度计，需用户授权）
    this.setupShake();
    this.loop();
  }

  private resizeCanvas(): void {
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    this.canvas.width = Math.round(window.innerWidth * dpr);
    this.canvas.height = Math.round(window.innerHeight * dpr);
  }

  /** 灯箱在画布上的视口（CSS 像素，contain 4:3） */
  private currentViewport(): { x: number; y: number; w: number; h: number } {
    const cw = window.innerWidth;
    const ch = window.innerHeight;
    const simAspect = (this.engine?.width ?? SIM_WIDTH) / (this.engine?.height ?? SIM_HEIGHT);
    const canvasAspect = cw / ch;
    let w: number;
    let h: number;
    if (canvasAspect > simAspect) {
      h = ch;
      w = h * simAspect;
    } else {
      w = cw;
      h = w / simAspect;
    }
    return { x: (cw - w) / 2, y: (ch - h) / 2, w, h };
  }

  private bindPointer(): void {
    const c = this.canvas;
    c.addEventListener('pointerdown', (e) => {
      c.setPointerCapture(e.pointerId);
      this.gestures.down(e);
      const norm = this.toNorm(e.clientX, e.clientY);
      if (this.recorder.isRecording) {
        this.recorder.addPoint(norm.x, norm.y, e.pressure || 0.5, this.manualTool, 'down');
      }
    });
    c.addEventListener('pointermove', (e) => {
      const { segments, tool } = this.gestures.move(e);
      if (segments.length > 0) {
        this.pendingSegments.push(...segments);
        const norm = this.toNorm(e.clientX, e.clientY);
        if (this.recorder.isRecording) {
          this.recorder.addPoint(norm.x, norm.y, e.pressure || 0.5, tool, 'move');
        }
      }
    });
    const finish = (e: PointerEvent) => {
      const tap = this.gestures.up(e);
      if (tap) this.pendingSegments.push(tap);
      const norm = this.toNorm(e.clientX, e.clientY);
      if (this.recorder.isRecording) {
        this.recorder.addPoint(norm.x, norm.y, e.pressure || 0.5, this.manualTool, 'up');
      }
    };
    c.addEventListener('pointerup', finish);
    c.addEventListener('pointercancel', finish);
  }

  private toNorm(clientX: number, clientY: number): { x: number; y: number } {
    const v = this.currentViewport();
    return {
      x: Math.min(1, Math.max(0, (clientX - v.x) / v.w)),
      y: Math.min(1, Math.max(0, (clientY - v.y) / v.h)),
    };
  }

  private setupShake(): void {
    const requestMotion = () => {
      const dme = DeviceMotionEvent as unknown as { requestPermission?: () => Promise<string> };
      if (typeof dme.requestPermission === 'function') {
        void dme.requestPermission();
      }
      window.removeEventListener('pointerdown', requestMotion);
    };
    window.addEventListener('pointerdown', requestMotion);

    let lastShake = 0;
    let lastX: number | null = null;
    let lastY: number | null = null;
    window.addEventListener('devicemotion', (e) => {
      const a = e.accelerationIncludingGravity;
      if (!a || a.x === null || a.y === null) return;
      if (lastX !== null && lastY !== null) {
        const jerk = Math.hypot(a.x - lastX, a.y - (lastY as number));
        if (jerk > 22 && performance.now() - lastShake > 900) {
          lastShake = performance.now();
          this.engine.flip(6);
          this.ui.toast('检测到翻转，沙子按惯性滑动');
        }
      }
      lastX = a.x;
      lastY = a.y;
    });
  }

  private loop = (): void => {
    // 示范回放按时间线产出笔迹
    if (this.demoPlayback) {
      const now = performance.now() - this.demoPlayback.startAt;
      const segs = this.recorder.segmentsUntil(
        this.demoPlayback.demo,
        this.demoLastT,
        now,
        this.engine.width,
        this.engine.height,
      );
      if (segs.length > 0) this.pendingSegments.push(...segs);
      this.demoLastT = now;
      if (now > this.demoPlayback.demo.durationMs + 400) {
        this.demoPlayback = null;
        this.ui.toast('示范回放完成');
      }
    }

    if (this.pendingSegments.length > 0) {
      if (!this.strokeActive) {
        this.engine.pushUndoSnapshot();
        this.strokeActive = true;
      }
      const batch = this.pendingSegments.splice(0, 8);
      this.engine.applyStrokes(batch, this.seed++);
    } else {
      this.strokeActive = false;
    }
    this.engine.runAvalanche();
    this.engine.render(this.canvas);

    const ctr = this.engine.getCounters();
    this.ui.setStats(ctr.fps || 60, `${this.engine.width}×${this.engine.height}`);
    this.rafId = requestAnimationFrame(this.loop);
  };

  private setMode(mode: AppMode): void {
    this.mode = mode;
    this.refreshPanel();
    this.drawGuide();
  }

  private setTool(tool: ManualTool): void {
    this.manualTool = tool;
    this.gestures.cfg.manualTool = tool;
  }

  private selectLesson(id: LessonId): void {
    this.activeLesson = id;
    this.lastResult = null;
    this.refreshPanel();
    this.drawGuide();
    const lesson = LESSONS.find((l) => l.id === id)!;
    this.ui.toast(`练习：${lesson.name}`);
  }

  private async evaluate(): Promise<void> {
    const lesson = LESSONS.find((l) => l.id === this.activeLesson)!;
    this.ui.toast('正在回读沙层评测…');
    const result = await this.checker.evaluate(lesson);
    this.lastResult = result;
    this.refreshPanel();
    this.ui.toast(result.pass ? '达标！' : '再试一次，看看面板建议');
  }

  private toggleRecord(): void {
    if (this.recorder.isRecording) {
      const demo = this.recorder.stop();
      this.ui.recording = false;
      if (demo) this.ui.toast(`已保存示范：${demo.name}`);
    } else {
      this.recorder.start(`示范 ${this.recorder.list().length + 1}`);
      this.ui.recording = true;
      this.ui.toast('开始录制，请在灯箱上示范');
    }
    this.refreshPanel();
  }

  private playDemo(demo: Demo): void {
    this.demoPlayback = { demo, startAt: performance.now() };
    this.demoLastT = 0;
    this.ui.toast('回放示范中…');
  }

  private toggleGuide(): void {
    this.ui.showGuide = !this.ui.showGuide;
    this.refreshPanel();
    this.drawGuide();
  }

  private drawGuide(): void {
    this.overlay.innerHTML = '';
    if (this.mode !== 'practice' && !(this.mode === 'demo' && this.ui.showGuide)) return;
    const v = this.currentViewport();
    if (this.mode === 'practice') {
      const lesson = LESSONS.find((l) => l.id === this.activeLesson)!;
      const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      rect.setAttribute('x', String(v.x + lesson.target.x * v.w));
      rect.setAttribute('y', String(v.y + lesson.target.y * v.h));
      rect.setAttribute('width', String(lesson.target.w * v.w));
      rect.setAttribute('height', String(lesson.target.h * v.h));
      rect.setAttribute('fill', 'none');
      rect.setAttribute('stroke', 'rgba(232,181,106,0.7)');
      rect.setAttribute('stroke-width', '2');
      rect.setAttribute('stroke-dasharray', '8 6');
      rect.setAttribute('rx', '10');
      this.overlay.appendChild(rect);
    } else if (this.mode === 'demo' && this.ui.showGuide) {
      const demo = this.recorder.list()[0];
      if (!demo) return;
      const path = this.recorder.guidePath(demo);
      if (path.length < 2) return;
      const poly = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
      poly.setAttribute(
        'points',
        path.map((p) => `${v.x + p.x * v.w},${v.y + p.y * v.h}`).join(' '),
      );
      poly.setAttribute('fill', 'none');
      poly.setAttribute('stroke', 'rgba(255,217,154,0.55)');
      poly.setAttribute('stroke-width', '3');
      poly.setAttribute('stroke-linecap', 'round');
      poly.setAttribute('stroke-linejoin', 'round');
      this.overlay.appendChild(poly);
    }
  }

  private applyParams(p: SimParams): void {
    this.params = p;
    this.ui.setPanelParams(p);
    this.engine.params = p;
  }

  private applyPreset(preset: Preset): void {
    this.presetIndex = PARAM_PRESETS.indexOf(preset);
    this.params = cloneParams(preset.params);
    this.ui.setPanelParams(this.params);
    this.engine.params = this.params;
    this.refreshPanel();
    this.ui.toast(`已切换预设：${preset.name}`);
  }

  private async changeResolution(index: number): Promise<void> {
    if (index === this.resolutionIndex) return;
    this.resolutionIndex = index;
    const res = RESOLUTION_OPTIONS[index];
    cancelAnimationFrame(this.rafId);
    // 重新创建引擎（沙层会清空，提示用户）
    this.engine = await GpuSandEngine.create(this.canvas, res.width, res.height, this.params);
    this.checker = new PracticeChecker(this.engine);
    this.gestures.cfg.simWidth = res.width;
    this.gestures.cfg.simHeight = res.height;
    this.gestures.cfg.viewport = this.currentViewport();
    this.ui.toast(`分辨率切换为 ${res.width}×${res.height}，沙层已重置`);
    this.loop();
    this.refreshPanel();
  }

  /** 撤销：恢复最近一次高度缓冲快照 */
  private undo(): void {
    if (this.engine.undo()) {
      this.ui.toast('已撤销上一步');
    } else {
      this.ui.toast('没有可撤销的步骤');
    }
  }

  private refreshPanel(): void {
    this.ui.renderPanel({
      activeLesson: this.activeLesson,
      demos: this.recorder.list(),
      params: this.params,
      presetIndex: this.presetIndex,
      resolutionIndex: this.resolutionIndex,
      resultHtml: this.lastResult ? this.resultHtml(this.lastResult) : undefined,
      resultPass: this.lastResult?.pass,
    });
    this.ui.setPanelParams(this.params);
  }

  private resultHtml(r: CheckResult): string {
    const metrics = Object.entries(r.metrics)
      .map(([k, val]) => `<div class="metric-line">${k}：<b>${val}</b></div>`)
      .join('');
    return `<div class="score">${r.score} 分 · ${r.pass ? '达标' : '未达标'}</div>
      <div class="metric-line">${r.message}</div>${metrics}`;
  }

  private showUnsupported(err: unknown): void {
    const box = document.getElementById('unsupported')!;
    box.hidden = false;
    const reason = document.getElementById('unsupported-reason')!;
    reason.textContent = `错误码：${err instanceof Error ? err.message : String(err)}`;
  }
}

void new SandApp().start();
