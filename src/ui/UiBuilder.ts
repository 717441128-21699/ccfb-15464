/**
 * 界面构建：顶部模式栏、底部工具栏、四个模式面板（手法练习/自由创作/老师示范/灯箱沙子设置）。
 * 纯 DOM，无框架；所有按钮事件通过回调上抛给 App。
 */
import { LESSONS, type Lesson, type LessonId } from '../practice/checker';
import { PARAM_PRESETS, RESOLUTION_OPTIONS, type Preset } from '../sim/presets';
import type { SimParams } from '../sim/constants';
import type { Demo } from '../demo/recorder';
import type { ManualTool } from '../input/gestures';

export type AppMode = 'practice' | 'free' | 'demo' | 'settings';

export interface UiCallbacks {
  onMode(mode: AppMode): void;
  onTool(tool: ManualTool): void;
  onClear(): void;
  onFlip(): void;
  onUndo(): void;
  onEvaluate(): void;
  onSelectLesson(id: LessonId): void;
  onRecordToggle(): void;
  onPlayDemo(demo: Demo): void;
  onGuideToggle(): void;
  onDeleteDemo(index: number): void;
  onParamsChange(params: SimParams): void;
  onPreset(preset: Preset): void;
  onResolution(index: number): void;
}

/** 面板渲染所需的全部外部数据（由 App 每次刷新时传入） */
export interface PanelExtra {
  activeLesson?: LessonId;
  demos?: Demo[];
  params?: SimParams;
  presetIndex?: number;
  resolutionIndex?: number;
  resultHtml?: string;
  resultPass?: boolean;
}

export class UiBuilder {
  root: HTMLDivElement;
  cb: UiCallbacks;
  mode: AppMode = 'practice';
  manualTool: ManualTool = 'auto';
  private panelEl!: HTMLDivElement;
  private toastEl!: HTMLDivElement;
  private statsFps!: HTMLElement;
  private statsRes!: HTMLElement;
  private paramInputs = new Map<string, HTMLInputElement>();
  recording = false;
  showGuide = true;

  constructor(root: HTMLDivElement, cb: UiCallbacks) {
    this.root = root;
    this.cb = cb;
    this.build();
  }

  private el<K extends keyof HTMLElementTagNameMap>(tag: K, cls = '', text = ''): HTMLElementTagNameMap[K] {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text) e.textContent = text;
    return e;
  }

  private build(): void {
    const top = this.el('div', 'topbar');
    const brand = this.el('div', 'brand', '🏜️ <span class="full">沙画手法训练台</span>');
    top.appendChild(brand);

    const tabs = this.el('div', 'mode-tabs');
    const tabDefs: Array<[AppMode, string]> = [
      ['practice', '手法练习'],
      ['free', '自由创作'],
      ['demo', '老师示范'],
      ['settings', '灯箱和沙子设置'],
    ];
    tabDefs.forEach(([mode, label]) => {
      const b = this.el('button', mode === this.mode ? 'active' : '', label);
      b.onclick = () => this.setMode(mode);
      b.dataset.mode = mode;
      tabs.appendChild(b);
    });
    top.appendChild(tabs);

    const stats = this.el('div', 'stats');
    this.statsFps = this.el('b', '', '–');
    this.statsRes = this.el('span', 'res');
    stats.appendChild(this.statsFps);
    stats.appendChild(this.el('span', '', ' FPS'));
    stats.appendChild(this.el('br'));
    stats.appendChild(this.statsRes);
    top.appendChild(stats);
    this.root.appendChild(top);

    this.panelEl = this.el('div', 'panel');
    this.root.appendChild(this.panelEl);

    const toolbar = this.el('div', 'toolbar');
    const tools: Array<[ManualTool, string, string]> = [
      ['auto', '🖐️', '自动'],
      ['push', '👆', '推沙'],
      ['tap', '·', '点孔'],
      ['sweep', '🤚', '侧扫'],
      ['sprinkle', '✨', '撒沙'],
    ];
    tools.forEach(([tool, ico, label]) => {
      const b = this.el('button', tool === 'auto' ? 'active' : '');
      b.innerHTML = `<span class="ico">${ico}</span>${label}`;
      b.dataset.tool = tool;
      b.onclick = () => this.setTool(tool);
      toolbar.appendChild(b);
    });
    const sep = this.el('span', '', '');
    toolbar.appendChild(sep);
    [
      ['undo', '↩️', '撤销', () => this.cb.onUndo()],
      ['flip', '🔄', '翻转', () => this.cb.onFlip()],
      ['clear', '🧹', '清场', () => this.cb.onClear()],
    ].forEach(([cls, ico, label, fn]) => {
      const b = this.el('button', cls === 'clear' ? 'danger' : '');
      b.innerHTML = `<span class="ico">${ico}</span>${label}`;
      b.onclick = fn as () => void;
      toolbar.appendChild(b);
    });
    this.root.appendChild(toolbar);

    this.toastEl = this.el('div', 'toast');
    this.root.appendChild(this.toastEl);
  }

  setMode(mode: AppMode): void {
    this.mode = mode;
    this.root.querySelectorAll<HTMLButtonElement>('.mode-tabs button').forEach((b) => {
      b.classList.toggle('active', b.dataset.mode === mode);
    });
    this.cb.onMode(mode);
    this.renderPanel();
  }

  setTool(tool: ManualTool): void {
    this.manualTool = tool;
    this.root.querySelectorAll<HTMLButtonElement>('.toolbar button[data-tool]').forEach((b) => {
      b.classList.toggle('active', b.dataset.tool === tool);
    });
    this.cb.onTool(tool);
  }

  setStats(fps: number, resLabel: string): void {
    this.statsFps.textContent = String(fps);
    this.statsRes.textContent = resLabel;
  }

  toast(msg: string, ms = 1800): void {
    this.toastEl.textContent = msg;
    this.toastEl.classList.add('show');
    window.setTimeout(() => this.toastEl.classList.remove('show'), ms);
  }

  renderPanel(extra?: PanelExtra): void {
    const p = this.panelEl;
    p.innerHTML = '';
    switch (this.mode) {
      case 'practice':
        this.renderPractice(p, extra);
        break;
      case 'free':
        this.renderFree(p);
        break;
      case 'demo':
        this.renderDemo(p, extra);
        break;
      case 'settings':
        this.renderSettings(p, extra);
        break;
    }
  }

  private renderPractice(p: HTMLDivElement, extra?: PanelExtra): void {
    p.appendChild(this.el('h3', '', '手法练习'));
    p.appendChild(this.el('p', 'sub', '选择一项手法，按提示在灯箱上完成后点“评测”。系统回读沙层厚度判定，标准固定、可重复。'));
    LESSONS.forEach((lesson: Lesson) => {
      const card = this.el('div', 'lesson-card' + (extra?.activeLesson === lesson.id ? ' active' : ''));
      card.appendChild(this.el('div', 'name', lesson.name));
      card.appendChild(this.el('div', 'desc', lesson.instruction));
      card.onclick = () => this.cb.onSelectLesson(lesson.id);
      p.appendChild(card);
    });
    const row = this.el('div', 'row');
    const evalBtn = this.el('button', 'primary', '评测本次练习');
    evalBtn.onclick = () => this.cb.onEvaluate();
    row.appendChild(evalBtn);
    p.appendChild(row);
    if (extra?.resultHtml) {
      const rc = this.el('div', 'result-card ' + (extra.resultPass ? 'pass' : 'fail'));
      rc.innerHTML = extra.resultHtml;
      p.appendChild(rc);
    }
  }

  private renderFree(p: HTMLDivElement): void {
    p.appendChild(this.el('h3', '', '自由创作'));
    p.appendChild(this.el('p', 'sub', '不限手法，自由铺沙、推抹、点孔、撒沙。底部工具栏可锁定单一手法，默认自动识别。'));
    const tips = this.el('div', 'lesson-card');
    tips.innerHTML =
      '<div class="name">手法提示</div><div class="desc">· 单指慢推：沙子在两侧堆脊，超休止角会自动塌落<br>· 掌侧贴沙横扫：拉出明暗渐变<br>· 垂直轻点：透光亮孔<br>· 双指或长按后移动：撒沙<br>· 翻转按钮：画面 180° 旋转并带惯性滑动</div>';
    p.appendChild(tips);
  }

  private renderDemo(p: HTMLDivElement, extra?: { demos?: Demo[] }): void {
    p.appendChild(this.el('h3', '', '老师示范'));
    p.appendChild(this.el('p', 'sub', '老师录一笔，学生可回放观看，也可打开描红照着描。录像存本机，换沙层分辨率仍可回放。'));
    const row = this.el('div', 'row');
    const recBtn = this.el('button', this.recording ? 'primary' : 'ghost', this.recording ? '■ 停止录制' : '● 开始录制');
    recBtn.onclick = () => this.cb.onRecordToggle();
    row.appendChild(recBtn);
    const guideBtn = this.el('button', 'ghost', this.showGuide ? '描红：开' : '描红：关');
    guideBtn.onclick = () => this.cb.onGuideToggle();
    row.appendChild(guideBtn);
    p.appendChild(row);

    const demos = extra?.demos ?? [];
    if (demos.length === 0) {
      p.appendChild(this.el('p', 'sub', '还没有示范录像。点“开始录制”，在灯箱上示范一笔后停止。'));
    }
    demos.forEach((demo, i) => {
      const item = this.el('div', 'demo-item');
      item.appendChild(this.el('div', 'name', demo.name));
      item.appendChild(this.el('div', 'desc', `${(demo.durationMs / 1000).toFixed(1)} 秒 · ${demo.points.length} 个采样点`));
      const btnRow = this.el('div', 'row');
      const play = this.el('button', 'primary', '回放');
      play.onclick = () => this.cb.onPlayDemo(demo);
      const del = this.el('button', 'ghost', '删除');
      del.onclick = () => this.cb.onDeleteDemo(i);
      btnRow.appendChild(play);
      btnRow.appendChild(del);
      item.appendChild(btnRow);
      p.appendChild(item);
    });
  }

  private renderSettings(p: HTMLDivElement, extra?: { params?: SimParams; presetIndex?: number; resolutionIndex?: number }): void {
    p.appendChild(this.el('h3', '', '灯箱和沙子设置'));
    const params = extra?.params;
    if (!params) return;

    p.appendChild(this.el('p', 'sub', '快速预设'));
    PARAM_PRESETS.forEach((preset, i) => {
      const card = this.el('div', 'preset-card' + (extra.presetIndex === i ? ' active' : ''));
      card.appendChild(this.el('div', 'name', preset.name));
      card.appendChild(this.el('div', 'desc', preset.description));
      card.onclick = () => this.cb.onPreset(preset);
      p.appendChild(card);
    });

    p.appendChild(this.el('h3', '', '沙子物理'));
    this.slider(p, 'reposeDeg', '休止角（度）', params.reposeDeg, 20, 45, 0.5, (v) => `${v}°`);
    this.slider(p, 'relaxIters', '每帧雪崩迭代', params.relaxIters, 1, 8, 1, (v) => `${v} 次`);
    this.slider(p, 'density', '光学厚度（密度）', params.density, 0.2, 1.5, 0.01, (v) => v.toFixed(2));

    p.appendChild(this.el('h3', '', '灯箱光线'));
    this.slider(p, 'lightBrightness', '灯箱亮度', params.lightBrightness, 0.5, 1.5, 0.01, (v) => v.toFixed(2));
    this.slider(p, 'ambient', '环境漏光（暗部）', params.ambient, 0, 0.2, 0.005, (v) => v.toFixed(3));
    this.slider(p, 'grain', '颗粒感', params.grain, 0, 1, 0.01, (v) => v.toFixed(2));
    this.slider(p, 'bodyAmount', '厚沙本色混合', params.bodyAmount, 0, 0.6, 0.01, (v) => v.toFixed(2));

    const colorRow = this.el('div', 'swatch-row');
    colorRow.appendChild(this.colorInput('灯光颜色', params.lightColor, (c) => {
      params.lightColor = c;
      this.cb.onParamsChange({ ...params });
    }));
    colorRow.appendChild(this.colorInput('沙子本色', params.sandTint, (c) => {
      params.sandTint = c;
      this.cb.onParamsChange({ ...params });
    }));
    p.appendChild(colorRow);

    p.appendChild(this.el('h3', '', '模拟分辨率'));
    const sel = this.el('select') as HTMLSelectElement;
    RESOLUTION_OPTIONS.forEach((r, i) => {
      const opt = this.el('option') as HTMLOptionElement;
      opt.value = String(i);
      opt.textContent = r.label;
      if (i === extra.resolutionIndex) opt.selected = true;
      sel.appendChild(opt);
    });
    sel.onchange = () => this.cb.onResolution(Number(sel.value));
    p.appendChild(sel);
  }

  private slider(
    p: HTMLDivElement,
    key: keyof SimParams,
    label: string,
    value: number,
    min: number,
    max: number,
    step: number,
    fmt: (v: number) => string,
  ): void {
    const field = this.el('div', 'field');
    const lab = this.el('label');
    lab.innerHTML = `${label}<b>${fmt(value)}</b>`;
    const input = this.el('input') as HTMLInputElement;
    input.type = 'range';
    input.min = String(min);
    input.max = String(max);
    input.step = String(step);
    input.value = String(value);
    input.oninput = () => {
      const v = Number(input.value);
      lab.querySelector('b')!.textContent = fmt(v);
      (this.panelParams as SimParams)[key] = v as never;
      this.cb.onParamsChange({ ...(this.panelParams as SimParams) });
    };
    field.appendChild(lab);
    field.appendChild(input);
    p.appendChild(field);
    this.paramInputs.set(key, input);
  }

  private panelParams: SimParams | null = null;
  setPanelParams(params: SimParams): void {
    this.panelParams = params;
  }

  private colorInput(label: string, initial: [number, number, number], onChange: (c: [number, number, number]) => void): HTMLDivElement {
    const wrap = this.el('div', 'field');
    const lab = this.el('label', '', label);
    const input = this.el('input') as HTMLInputElement;
    input.type = 'color';
    const toHex = (c: [number, number, number]) =>
      '#' + c.map((v) => Math.round(Math.min(1, Math.max(0, v)) * 255).toString(16).padStart(2, '0')).join('');
    input.value = toHex(initial);
    input.oninput = () => {
      const hex = input.value;
      const c: [number, number, number] = [
        parseInt(hex.slice(1, 3), 16) / 255,
        parseInt(hex.slice(3, 5), 16) / 255,
        parseInt(hex.slice(5, 7), 16) / 255,
      ];
      onChange(c);
    };
    wrap.appendChild(lab);
    wrap.appendChild(input);
    return wrap;
  }
}
