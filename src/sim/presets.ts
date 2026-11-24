import type { SimParams } from './constants';
import { SIM_WIDTH, SIM_HEIGHT } from './constants';

export interface ResolutionOption {
  label: string;
  width: number;
  height: number;
}

/** 模拟分辨率选项。课堂平板默认全精度 4096x3072，低配设备可降级。 */
export const RESOLUTION_OPTIONS: ResolutionOption[] = [
  { label: '4096 × 3072（全精度灯箱）', width: SIM_WIDTH, height: SIM_HEIGHT },
  { label: '2048 × 1536（流畅）', width: 2048, height: 1536 },
  { label: '1024 × 768（省电）', width: 1024, height: 768 },
];

/**
 * 四组老师可调的灯箱/沙子参数预设。
 * 每组都给出了不同的课堂气质：标准暖黄、高亮高对比、粗砂、细砂柔光。
 */
export interface Preset {
  name: string;
  description: string;
  params: SimParams;
}

export const PARAM_PRESETS: Preset[] = [
  {
    name: '暖黄灯箱 · 标准课堂',
    description: '经典沙画台暖光，休止角 34°，颗粒适中，最接近真实灯箱。',
    params: {
      reposeDeg: 34,
      relaxIters: 3,
      density: 0.62,
      lightColor: [1.0, 0.74, 0.4],
      lightBrightness: 1.05,
      ambient: 0.035,
      grain: 0.55,
      sandTint: [0.52, 0.35, 0.18],
      bodyAmount: 0.28,
    },
  },
  {
    name: '高亮灯箱 · 强对比',
    description: '灯箱更亮更白，薄沙处近乎全透，厚沙压得更暗，适合演示明暗节奏。',
    params: {
      reposeDeg: 34,
      relaxIters: 4,
      density: 0.78,
      lightColor: [1.0, 0.88, 0.68],
      lightBrightness: 1.25,
      ambient: 0.02,
      grain: 0.4,
      sandTint: [0.45, 0.3, 0.16],
      bodyAmount: 0.22,
    },
  },
  {
    name: '粗砂 · 明显颗粒',
    description: '颗粒感拉满、休止角更大（粗砂堆得更陡），适合练撒沙与点孔。',
    params: {
      reposeDeg: 38,
      relaxIters: 3,
      density: 0.55,
      lightColor: [1.0, 0.7, 0.36],
      lightBrightness: 1.0,
      ambient: 0.045,
      grain: 0.9,
      sandTint: [0.58, 0.4, 0.22],
      bodyAmount: 0.32,
    },
  },
  {
    name: '细砂 · 柔光渐变',
    description: '细砂休止角小、塌得平顺，颗粒细腻，侧扫渐变层次最柔和。',
    params: {
      reposeDeg: 29,
      relaxIters: 5,
      density: 0.66,
      lightColor: [1.0, 0.8, 0.55],
      lightBrightness: 0.95,
      ambient: 0.05,
      grain: 0.22,
      sandTint: [0.5, 0.36, 0.22],
      bodyAmount: 0.25,
    },
  },
];

export function cloneParams(p: SimParams): SimParams {
  return {
    ...p,
    lightColor: [...p.lightColor] as [number, number, number],
    sandTint: [...p.sandTint] as [number, number, number],
  };
}
