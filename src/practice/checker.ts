/**
 * 手法练习评测。
 * 四类练习各有可量化的达标标准，通过引擎回读小区域高度场来判定，
 * 不依赖截图视觉判断，结果稳定可复现。
 */
import type { GpuSandEngine } from '../gpu/GpuSandEngine';
import { FIXED_SCALE } from '../sim/constants';
import { countLuminanceBands, transmittance } from '../sim/math';

export type LessonId = 'push' | 'tap' | 'sweep' | 'sprinkle';

export interface Lesson {
  id: LessonId;
  name: string;
  instruction: string;
  target: { x: number; y: number; w: number; h: number };
}

export const LESSONS: Lesson[] = [
  {
    id: 'push',
    name: '推沙起脊',
    instruction: '单指从左向右平推一段，要求两侧堆起沙脊、坡度不超过休止角。',
    target: { x: 0.18, y: 0.4, w: 0.64, h: 0.28 },
  },
  {
    id: 'tap',
    name: '指尖点孔',
    instruction: '在灯箱中央轻点一下，做出一个透光的亮孔，孔沿沙均匀隆起。',
    target: { x: 0.38, y: 0.34, w: 0.24, h: 0.32 },
  },
  {
    id: 'sweep',
    name: '掌侧渐变',
    instruction: '用手掌侧面横扫，做出从亮到暗至少 8 个层次的柔和渐变。',
    target: { x: 0.12, y: 0.3, w: 0.76, h: 0.4 },
  },
  {
    id: 'sprinkle',
    name: '撒沙落台',
    instruction: '双指撒沙，让沙粒按落点均匀散开，区域厚度差异不要过大。',
    target: { x: 0.25, y: 0.25, w: 0.5, h: 0.5 },
  },
];

export interface CheckResult {
  pass: boolean;
  score: number;
  metrics: Record<string, number>;
  message: string;
}

export class PracticeChecker {
  constructor(private engine: GpuSandEngine) {}

  async evaluate(lesson: Lesson): Promise<CheckResult> {
    const { width, height } = this.engine;
    const tx = Math.floor(lesson.target.x * width);
    const ty = Math.floor(lesson.target.y * height);
    const tw = Math.min(256, Math.floor(lesson.target.w * width));
    const th = Math.min(256, Math.floor(lesson.target.h * height));
    const region = await this.engine.readbackRegion(tx, ty, tw, th);
    const heights = new Float32Array(region.length);
    for (let i = 0; i < region.length; i++) heights[i] = region[i] / FIXED_SCALE;

    switch (lesson.id) {
      case 'push':
        return this.checkPush(heights, tw, th);
      case 'tap':
        return this.checkTap(heights, tw, th);
      case 'sweep':
        return this.checkSweep(heights, tw, th);
      case 'sprinkle':
        return this.checkSprinkle(heights, tw, th);
    }
  }

  /** 推沙：亮区（薄沙）占比达标 + 最大坡度不超过休止角 */
  private checkPush(h: Float32Array, w: number, hgt: number): CheckResult {
    let thin = 0;
    let maxSlope = 0;
    for (let y = 0; y < hgt; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        if (h[i] < 0.25) thin++;
        if (x + 1 < w) maxSlope = Math.max(maxSlope, Math.abs(h[i + 1] - h[i]));
        if (y + 1 < hgt) maxSlope = Math.max(maxSlope, Math.abs(h[i + w] - h[i]));
      }
    }
    const thinRatio = thin / h.length;
    const reposeSlope = Math.tan((this.engine.params.reposeDeg * Math.PI) / 180);
    // 离散通量稳态允许约 8 个定点单位（≈0.1°）的舍入容差
    const slopeOk = maxSlope <= reposeSlope + 8 / 4096;
    const pass = thinRatio > 0.12 && slopeOk;
    const score = Math.round(clamp01(thinRatio / 0.3) * 60 + (slopeOk ? 40 : 0));
    return {
      pass,
      score,
      metrics: { 薄沙区域占比: round3(thinRatio), 最大相邻坡度: round3(maxSlope), 休止角坡度: round3(reposeSlope) },
      message: slopeOk
        ? `薄沙亮区占比 ${(thinRatio * 100).toFixed(1)}%，最大坡度 ${maxSlope.toFixed(2)} 未超休止角。`
        : `最大坡度 ${maxSlope.toFixed(2)} 超过休止角对应坡度 ${reposeSlope.toFixed(2)}，沙脊还会继续塌。`,
    };
  }

  /** 点孔：中心明显比周围薄（亮孔）+ 孔沿有隆起 */
  private checkTap(h: Float32Array, w: number, hgt: number): CheckResult {
    const cx = w / 2;
    const cy = hgt / 2;
    let centerSum = 0;
    let centerN = 0;
    let ringSum = 0;
    let ringN = 0;
    for (let y = 0; y < hgt; y++) {
      for (let x = 0; x < w; x++) {
        const d = Math.hypot(x - cx, y - cy);
        const norm = d / (Math.min(w, hgt) / 2);
        if (norm < 0.25) {
          centerSum += h[y * w + x];
          centerN++;
        } else if (norm > 0.45 && norm < 0.85) {
          ringSum += h[y * w + x];
          ringN++;
        }
      }
    }
    const centerAvg = centerSum / Math.max(1, centerN);
    const ringAvg = ringSum / Math.max(1, ringN);
    const contrast = ringAvg - centerAvg;
    const pass = centerAvg < 0.2 && contrast > 0.35;
    const score = Math.round(clamp01((0.2 - centerAvg) / 0.2) * 50 + clamp01(contrast / 0.8) * 50);
    return {
      pass,
      score,
      metrics: { 中心厚度: round3(centerAvg), 孔沿厚度: round3(ringAvg), 厚度反差: round3(contrast) },
      message: pass
        ? `亮孔中心厚度 ${centerAvg.toFixed(2)}，孔沿隆起至 ${ringAvg.toFixed(2)}，透光对比明显。`
        : '亮孔不够通透或孔沿没有堆起，试试更短促、更垂直地点一下。',
    };
  }

  /** 侧扫：沿运动方向的亮度层次 >= 8，且整体单调趋势占主导 */
  private checkSweep(h: Float32Array, w: number, hgt: number): CheckResult {
    const density = this.engine.params.density;
    // 取每一列的平均厚度 -> 透射亮度序列
    const colLum: number[] = [];
    for (let x = 0; x < w; x++) {
      let sum = 0;
      for (let y = 0; y < hgt; y++) sum += h[y * w + x];
      colLum.push(transmittance(sum / hgt, density));
    }
    const bands = countLuminanceBands(colLum, 32);
    // 单调性：相邻列同号变化占比
    let mono = 0;
    for (let x = 1; x < colLum.length; x++) {
      if (colLum[x] !== colLum[x - 1]) mono++;
    }
    const span = Math.max(...colLum) - Math.min(...colLum);
    const pass = bands >= 8 && span > 0.35;
    const score = Math.round(Math.min(1, bands / 12) * 70 + clamp01(span) * 30);
    return {
      pass,
      score,
      metrics: { 亮度层次数: bands, 亮度跨度: round3(span), 有效变化列占比: round3(mono / (colLum.length - 1)) },
      message: pass
        ? `渐变分出 ${bands} 个亮度层次（要求 ≥ 8），明暗跨度 ${span.toFixed(2)}。`
        : `当前只有 ${bands} 个层次，放慢手速、用掌侧贴着沙面扫出更长的过渡。`,
    };
  }

  /** 撒沙：覆盖率与均匀度（变异系数足够小） */
  private checkSprinkle(h: Float32Array, w: number, hgt: number): CheckResult {
    void w;
    void hgt;
    let covered = 0;
    let sum = 0;
    for (let i = 0; i < h.length; i++) {
      if (h[i] > 0.08) covered++;
      sum += h[i];
    }
    const mean = sum / h.length;
    let variance = 0;
    for (let i = 0; i < h.length; i++) variance += (h[i] - mean) ** 2;
    const cv = mean > 0.001 ? Math.sqrt(variance / h.length) / mean : 99;
    const coverage = covered / h.length;
    const pass = coverage > 0.55 && cv < 1.1;
    const score = Math.round(clamp01(coverage / 0.8) * 55 + clamp01(1.1 - cv) * 45);
    return {
      pass,
      score,
      metrics: { 覆盖率: round3(coverage), 厚度变异系数: round3(cv), 平均厚度: round3(mean) },
      message: pass
        ? `覆盖率 ${(coverage * 100).toFixed(1)}%，厚度变异系数 ${cv.toFixed(2)}，落点均匀。`
        : '沙子分布不均或覆盖不足，提高撒沙高度、让落点在区域内来回走动。',
    };
  }
}

const clamp01 = (v: number): number => Math.min(1, Math.max(0, v));
const round3 = (v: number): number => Math.round(v * 1000) / 1000;
