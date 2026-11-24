import { FIXED_SCALE } from './constants';

export const clamp = (v: number, lo: number, hi: number): number =>
  v < lo ? lo : v > hi ? hi : v;

export const lerp = (a: number, b: number, t: number): number => a + (b - a) * t;

export function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = clamp((x - edge0) / (edge1 - edge0), 0, 1);
  return t * t * (3 - 2 * t);
}

/** 定点高度 -> 厚度单元（浮点，供渲染） */
export const fixedToHeight = (fixed: number): number => fixed / FIXED_SCALE;

export const heightToFixed = (h: number): number => Math.round(h * FIXED_SCALE);

/**
 * 确定性二维哈希，CPU 与 WGSL 共用同一组魔数。
 * 输出 0..1。WGSL 中用 u32 溢出回绕实现同样的运算。
 */
export function hash21(x: number, y: number, seed: number): number {
  let h = Math.imul(x | 0, 374761393) ^ Math.imul(y | 0, 668265263) ^ Math.imul(seed | 0, 1442695041);
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  h = h ^ (h >>> 16);
  return (h >>> 0) / 4294967295;
}

/** 点到线段距离 */
export function pointSegmentDist(
  px: number,
  py: number,
  x0: number,
  y0: number,
  x1: number,
  y1: number,
): number {
  const dx = x1 - x0;
  const dy = y1 - y0;
  const len2 = dx * dx + dy * dy;
  let t = len2 === 0 ? 0 : ((px - x0) * dx + (py - y0) * dy) / len2;
  t = clamp(t, 0, 1);
  const cx = x0 + t * dx;
  const cy = y0 + t * dy;
  return Math.hypot(px - cx, py - cy);
}

/**
 * 统计一组亮度值中能分出多少个层次。
 * 亮度按 bins 个档位量化，统计被占据的档位数。
 * 侧扫渐变要求 >= 8 个可分辨层次。
 */
export function countLuminanceBands(values: ArrayLike<number>, bins = 32): number {
  const occupied = new Set<number>();
  for (let i = 0; i < values.length; i++) {
    const b = Math.floor(clamp(values[i], 0, 1) * bins);
    occupied.add(Math.min(b, bins - 1));
  }
  return occupied.size;
}

/** Beer-Lambert 透射率：沙越厚，透过的灯箱光越少 */
export function transmittance(heightUnits: number, density: number): number {
  return Math.exp(-heightUnits * density);
}

/** 线性光 -> sRGB gamma（与渲染着色器中的近似一致） */
export function linearToSrgb(c: number): number {
  c = clamp(c, 0, 1);
  return c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055;
}
