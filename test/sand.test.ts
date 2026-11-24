import { describe, expect, it } from 'vitest';
import { CpuSandGrid } from '../src/sim/cpu';
import { FIXED_SCALE, Tool, reposeDeltaFixed, type SimParams, type StrokeSegment } from '../src/sim/constants';
import { PARAM_PRESETS } from '../src/sim/presets';
import { countLuminanceBands, hash21, transmittance } from '../src/sim/math';

const baseParams: SimParams = PARAM_PRESETS[0].params;
const params = (over: Partial<SimParams> = {}): SimParams => ({ ...baseParams, ...over });

const seg = (s: Partial<StrokeSegment>): StrokeSegment => ({
  x0: 40,
  y0: 32,
  x1: 60,
  y1: 32,
  radius: 8,
  pressure: 0.8,
  tool: Tool.Push,
  pushDist: 6,
  pickup: 0.85,
  seed: 42,
  ...s,
});

describe('定点数与哈希', () => {
  it('休止角阈值随角度单调增大', () => {
    const d29 = reposeDeltaFixed(29);
    const d34 = reposeDeltaFixed(34);
    const d38 = reposeDeltaFixed(38);
    expect(d29).toBeLessThan(d34);
    expect(d34).toBeLessThan(d38);
    // 34° 的 tan ≈ 0.6745，定点阈值约 2763
    expect(d34).toBeGreaterThan(2400);
    expect(d34).toBeLessThan(3000);
  });

  it('hash21 确定性且分布在 0..1', () => {
    expect(hash21(3, 7, 99)).toBe(hash21(3, 7, 99));
    for (let i = 0; i < 50; i++) {
      const v = hash21(i, i * 3, i * 7);
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(1);
    }
  });
});

describe('雪崩松弛与休止角（需求 1）', () => {
  it('超过休止角的陡坡会自动塌落到阈值以内', () => {
    const grid = new CpuSandGrid(96, 96);
    // 制造一个远超休止角的陡峭沙柱
    grid.addPile(48, 48, 4, 12);
    const threshold = reposeDeltaFixed(baseParams.reposeDeg);
    expect(grid.maxNeighborDiff()).toBeGreaterThan(threshold);

    // 连续多帧松弛，模拟真实课堂里沙堆自己稳定下来的过程
    for (let frame = 0; frame < 40; frame++) {
      grid.avalanche(params({ relaxIters: 4 }));
    }
    // 离散通量稳态：通量在高差落到阈值附近的一个步长内停止，
    // 阈值 2764 + 8 = 2772，对应角度 tan⁻¹(2772/4096) ≈ 34.07°，与 34° 休止角一致
    expect(grid.maxNeighborDiff()).toBeLessThanOrEqual(threshold + 8);
  });

  it('不同休止角预设下，粗砂（38°）稳态比细砂（29°）更陡', () => {
    const settle = (deg: number): number => {
      const g = new CpuSandGrid(80, 80);
      g.addPile(40, 40, 3, 10);
      const p = params({ reposeDeg: deg, relaxIters: 5 });
      for (let f = 0; f < 40; f++) g.avalanche(p);
      return g.maxNeighborDiff();
    };
    expect(settle(38)).toBeGreaterThan(settle(29));
  });

  it('没有脏 tile 时雪崩直接跳过（性能特性）', () => {
    const grid = new CpuSandGrid(32, 32);
    expect(grid.avalanche(params())).toBe(0);
  });

  it('雪崩过程总沙量守恒（仅整除舍入误差）', () => {
    const grid = new CpuSandGrid(80, 80);
    grid.addPile(40, 40, 5, 10);
    const before = grid.totalSand();
    for (let f = 0; f < 30; f++) grid.avalanche(params({ relaxIters: 4 }));
    const after = grid.totalSand();
    expect(Math.abs(after - before)).toBeLessThanOrEqual(before * 0.002 + 4);
  });
});

describe('推沙与两侧沙脊（需求 1）', () => {
  it('推过的路径变薄、两侧与前缘出现堆积，且总沙量守恒', () => {
    const grid = new CpuSandGrid(96, 80);
    grid.fillLevel(3);
    const before = grid.totalSand();
    // 连续推几笔
    for (let k = 0; k < 6; k++) {
      grid.applyStroke(seg({ x0: 20 + k * 8, y0: 40, x1: 28 + k * 8, y1: 40 }));
      grid.avalanche(params({ relaxIters: 3 }));
    }
    // 路径中心平均厚度明显低于初始
    const pathAvg = avgAlong(grid, 24, 64, 40);
    expect(pathAvg).toBeLessThan(2.4);
    // 沙脊（路径上方一行）比路径厚
    const ridgeAvg = avgAlong(grid, 24, 64, 30);
    expect(ridgeAvg).toBeGreaterThan(pathAvg);
    const after = grid.totalSand();
    expect(after).toBe(before); // scatter 模型严格守恒
  });

  it('推沙后稳态坡度不超过休止角', () => {
    const grid = new CpuSandGrid(96, 96);
    grid.fillLevel(4);
    for (let k = 0; k < 8; k++) {
      grid.applyStroke(seg({ x0: 18 + k * 7, y0: 48, x1: 26 + k * 7, y1: 48, radius: 9 }));
      grid.avalanche(params({ relaxIters: 4 }));
    }
    for (let f = 0; f < 40; f++) grid.avalanche(params({ relaxIters: 4 }));
    expect(grid.maxNeighborDiff()).toBeLessThanOrEqual(reposeDeltaFixed(baseParams.reposeDeg) + 8);
  });
});

describe('掌侧侧扫的明暗渐变（需求 1：至少 8 个层次）', () => {
  it('侧扫后沿运动方向的透射亮度可分出 8 个以上层次', () => {
    const grid = new CpuSandGrid(160, 120);
    grid.fillLevel(3.2);
    // 宽笔刷横扫，多帧叠加形成从亮到暗的过渡
    for (let k = 0; k < 14; k++) {
      grid.applyStroke(
        seg({
          tool: Tool.Sweep,
          x0: 20 + k * 8,
          y0: 60,
          x1: 28 + k * 8,
          y1: 60,
          radius: 22,
          pushDist: 10,
          pickup: 0.7,
        }),
      );
      grid.avalanche(params({ relaxIters: 4 }));
    }
    for (let f = 0; f < 16; f++) grid.avalanche(params({ relaxIters: 4 }));

    // 沿 x 方向取列平均厚度 -> 灯箱透射亮度
    const lum: number[] = [];
    for (let x = 16; x < 144; x++) {
      let sum = 0;
      for (let y = 40; y < 80; y++) sum += grid.h[y * grid.width + x] / FIXED_SCALE;
      lum.push(transmittance(sum / 40, baseParams.density));
    }
    const bands = countLuminanceBands(lum, 32);
    expect(bands).toBeGreaterThanOrEqual(8);
    // 明暗跨度足够大，不是“一片死黑/一片死亮”
    expect(Math.max(...lum) - Math.min(...lum)).toBeGreaterThan(0.3);
  });
});

describe('指尖点孔', () => {
  it('点出亮孔（中心清零）且孔沿隆起，沙量守恒', () => {
    const grid = new CpuSandGrid(80, 80);
    grid.fillLevel(4);
    const before = grid.totalSand();
    grid.applyStroke(
      seg({ tool: Tool.Tap, x0: 40, y0: 40, x1: 40, y1: 40, radius: 8, pickup: 1, pushDist: 0 }),
    );
    const center = grid.h[40 * grid.width + 40] / FIXED_SCALE;
    expect(center).toBe(0);
    // 孔沿环带（孔半径 8 附近）取最大值，沙被确定性散射到 24 个环槽
    let rim = 0;
    for (let y = 30; y < 51; y++) {
      for (let x = 30; x < 51; x++) {
        const d = Math.hypot(x - 40, y - 40);
        if (d >= 7.5 && d <= 11) rim = Math.max(rim, grid.h[y * grid.width + x] / FIXED_SCALE);
      }
    }
    expect(rim).toBeGreaterThan(4.5);
    expect(grid.totalSand()).toBe(before);
    // 松弛后依旧稳定
    for (let f = 0; f < 20; f++) grid.avalanche(params({ relaxIters: 4 }));
    expect(grid.h[40 * grid.width + 40]).toBe(0);
  });

  it('点孔在手势实际上限半径 7 下同样守恒且成孔（覆盖 GPU 单 workgroup 场景）', () => {
    const grid = new CpuSandGrid(80, 80);
    grid.fillLevel(4);
    const before = grid.totalSand();
    grid.applyStroke(
      seg({ tool: Tool.Tap, x0: 40, y0: 40, x1: 40, y1: 40, radius: 7, pickup: 1, pushDist: 0 }),
    );
    expect(grid.h[40 * grid.width + 40]).toBe(0);
    expect(grid.totalSand()).toBe(before);
    let rim = 0;
    for (let y = 30; y < 51; y++) {
      for (let x = 30; x < 51; x++) {
        const d = Math.hypot(x - 40, y - 40);
        if (d >= 6.5 && d <= 10) rim = Math.max(rim, grid.h[y * grid.width + x] / FIXED_SCALE);
      }
    }
    expect(rim).toBeGreaterThan(4.5);
  });
});

describe('撒沙', () => {
  it('沿笔迹落沙并按落点散开，落点之外不被污染', () => {
    const grid = new CpuSandGrid(120, 120);
    grid.applyStroke(
      seg({
        tool: Tool.Sprinkle,
        x0: 30,
        y0: 60,
        x1: 90,
        y1: 60,
        radius: 10,
        pressure: 0.9,
        pickup: 0,
        pushDist: 0,
        seed: 1234,
      }),
    );
    let covered = 0;
    for (let y = 40; y < 80; y++) {
      for (let x = 20; x < 100; x++) {
        if (grid.h[y * grid.width + x] > 0) covered++;
      }
    }
    expect(covered).toBeGreaterThan(200);
    // 远离落点的角落保持干净
    expect(grid.h[2 * grid.width + 2]).toBe(0);
    expect(grid.h[118 * grid.width + 118]).toBe(0);
  });
});

describe('惯性与翻转', () => {
  it('翻转做 180° 旋转且沙量守恒，并注入可衰减的惯性速度', () => {
    const grid = new CpuSandGrid(64, 64);
    grid.addPile(16, 16, 4, 8);
    const before = grid.totalSand();
    grid.flip(6);
    // 原左上角的沙堆翻到右下角
    expect(grid.h[(64 - 1 - 16) * 64 + (64 - 1 - 16)]).toBeGreaterThan(0);
    expect(grid.h[16 * 64 + 16]).toBe(0);
    expect(grid.totalSand()).toBe(before);
    // 注入了速度
    let moving = 0;
    for (let i = 0; i < grid.vx.length; i++) {
      if (Math.abs(grid.vx[i]) > 0.1 || Math.abs(grid.vy[i]) > 0.1) moving++;
    }
    expect(moving).toBeGreaterThan(0);
    // 惯性多帧后速度衰减到近乎停止（截止阈值与 CPU 惯性实现一致）
    for (let f = 0; f < 40; f++) grid.inertia(0.86);
    let stillMoving = 0;
    for (let i = 0; i < grid.vx.length; i++) {
      if (Math.abs(grid.vx[i]) > 0.05 || Math.abs(grid.vy[i]) > 0.05) stillMoving++;
    }
    expect(stillMoving).toBe(0);
    expect(grid.totalSand()).toBe(before);
  });
});

describe('清场与渲染层次工具', () => {
  it('clear 完全清空', () => {
    const grid = new CpuSandGrid(48, 48);
    grid.addPile(24, 24, 6, 9);
    grid.clear();
    expect(grid.totalSand()).toBe(0);
    expect(grid.tiles.every((t) => t === 0)).toBe(true);
  });

  it('Beer-Lambert 透射率随厚度单调下降', () => {
    expect(transmittance(0, 0.62)).toBeCloseTo(1, 5);
    const t1 = transmittance(1, 0.62);
    const t3 = transmittance(3, 0.62);
    const t6 = transmittance(6, 0.62);
    expect(t1).toBeGreaterThan(t3);
    expect(t3).toBeGreaterThan(t6);
    expect(t6).toBeLessThan(0.05);
  });

  it('亮度分档工具能区分渐变序列', () => {
    const ramp = Array.from({ length: 40 }, (_, i) => i / 39);
    expect(countLuminanceBands(ramp, 32)).toBeGreaterThanOrEqual(30);
    expect(countLuminanceBands(new Array(40).fill(0.5), 32)).toBe(1);
  });
});

function avgAlong(grid: CpuSandGrid, x0: number, x1: number, y: number): number {
  let sum = 0;
  let n = 0;
  for (let x = x0; x <= x1; x++) {
    sum += grid.h[y * grid.width + x] / FIXED_SCALE;
    n++;
  }
  return sum / n;
}
