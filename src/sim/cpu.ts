/**
 * 沙层模拟的 CPU 参考实现。
 *
 * 这里的每个函数都与 shaders/ 下的 WGSL compute 着色器一一对应、公式完全相同
 * （同样的 u32 定点数、同样的通量系数、同样的魔数哈希）。
 * GPU 无法在无显卡的 CI 里跑，因此物理性质（休止角、八级渐变、质量守恒…）
 * 全部通过本参考模型在 vitest 中验证。
 */
import {
  FIXED_SCALE,
  TILE_SIZE,
  Tool,
  reposeDeltaFixed,
  type SimParams,
  type StrokeSegment,
} from './constants';
import { clamp, hash21, pointSegmentDist, smoothstep } from './math';

export class CpuSandGrid {
  readonly width: number;
  readonly height: number;
  /** 当前高度场（u32 定点） */
  h: Uint32Array;
  /** 上一帧高度场（雪崩 Jacobi 迭代的只读源） */
  hPrev: Uint32Array;
  /** 速度场（f32，单位：纹理像素/帧），与 GPU 的 RG16F 纹理对应 */
  vx: Float32Array;
  vy: Float32Array;
  /** tile 脏标记，1 = 本帧该 tile 被改动或仍不稳定 */
  tiles: Uint8Array;
  readonly tilesX: number;
  readonly tilesY: number;

  constructor(width: number, height: number) {
    this.width = width;
    this.height = height;
    const n = width * height;
    this.h = new Uint32Array(n);
    this.hPrev = new Uint32Array(n);
    this.vx = new Float32Array(n);
    this.vy = new Float32Array(n);
    this.tilesX = Math.ceil(width / TILE_SIZE);
    this.tilesY = Math.ceil(height / TILE_SIZE);
    this.tiles = new Uint8Array(this.tilesX * this.tilesY);
  }

  clear(): void {
    this.h.fill(0);
    this.hPrev.fill(0);
    this.vx.fill(0);
    this.vy.fill(0);
    this.tiles.fill(0);
  }

  markTile(x: number, y: number): void {
    const tx = Math.floor(x / TILE_SIZE);
    const ty = Math.floor(y / TILE_SIZE);
    if (tx >= 0 && ty >= 0 && tx < this.tilesX && ty < this.tilesY) {
      this.tiles[ty * this.tilesX + tx] = 1;
    }
  }

  /** 小网格测试用：铺一层均匀沙 */
  fillLevel(levelUnits: number): void {
    this.h.fill(Math.round(levelUnits * FIXED_SCALE));
  }

  /** 小网格测试用：在某处放一个高斯沙堆 */
  addPile(cx: number, cy: number, radius: number, heightUnits: number): void {
    const r = Math.ceil(radius);
    for (let y = Math.max(0, cy - r); y <= Math.min(this.height - 1, cy + r); y++) {
      for (let x = Math.max(0, cx - r); x <= Math.min(this.width - 1, cx + r); x++) {
        const d = Math.hypot(x - cx, y - cy);
        const w = Math.exp(-(d * d) / (2 * (radius / 2) ** 2));
        this.h[y * this.width + x] += Math.round(heightUnits * FIXED_SCALE * w);
        this.markTile(x, y);
      }
    }
  }

  /**
   * 应用一段笔迹。GPU 上是“先复制包围盒再原子写”的两次 dispatch，
   * CPU 直接整网格复制一次即可，逻辑等价。
   */
  applyStroke(seg: StrokeSegment): void {
    this.hPrev.set(this.h);
    this.tiles.fill(0);
    switch (seg.tool) {
      case Tool.Push:
        this.applyPush(seg, false);
        break;
      case Tool.Sweep:
        this.applyPush(seg, true);
        break;
      case Tool.Tap:
        this.applyTap(seg);
        break;
      case Tool.Sprinkle:
        this.applySprinkle(seg);
        break;
    }
  }

  /**
   * 推沙 / 侧扫共用的 scatter 原子模型（GPU 上为 atomicSub + 3 个 atomicAdd）。
   * 对笔刷覆盖的每个格子，从只读的 hPrev 算出带走量 take：
   *   - 本格 -= take（手指抹走）
   *   - 60% 撒到前进方向 push 距离处（前缘堆积）
   *   - 20% / 20% 堆到笔迹两侧（沙脊，随后由雪崩松弛到休止角）
   * 沉积量之和 = take，严格质量守恒；宽笔刷侧扫经多帧平流 + 松弛形成明暗渐变。
   */
  private applyPush(seg: StrokeSegment, wide: boolean): void {
    const { width, height } = this;
    const r = seg.radius;
    const reach = r + seg.pushDist + 2;
    const minX = Math.max(0, Math.floor(Math.min(seg.x0, seg.x1) - reach));
    const maxX = Math.min(width - 1, Math.ceil(Math.max(seg.x0, seg.x1) + reach));
    const minY = Math.max(0, Math.floor(Math.min(seg.y0, seg.y1) - reach));
    const maxY = Math.min(height - 1, Math.ceil(Math.max(seg.y0, seg.y1) + reach));

    const dx = seg.x1 - seg.x0;
    const dy = seg.y1 - seg.y0;
    const segLen = Math.hypot(dx, dy) || 1;
    const ux = dx / segLen;
    const uy = dy / segLen;
    // 笔迹法向（指向左侧）
    const nx = -uy;
    const ny = ux;
    const push = seg.pushDist;
    const pickup = clamp(seg.pickup, 0, 1);
    // 侧扫笔刷更扁更软：边缘保留更多，渐变拉得更长
    const inner = wide ? r * 0.3 : r * 0.45;

    for (let y = minY; y <= maxY; y++) {
      for (let x = minX; x <= maxX; x++) {
        const d = pointSegmentDist(x, y, seg.x0, seg.y0, seg.x1, seg.y1);
        if (d > r) continue;
        const idx = y * width + x;
        const src = this.hPrev[idx];
        if (src === 0) continue;

        // 笔刷中心完全带走，边缘逐渐保留（软笔刷）
        const falloff = 1 - smoothstep(inner, r, d);
        const shape = wide ? 0.25 + 0.75 * falloff : 0.35 + 0.65 * falloff;
        const take = Math.round(src * pickup * shape);
        if (take <= 0) continue;

        // 本格抹走
        this.h[idx] -= take;
        this.markTile(x, y);

        // 60% 前缘堆积，整除余数并入前缘，保证 take 完整分配（严格守恒）
        const sideAmt = Math.floor((take - Math.round(take * 0.6)) / 2);
        const front = take - 2 * sideAmt;
        const fx = Math.round(x + ux * push);
        const fy = Math.round(y + uy * push);
        if (fx >= 0 && fy >= 0 && fx < width && fy < height) {
          this.h[fy * width + fx] += front;
          this.markTile(fx, fy);
        } else {
          this.h[idx] += front; // 前缘越界退回本格
        }

        // 剩余量 20% / 20% 堆到两侧沙脊
        for (const side of [-1, 1]) {
          const sx = Math.round(x + nx * side * (r * 0.92));
          const sy = Math.round(y + ny * side * (r * 0.92));
          if (sx >= 0 && sy >= 0 && sx < width && sy < height) {
            this.h[sy * width + sx] += sideAmt;
            this.markTile(sx, sy);
          } else {
            this.h[idx] += sideAmt;
          }
        }

        // 速度交给惯性 pass：方向与笔迹一致，幅度与推距、压感相关
        const speed = push * (0.5 + 0.5 * seg.pressure);
        this.vx[idx] = ux * speed;
        this.vy[idx] = uy * speed;
      }
    }
  }

  /**
   * 指尖点孔：清空小圆，沙量确定性地重新分布到孔沿 24 个环槽。
   * 两阶段：先清空孔内并把每个孔内格的沙按角度分配到“环槽累加器”，
   * 最后统一把环槽写入高度场。环槽按槽位索引累积，多格落到同一像素也不会丢沙，
   * 严格质量守恒（与 interaction.wgsl 的 applyTap 同构）。
   */
  private applyTap(seg: StrokeSegment): void {
    const { width, height } = this;
    const r = seg.radius;
    const cx = seg.x1;
    const cy = seg.y1;
    const minX = Math.max(0, Math.floor(cx - r - 3));
    const maxX = Math.min(width - 1, Math.ceil(cx + r + 3));
    const minY = Math.max(0, Math.floor(cy - r - 3));
    const maxY = Math.min(height - 1, Math.ceil(cy + r + 3));
    const RING_SLOTS = 24;
    const TWO_PI = Math.PI * 2;
    const ring = new Uint32Array(RING_SLOTS);

    // 阶段一：清空孔内，沙量进环槽累加器
    for (let y = minY; y <= maxY; y++) {
      for (let x = minX; x <= maxX; x++) {
        const idx = y * width + x;
        const d = Math.hypot(x - cx, y - cy);
        if (d > r * 0.8) continue;
        const v = this.h[idx];
        if (v === 0) continue;
        this.h[idx] = 0;
        this.markTile(x, y);
        const per = Math.floor(v / RING_SLOTS);
        for (let k = 0; k < RING_SLOTS; k++) ring[k] += per;
        // 余数给与本格方位最近的槽
        const angle = Math.atan2(y - cy, x - cx);
        const baseSlot = Math.floor(((angle + Math.PI) / TWO_PI) * RING_SLOTS) % RING_SLOTS;
        ring[baseSlot] += v - per * RING_SLOTS;
      }
    }

    // 阶段二：环槽写入孔沿；槽位像素若落回孔内或越界，沙退回中心，绝不丢失
    for (let k = 0; k < RING_SLOTS; k++) {
      const amount = ring[k];
      if (amount === 0) continue;
      const a = (k + 0.5) * (TWO_PI / RING_SLOTS);
      const sx = Math.min(width - 1, Math.max(0, Math.round(cx + Math.cos(a) * (r + 1))));
      const sy = Math.min(height - 1, Math.max(0, Math.round(cy + Math.sin(a) * (r + 1))));
      if (Math.hypot(sx - cx, sy - cy) <= r * 0.8) {
        this.h[Math.round(cy) * width + Math.round(cx)] += amount;
      } else {
        this.h[sy * width + sx] += amount;
        this.markTile(sx, sy);
      }
    }
  }

  /** 撒沙：沿笔迹按落点高斯落沙，带确定性随机散布 */
  private applySprinkle(seg: StrokeSegment): void {
    const { width, height } = this;
    const dx = seg.x1 - seg.x0;
    const dy = seg.y1 - seg.y0;
    const len = Math.hypot(dx, dy);
    const steps = Math.max(1, Math.ceil(len / Math.max(1, seg.radius * 0.5)));
    const baseAdd = Math.round(FIXED_SCALE * 0.9 * seg.pressure);
    for (let s = 0; s <= steps; s++) {
      const t = s / steps;
      const jx = (hash21(s, seg.seed & 0xffff, 7) - 0.5) * seg.radius * 1.6;
      const jy = (hash21(s, seg.seed & 0xffff, 19) - 0.5) * seg.radius * 1.6;
      const cx = seg.x0 + dx * t + jx;
      const cy = seg.y0 + dy * t + jy;
      const spread = seg.radius * (0.7 + 0.6 * hash21(s, seg.seed & 0xffff, 31));
      const r = Math.ceil(spread * 1.6);
      const add = Math.round(baseAdd * (0.6 + 0.8 * hash21(s, seg.seed & 0xffff, 43)));
      for (let y = Math.max(0, Math.floor(cy - r)); y <= Math.min(height - 1, Math.floor(cy + r)); y++) {
        for (let x = Math.max(0, Math.floor(cx - r)); x <= Math.min(width - 1, Math.floor(cx + r)); x++) {
          const d = Math.hypot(x - cx, y - cy);
          if (d > spread * 1.6) continue;
          const w = Math.exp(-(d * d) / (2 * (spread / 2) ** 2));
          const idx = y * width + x;
          this.h[idx] = Math.min(0xffffffff, this.h[idx] + Math.round(add * w));
          this.markTile(x, y);
        }
      }
    }
  }

  /**
   * 雪崩松弛（对应 avalanche.wgsl）。
   * Jacobi 迭代：四邻域各自从“上一缓冲”独立计算通量，再汇回本格。
   * 相邻格高差超过休止角阈值时产生通量，通量上限保证单步稳定。
   * 只处理脏 tile；稳定下来的 tile 清标记，后续帧不再参与计算。
   */
  avalanche(params: SimParams): number {
    const { width, height, tilesX } = this;
    const maxDelta = reposeDeltaFixed(params.reposeDeg);
    const K_NUM = 1; // 通量系数分子（K_DEN = 4）
    const K_DEN = 4;
    let activeTiles = 0;
    for (let i = 0; i < this.tiles.length; i++) {
      if (this.tiles[i] !== 0) activeTiles++;
    }
    if (activeTiles === 0) return 0;

    for (let iter = 0; iter < params.relaxIters; iter++) {
      this.hPrev.set(this.h);
      // 严格 Jacobi：next 从源快照复制，所有通量读 hPrev、只写 next，
      // 一格在本轮内不会读到被别的格污染的值（与 GPU 的“源只读、原子写主缓冲”等价）。
      const next = new Uint32Array(this.hPrev);
      const nextTiles = new Uint8Array(this.tiles.length);
      // 活动区向外扩一圈 halo tile：接收通量的邻居格可能落在当前活动 tile 之外。
      const workTiles = new Uint8Array(this.tiles);
      for (let ty = 0; ty < this.tilesY; ty++) {
        for (let tx = 0; tx < this.tilesX; tx++) {
          if (this.tiles[ty * tilesX + tx] === 0) continue;
          for (let dy = -1; dy <= 1; dy++) {
            for (let dx = -1; dx <= 1; dx++) {
              const nx = tx + dx;
              const ny = ty + dy;
              if (nx >= 0 && ny >= 0 && nx < this.tilesX && ny < this.tilesY) {
                workTiles[ny * tilesX + nx] = 1;
              }
            }
          }
        }
      }
      for (let ty = 0; ty < this.tilesY; ty++) {
        for (let tx = 0; tx < this.tilesX; tx++) {
          if (workTiles[ty * tilesX + tx] === 0) continue;
          const x0 = tx * TILE_SIZE;
          const y0 = ty * TILE_SIZE;
          for (let y = y0; y < Math.min(height, y0 + TILE_SIZE); y++) {
            for (let x = x0; x < Math.min(width, x0 + TILE_SIZE); x++) {
              const idx = y * width + x;
              const center = this.hPrev[idx];
              let outflow = 0;
              const neighbors: Array<[number, number]> = [
                [x - 1, y],
                [x + 1, y],
                [x, y - 1],
                [x, y + 1],
              ];
              for (const [nx, ny] of neighbors) {
                if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
                const nIdx = ny * width + nx;
                const n = this.hPrev[nIdx];
                const diff = center - n;
                if (diff > maxDelta) {
                  let f = Math.floor(((diff - maxDelta) * K_NUM) / K_DEN);
                  const cap = Math.floor(diff / 2);
                  if (f > cap) f = cap;
                  if (f > 0) {
                    outflow += f;
                    next[nIdx] += f; // 只写独立缓冲，不污染源
                    nextTiles[Math.floor(ny / TILE_SIZE) * tilesX + Math.floor(nx / TILE_SIZE)] = 1;
                  }
                }
              }
              if (outflow > 0) {
                next[idx] -= outflow;
                nextTiles[ty * tilesX + tx] = 1;
              }
            }
          }
        }
      }
      this.h = next;
      this.tiles = nextTiles;
    }
    return activeTiles;
  }

  /**
   * 惯性平流（对应 inertia.wgsl）。
   * 采用守恒的整数 scatter：每格把沙按速度送到整数邻居格，余数留在本格，
   * 边界钳制保证不流出画面。速度每帧按 damping 衰减并做绝对截止。
   */
  inertia(damping: number): void {
    const { width, height } = this;
    const nvx = new Float32Array(this.vx.length);
    const nvy = new Float32Array(this.vy.length);
    const next = new Uint32Array(this.h.length);
    for (let i = 0; i < this.vx.length; i++) {
      const ax = this.vx[i];
      const ay = this.vy[i];
      const dax = ax * damping;
      const day = ay * damping;
      nvx[i] = Math.abs(dax) < 0.05 ? 0 : dax;
      nvy[i] = Math.abs(day) < 0.05 ? 0 : day;
    }
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const idx = y * width + x;
        const v = this.h[idx];
        if (v === 0) continue;
        const ax = this.vx[idx];
        const ay = this.vy[idx];
        if (Math.abs(ax) < 0.02 && Math.abs(ay) < 0.02) {
          next[idx] += v;
          continue;
        }
        // 目标位置（整数），钳到边界
        const tx = clamp(Math.round(x + ax), 0, width - 1);
        const ty = clamp(Math.round(y + ay), 0, height - 1);
        next[ty * width + tx] += v;
      }
    }
    this.h = next;
    this.vx = nvx;
    this.vy = nvy;
  }

  /** 快速翻转画面：180° 旋转沙层并注入反向惯性速度（对应 mirror.wgsl） */
  flip(impulse: number): void {
    const { width, height } = this;
    const flipped = new Uint32Array(this.h.length);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const srcIdx = (height - 1 - y) * width + (width - 1 - x);
        const idx = y * width + x;
        flipped[idx] = this.h[srcIdx];
        // 翻转后沙子相对新画面朝“反方向”滑一下
        const cx = x - width / 2;
        const cy = y - height / 2;
        const norm = Math.hypot(cx, cy) || 1;
        this.vx[idx] = -(cx / norm) * impulse;
        this.vy[idx] = -(cy / norm) * impulse;
      }
    }
    this.h = flipped;
    this.tiles.fill(1);
  }

  /** 总沙量（定点单位），用于质量守恒测试 */
  totalSand(): number {
    let s = 0;
    for (let i = 0; i < this.h.length; i++) s += this.h[i];
    return s;
  }

  /** 当前最大相邻高差（定点单位），用于休止角测试 */
  maxNeighborDiff(): number {
    const { width, height } = this;
    let m = 0;
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const c = this.h[y * width + x];
        if (x + 1 < width) m = Math.max(m, Math.abs(c - this.h[y * width + x + 1]));
        if (y + 1 < height) m = Math.max(m, Math.abs(c - this.h[(y + 1) * width + x]));
      }
    }
    return m;
  }
}
