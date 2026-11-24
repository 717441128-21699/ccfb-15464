// 沙画模拟的共享 WGSL 头：布局声明、定点常量、与 CPU 参考模型完全一致的工具函数。
// 每个 compute 着色器通过 ?raw 拼接本文件后再写自己的 main。

// 与 src/sim/constants.ts 保持一致
const FIXED_SHIFT: u32 = 12u;
const FIXED_SCALE: f32 = 4096.0;
const TILE_SIZE: u32 = 8u;
const MAX_SEGMENTS: u32 = 8u;

const TOOL_PUSH: u32 = 0u;
const TOOL_TAP: u32 = 1u;
const TOOL_SWEEP: u32 = 2u;
const TOOL_SPRINKLE: u32 = 3u;

const PASS_COPY: u32 = 0u;
const PASS_APPLY: u32 = 1u;

struct GpuSim {
  width: u32,
  height: u32,
  tilesX: u32,
  tilesY: u32,
  reposeDelta: u32,  // 休止角对应的定点高差阈值
  relaxIters: u32,
  density: f32,
  grain: f32,
  lightColor: vec3<f32>,
  lightBrightness: f32,
  ambient: f32,
  bodyAmount: f32,
  sandTint: vec3<f32>,
  pass: u32,         // PASS_COPY / PASS_APPLY
  segmentIndex: u32,
  frame: u32,
  useB: u32,         // 1 = 主缓冲是 B（写 B、只读源 A、活动 tileB）；0 反之
  flipImpulse: f32,
  velocityDamping: f32,
  originX: u32,      // 局部 dispatch 的全局原点（按 tile 对齐），全网格 pass 为 0
  originY: u32,
};

struct Stroke {
  x0: f32,
  y0: f32,
  x1: f32,
  y1: f32,
  radius: f32,
  pressure: f32,
  tool: u32,
  pushDist: f32,
};

struct StrokeUniform {
  count: u32,
  seed: u32,
  pad0: u32,
  pad1: u32,
  segments: array<Stroke, 8>,
};

// group 0：模拟资源（所有模拟 pass 共用同一个 bind group layout）
@group(0) @binding(0) var<uniform> sim: GpuSim;
@group(0) @binding(1) var<uniform> strokes: StrokeUniform;
@group(0) @binding(2) var<storage, read_write> heightA: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> heightB: array<atomic<u32>>;
// 速度只有 inertia pass 读写，原地更新即可，单纹理即可（省一份约 50MB 显存）
@group(0) @binding(4) var velTex: texture_storage_2d<rg16float, read_write>;
@group(0) @binding(5) var<storage, read_write> tileA: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> tileB: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read> heightReadOnly: array<u32>;
// group(1) 仅清场/渲染 pass 使用

fn clampf(v: f32, lo: f32, hi: f32) -> f32 {
  return min(max(v, lo), hi);
}

fn clampi(v: i32, lo: i32, hi: i32) -> i32 {
  return min(max(v, lo), hi);
}

fn smoothstepf(edge0: f32, edge1: f32, x: f32) -> f32 {
  let t = clampf((x - edge0) / (edge1 - edge0), 0.0, 1.0);
  return t * t * (3.0 - 2.0 * t);
}

// 与 src/sim/math.ts 的 hash21 同魔数、同回绕语义
fn hash21(x: i32, y: i32, seed: i32) -> f32 {
  var h: u32 = bitcast<u32>(x * 374761393) ^ bitcast<u32>(y * 668265263) ^ bitcast<u32>(seed * 1442695041);
  h = bitcast<u32>(bitcast<i32>(h ^ (h >> 13u)) * 1274126177);
  h = h ^ (h >> 16u);
  return f32(h) / 4294967295.0;
}

fn pointSegmentDist(px: f32, py: f32, x0: f32, y0: f32, x1: f32, y1: f32) -> f32 {
  let dx = x1 - x0;
  let dy = y1 - y0;
  let len2 = dx * dx + dy * dy;
  var t = 0.0;
  if (len2 > 0.000001) {
    t = clampf(((px - x0) * dx + (py - y0) * dy) / len2, 0.0, 1.0);
  }
  let cx = x0 + t * dx;
  let cy = y0 + t * dy;
  return distance(vec2<f32>(px, py), vec2<f32>(cx, cy));
}

fn tileIndex(x: u32, y: u32) -> u32 {
  return (y / TILE_SIZE) * sim.tilesX + (x / TILE_SIZE);
}

// 双缓冲抽象：useB=1 时主缓冲为 B、只读源为 A，活动 tile 为 tileB、下一轮写 tileA；
// useB=0 时相反。引擎在每个 pass 前设置 useB，各着色器只通过下面的辅助函数访问。
fn loadHeight(i: u32) -> u32 {
  if (sim.useB == 1u) {
    return atomicLoad(&heightB[i]);
  }
  return atomicLoad(&heightA[i]);
}

fn storeHeight(i: u32, v: u32) {
  if (sim.useB == 1u) {
    atomicStore(&heightB[i], v);
  } else {
    atomicStore(&heightA[i], v);
  }
}

fn addHeight(i: u32, v: u32) {
  if (sim.useB == 1u) {
    atomicAdd(&heightB[i], v);
  } else {
    atomicAdd(&heightA[i], v);
  }
}

fn subHeight(i: u32, v: u32) {
  if (sim.useB == 1u) {
    atomicSub(&heightB[i], v);
  } else {
    atomicSub(&heightA[i], v);
  }
}

fn readSource(i: u32) -> u32 {
  // heightReadOnly 由引擎在每个 pass 绑定到与主缓冲相反的只读源缓冲
  return heightReadOnly[i];
}

fn loadActiveTile(i: u32) -> u32 {
  if (sim.useB == 1u) {
    return atomicLoad(&tileB[i]);
  }
  return atomicLoad(&tileA[i]);
}

fn storeActiveTile(i: u32, v: u32) {
  if (sim.useB == 1u) {
    atomicStore(&tileB[i], v);
  } else {
    atomicStore(&tileA[i], v);
  }
}

fn storeNextTile(i: u32, v: u32) {
  if (sim.useB == 1u) {
    atomicStore(&tileA[i], v);
  } else {
    atomicStore(&tileB[i], v);
  }
}

fn markTile(x: i32, y: i32) {
  if (x >= 0 && y >= 0 && x < i32(sim.width) && y < i32(sim.height)) {
    let ti = (u32(y) / TILE_SIZE) * sim.tilesX + (u32(x) / TILE_SIZE);
    storeActiveTile(ti, 1u);
  }
}

fn inBounds(x: i32, y: i32) -> bool {
  return x >= 0 && y >= 0 && x < i32(sim.width) && y < i32(sim.height);
}
