// 全屏三角形：把模拟分辨率的渲染纹理等比（contain）缩放到平板画布。
// 所有坐标统一在片元阶段用帧缓冲坐标（原点左上、y 向下）计算，避免 NDC 与帧缓冲 y 方向混淆。
struct PresentUniform {
  viewport: vec4<f32>, // x, y, w, h（帧缓冲像素，左上原点）
  canvasSize: vec2<f32>,
  pad0: f32,
  pad1: f32,
};
@group(0) @binding(0) var<uniform> pu: PresentUniform;
@group(0) @binding(1) var simTex: texture_2d<f32>;
@group(0) @binding(2) var simSampler: sampler;

struct VsOut {
  @builtin(position) pos: vec4<f32>,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VsOut {
  var p = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -3.0),
    vec2<f32>(3.0, 1.0),
    vec2<f32>(-1.0, 1.0),
  );
  var out: VsOut;
  out.pos = vec4<f32>(p[vi], 0.0, 1.0);
  return out;
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
  // 帧缓冲坐标：原点左上、y 向下，与 PresentUniform.viewport 同坐标系
  let px = in.pos.xy;
  let inside =
    px.x >= pu.viewport.x &&
    px.x <= pu.viewport.x + pu.viewport.z &&
    px.y >= pu.viewport.y &&
    px.y <= pu.viewport.y + pu.viewport.w;
  if (!inside) {
    return vec4<f32>(0.012, 0.01, 0.008, 1.0);
  }
  let local = (px - pu.viewport.xy) / pu.viewport.zw;
  // 模拟纹理由 compute textureStore 写入，其 texel (0,0) 在左上，
  // 与帧缓冲同方向，因此采样 uv 直接用 local，无需翻转。
  let uv = clamp(local, vec2<f32>(0.0), vec2<f32>(1.0));
  return textureSample(simTex, simSampler, uv);
}
