// 灯箱渲染 compute：把高度场渲染成 RGBA8 图像（对应需求“灯箱从底下往上打光”）。
// 物理模型：
//   透射率 T = exp(-厚度 * density)（Beer-Lambert，沙越厚透光越少）
//   出射光 = 灯箱暖光 * T + 环境漏光
//   厚沙处混入沙体本色；颗粒用确定性哈希噪声 + 坡度细碎明暗；最后转 sRGB。
// 输出到 rgba8unorm storage texture，再由 present.wgsl 缩放到画布。
@group(1) @binding(0) var outTex: texture_storage_2d<rgba8unorm, write>;

fn linearToSrgb(c: f32) -> f32 {
  let x = clampf(c, 0.0, 1.0);
  if (x <= 0.0031308) {
    return 12.92 * x;
  }
  return 1.055 * pow(x, 1.0 / 2.4) - 0.055;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = i32(gid.x);
  let y = i32(gid.y);
  if (x >= i32(sim.width) || y >= i32(sim.height)) {
    return;
  }
  let idx = u32(y) * sim.width + u32(x);
  let h = f32(loadHeight(idx)) / FIXED_SCALE; // 厚度单元

  // 中心差分求坡度，用于颗粒边缘的细碎明暗
  var hL = 0.0;
  var hR = 0.0;
  var hU = 0.0;
  var hD = 0.0;
  if (x > 0) { hL = f32(loadHeight(idx - 1u)) / FIXED_SCALE; }
  if (x < i32(sim.width) - 1) { hR = f32(loadHeight(idx + 1u)) / FIXED_SCALE; }
  if (y > 0) { hU = f32(loadHeight(idx - sim.width)) / FIXED_SCALE; }
  if (y < i32(sim.height) - 1) { hD = f32(loadHeight(idx + sim.width)) / FIXED_SCALE; }
  let slope = vec2<f32>(hR - hL, hD - hU);
  let slopeAmt = clampf(length(slope) * 0.5, 0.0, 1.0);

  // Beer-Lambert 透射
  let t = exp(-h * sim.density);
  var rgb = sim.lightColor * sim.lightBrightness * t;
  rgb = rgb + vec3<f32>(sim.ambient);

  // 厚沙混入沙体本色
  let bodyMix = clampf(1.0 - t, 0.0, 1.0) * sim.bodyAmount;
  rgb = mix(rgb, sim.sandTint, bodyMix);

  // 颗粒感：每格静态颗粒噪声（随帧轻微闪烁，模拟沙粒边缘的细碎高光）
  let grainStatic = hash21(x, y, 0) - 0.5;
  let grainFlicker = hash21(x + i32(sim.frame) * 131, y - i32(sim.frame) * 79, 11) - 0.5;
  let grainVal = (grainStatic * 0.7 + grainFlicker * 0.3) * sim.grain * (0.4 + 0.6 * slopeAmt + 0.3 * clampf(h, 0.0, 2.0));
  // 颗粒在有沙的地方才出现，薄沙处更亮、厚沙处更暗
  let sandMask = clampf(h * 1.5, 0.0, 1.0);
  rgb = rgb + vec3<f32>(grainVal * sandMask);

  // 坡度迎光侧微微提亮，模拟沙粒边缘透光
  let ridge = clampf(-(slope.x + slope.y) * 0.08, 0.0, 0.18) * sandMask;
  rgb = rgb + sim.lightColor * ridge;

  let out = vec4<f32>(
    linearToSrgb(rgb.r),
    linearToSrgb(rgb.g),
    linearToSrgb(rgb.b),
    1.0,
  );
  textureStore(outTex, vec2<i32>(x, y), out);
}
