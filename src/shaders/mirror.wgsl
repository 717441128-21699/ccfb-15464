// 快速翻转画面 pass（对应 CpuSandGrid.flip）。
// 引擎先做一次全网格 copy 建立源，再运行本 pass：
// 180° 旋转沙层写入主缓冲，并为每格注入“远离中心”的反向惯性速度，
// 随后的 inertia pass 让沙子按惯性再滑一下。

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = i32(gid.x);
  let y = i32(gid.y);
  if (x >= i32(sim.width) || y >= i32(sim.height)) {
    return;
  }
  let idx = u32(y) * sim.width + u32(x);
  let srcX = i32(sim.width) - 1 - x;
  let srcY = i32(sim.height) - 1 - y;
  let srcIdx = u32(srcY) * sim.width + u32(srcX);
  storeHeight(idx, readSource(srcIdx));

  let cx = f32(x) - f32(sim.width) * 0.5;
  let cy = f32(y) - f32(sim.height) * 0.5;
  let norm = max(sqrt(cx * cx + cy * cy), 0.0001);
  let impulse = sim.flipImpulse;
  let v = vec4<f32>(-(cx / norm) * impulse, -(cy / norm) * impulse, 0.0, 0.0);
  textureStore(velTex, vec2<i32>(x, y), v);
  markTile(x, y);
}
