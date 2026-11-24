// 清场 pass（对应 CpuSandGrid.clear）。
// mode 0：只清主高度缓冲；mode 1：高度 + 主速度纹理 + 两组 tile 标记全部清零（重新开课）。
struct ClearUniform {
  mode: u32,
  pad0: u32,
  pad1: u32,
  pad2: u32,
};
@group(1) @binding(0) var<uniform> clearU: ClearUniform;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = i32(gid.x);
  let y = i32(gid.y);
  if (x >= i32(sim.width) || y >= i32(sim.height)) {
    return;
  }
  let idx = u32(y) * sim.width + u32(x);
  storeHeight(idx, 0u);
  if (clearU.mode == 1u) {
    let zero = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    textureStore(velTex, vec2<i32>(x, y), zero);
    let ti = tileIndex(u32(x), u32(y));
    atomicStore(&tileA[ti], 0u);
    atomicStore(&tileB[ti], 0u);
  }
}
