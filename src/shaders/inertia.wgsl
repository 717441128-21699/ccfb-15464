// 惯性平流 pass（对应 CpuSandGrid.inertia）。
// 守恒的整数 scatter：每格从只读源读出沙量，按速度原子加到整数目标格，
// 目标钳到边界，不丢沙也不造沙。速度纹理只有这一个 pass 访问，直接原地读改写。
// 引擎在本 pass 前先 copygrid 建源并交换主源，因此 readSource 是上一帧、主缓冲是干净的。

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = i32(gid.x);
  let y = i32(gid.y);
  if (x >= i32(sim.width) || y >= i32(sim.height)) {
    return;
  }
  let coord = vec2<i32>(x, y);
  let idx = u32(y) * sim.width + u32(x);

  // 原地读速度
  let vel = textureLoad(velTex, coord);
  let ax = vel.x;
  let ay = vel.y;

  // 速度衰减 + 绝对截止，原地写回
  var dx = ax * sim.velocityDamping;
  var dy = ay * sim.velocityDamping;
  if (abs(dx) < 0.05) { dx = 0.0; }
  if (abs(dy) < 0.05) { dy = 0.0; }
  textureStore(velTex, coord, vec4<f32>(dx, dy, 0.0, 0.0));

  // 守恒 scatter：源沙量整体搬到整数目标格
  let v = readSource(idx);
  if (v == 0u) {
    return;
  }
  if (abs(ax) < 0.02 && abs(ay) < 0.02) {
    addHeight(idx, v);
    return;
  }
  let tx = clampi(i32(round(f32(x) + ax)), 0, i32(sim.width) - 1);
  let ty = clampi(i32(round(f32(y) + ay)), 0, i32(sim.height) - 1);
  addHeight(u32(ty) * sim.width + u32(tx), v);
  markTile(tx, ty);
}
