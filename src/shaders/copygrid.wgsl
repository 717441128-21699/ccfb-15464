// 全网格建源 pass：把“源高度缓冲”复制到“主高度缓冲”。
// 用于雪崩/惯性/翻转前建立 Jacobi 只读源：先让主=源（同值），随后交换主源，
// 计算 pass 从只读源读、把结果写入新的主缓冲。
// 直接按 useB 访问物理缓冲，语义明确，不依赖 readSource 的绑定方向。
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = u32(gid.x);
  let y = u32(gid.y);
  if (x >= sim.width || y >= sim.height) {
    return;
  }
  let idx = y * sim.width + x;
  if (sim.useB == 1u) {
    // 主 B、源 A：B <- A
    atomicStore(&heightB[idx], atomicLoad(&heightA[idx]));
  } else {
    // 主 A、源 B：A <- B
    atomicStore(&heightA[idx], atomicLoad(&heightB[idx]));
  }
}
