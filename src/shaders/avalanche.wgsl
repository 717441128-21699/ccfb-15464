// 雪崩松弛 pass（对应 CpuSandGrid.avalanche）。
// 引擎连续 dispatch relaxIters 次，每次交换主/源缓冲与活动/下一 tile 缓冲。
// Jacobi 迭代：通量全部从只读源 readSource 独立计算，再原子汇回主缓冲，无竞争。
// 只处理活动 tile；仍有通量交换的 tile（含邻居）写入“下一 tile”缓冲，稳定 tile 自然淘汰。

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = u32(gid.x);
  let y = u32(gid.y);
  if (x >= sim.width || y >= sim.height) {
    return;
  }
  let ti = tileIndex(x, y);
  if (loadActiveTile(ti) == 0u) {
    return;
  }

  let idx = y * sim.width + x;
  let center = readSource(idx);
  let maxDelta = sim.reposeDelta;
  let K_DEN: u32 = 4u;
  var outflow: u32 = 0u;

  let offsets = array<vec2<i32>, 4>(
    vec2<i32>(-1, 0),
    vec2<i32>(1, 0),
    vec2<i32>(0, -1),
    vec2<i32>(0, 1),
  );

  for (var i: u32 = 0u; i < 4u; i = i + 1u) {
    let nx = i32(x) + offsets[i].x;
    let ny = i32(y) + offsets[i].y;
    if (nx < 0 || ny < 0 || nx >= i32(sim.width) || ny >= i32(sim.height)) {
      continue;
    }
    let nIdx = u32(ny) * sim.width + u32(nx);
    let n = readSource(nIdx);
    if (center > n) {
      let diff = center - n;
      if (diff > maxDelta) {
        var f = (diff - maxDelta) / K_DEN;
        let cap = diff / 2u;
        if (f > cap) {
          f = cap;
        }
        if (f > 0u) {
          outflow += f;
          addHeight(nIdx, f);
          let nti = (u32(ny) / TILE_SIZE) * sim.tilesX + (u32(nx) / TILE_SIZE);
          storeNextTile(nti, 1u);
        }
      }
    }
  }

  if (outflow > 0u) {
    subHeight(idx, outflow);
    storeNextTile(ti, 1u);
  }
}
