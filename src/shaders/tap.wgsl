// 指尖点孔专用 pass：单 workgroup（16x16=256 线程）+ workgroup 共享的 24 环槽累加器。
// 两阶段、严格质量守恒：
//   1) 所有线程清零环槽 -> 孔内格清空并把沙量按角度原子累加进环槽
//   2) 栅栏同步后，线程分工把环槽写到孔沿；落回孔内/越界的沙退回中心
// 引擎对点孔按 16x16 包围盒只 dispatch 一个 workgroup，并通过 origin 定位。
// 公式与 CpuSandGrid.applyTap 同构。

var<workgroup> ringSlots: array<atomic<u32>, 24>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let x = i32(gid.x) + i32(sim.originX);
  let y = i32(gid.y) + i32(sim.originY);
  let seg = strokes.segments[sim.segmentIndex];
  let r = seg.radius;
  let cx = seg.x1;
  let cy = seg.y1;

  // 线程分工清零环槽（256 线程覆盖 24 槽）
  for (var k: u32 = lid.x + lid.y * 16u; k < 24u; k = k + 256u) {
    atomicStore(&ringSlots[k], 0u);
  }
  workgroupBarrier();

  // 阶段一：孔内格清空，沙量进环槽累加器
  if (x < i32(sim.width) && y < i32(sim.height)) {
    let idx = u32(y) * sim.width + u32(x);
    let dist = distance(vec2<f32>(f32(x), f32(y)), vec2<f32>(cx, cy));
    if (dist <= r * 0.8) {
      let v = readSource(idx);
      if (v > 0u) {
        subHeight(idx, v);
        markTile(x, y);
        let per = v / 24u;
        for (var k: u32 = 0u; k < 24u; k = k + 1u) {
          atomicAdd(&ringSlots[k], per);
        }
        let angle = atan2(f32(y) - cy, f32(x) - cx);
        let baseSlot = u32(floor((angle + 3.14159265) / (6.2831853) * 24.0)) % 24u;
        atomicAdd(&ringSlots[baseSlot], v - per * 24u);
      }
    }
  }
  workgroupBarrier();

  // 阶段二：环槽写入孔沿（线程分工），退回逻辑保证不丢沙
  for (var k: u32 = lid.x + lid.y * 16u; k < 24u; k = k + 256u) {
    let amount = atomicLoad(&ringSlots[k]);
    if (amount == 0u) {
      continue;
    }
    let a = (f32(k) + 0.5) * 6.2831853 / 24.0;
    let sx = clampi(i32(round(cx + cos(a) * (r + 1.0))), 0, i32(sim.width) - 1);
    let sy = clampi(i32(round(cy + sin(a) * (r + 1.0))), 0, i32(sim.height) - 1);
    if (distance(vec2<f32>(f32(sx), f32(sy)), vec2<f32>(cx, cy)) <= r * 0.8) {
      let cxi = clampi(i32(round(cx)), 0, i32(sim.width) - 1);
      let cyi = clampi(i32(round(cy)), 0, i32(sim.height) - 1);
      addHeight(u32(cyi) * sim.width + u32(cxi), amount);
      markTile(cxi, cyi);
    } else {
      addHeight(u32(sy) * sim.width + u32(sx), amount);
      markTile(sx, sy);
    }
  }
}
