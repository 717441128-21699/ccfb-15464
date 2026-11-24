// 手势交互 pass：推沙 / 侧扫 / 点孔 / 撒沙。
// 引擎对每个笔迹段先做一次包围盒 copy（PASS_COPY，把主缓冲复制到源缓冲、
// 清空活动 tile 标记），再做一次 apply（PASS_APPLY，从只读源读、对主缓冲原子 scatter）。
// 多段时引擎交换 A/B 并逐段串联。公式与 CpuSandGrid 各 apply* 方法相同。

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  // 局部 dispatch：global_invocation_id 是工作项在本次 dispatch 内的坐标，加上原点
  let x = i32(gid.x) + i32(sim.originX);
  let y = i32(gid.y) + i32(sim.originY);
  if (x >= i32(sim.width) || y >= i32(sim.height)) {
    return;
  }
  let idx = u32(y) * sim.width + u32(x);

  if (sim.pass == PASS_COPY) {
    storeHeight(idx, readSource(idx));
    let ti = tileIndex(u32(x), u32(y));
    storeActiveTile(ti, 0u);
    return;
  }

  let seg = strokes.segments[sim.segmentIndex];
  switch seg.tool {
    case TOOL_PUSH: {
      applyPush(seg, false, x, y, idx);
    }
    case TOOL_SWEEP: {
      applyPush(seg, true, x, y, idx);
    }
    case TOOL_SPRINKLE: {
      applySprinkle(seg, x, y, idx);
    }
    // 点孔由独立的 tap.wgsl 处理（需要 workgroup 共享环槽）
    default: {}
  }
}

fn applyPush(seg: Stroke, wide: bool, x: i32, y: i32, idx: u32) {
  let r = seg.radius;
  let d = pointSegmentDist(f32(x), f32(y), seg.x0, seg.y0, seg.x1, seg.y1);
  if (d > r) {
    return;
  }
  // 只读源（copy 已建好），主缓冲做原子写
  let src = readSource(idx);
  if (src == 0u) {
    return;
  }

  let inner = select(r * 0.45, r * 0.3, wide);
  let falloff = 1.0 - smoothstepf(inner, r, d);
  let shape = select(0.35 + 0.65 * falloff, 0.25 + 0.75 * falloff, wide);
  let take = u32(round(f32(src) * clampf(seg.pressure, 0.0, 1.0) * shape));
  if (take == 0u) {
    return;
  }

  // 笔迹方向与法向
  let dx = seg.x1 - seg.x0;
  let dy = seg.y1 - seg.y0;
  let segLen = max(sqrt(dx * dx + dy * dy), 0.0001);
  let ux = dx / segLen;
  let uy = dy / segLen;
  let nx = -uy;
  let ny = ux;

  // 本格抹走
  subHeight(idx, take);
  markTile(x, y);

  // 60% 前缘堆积，整除余数并入前缘，保证 take 完整分配（严格守恒）
  var front = u32(round(f32(take) * 0.6));
  let sideAmt = (take - front) / 2u;
  front = take - sideAmt * 2u;
  let fx = i32(round(f32(x) + ux * seg.pushDist));
  let fy = i32(round(f32(y) + uy * seg.pushDist));
  if (inBounds(fx, fy)) {
    addHeight(u32(fy) * sim.width + u32(fx), front);
    markTile(fx, fy);
  } else {
    addHeight(idx, front); // 越界退回本格
  }

  // 剩余量 20% / 20% 两侧沙脊
  let side1x = i32(round(f32(x) + nx * (r * 0.92)));
  let side1y = i32(round(f32(y) + ny * (r * 0.92)));
  if (inBounds(side1x, side1y)) {
    addHeight(u32(side1y) * sim.width + u32(side1x), sideAmt);
    markTile(side1x, side1y);
  } else {
    addHeight(idx, sideAmt);
  }
  let side2x = i32(round(f32(x) - nx * (r * 0.92)));
  let side2y = i32(round(f32(y) - ny * (r * 0.92)));
  if (inBounds(side2x, side2y)) {
    addHeight(u32(side2y) * sim.width + u32(side2x), sideAmt);
    markTile(side2x, side2y);
  } else {
    addHeight(idx, sideAmt);
  }

  // 速度写入速度纹理，交给惯性 pass
  let speed = seg.pushDist * (0.5 + 0.5 * seg.pressure);
  textureStore(velTex, vec2<i32>(x, y), vec4<f32>(ux * speed, uy * speed, 0.0, 0.0));
}

fn applySprinkle(seg: Stroke, x: i32, y: i32, idx: u32) {
  let dx = seg.x1 - seg.x0;
  let dy = seg.y1 - seg.y0;
  let len = sqrt(dx * dx + dy * dy);
  let steps = max(1u, u32(ceil(len / max(1.0, seg.radius * 0.5))));
  let baseAdd = u32(round(FIXED_SCALE * 0.9 * seg.pressure));
  let px = f32(x);
  let py = f32(y);

  for (var s: u32 = 0u; s <= steps; s = s + 1u) {
    let t = f32(s) / f32(steps);
    let jx = (hash21(i32(s), i32(strokes.seed & 0xffffu), 7) - 0.5) * seg.radius * 1.6;
    let jy = (hash21(i32(s), i32(strokes.seed & 0xffffu), 19) - 0.5) * seg.radius * 1.6;
    let gx = seg.x0 + dx * t + jx;
    let gy = seg.y0 + dy * t + jy;
    let spread = seg.radius * (0.7 + 0.6 * hash21(i32(s), i32(strokes.seed & 0xffffu), 31));
    let d = distance(vec2<f32>(px, py), vec2<f32>(gx, gy));
    if (d > spread * 1.6) {
      continue;
    }
    let w = exp(-(d * d) / (2.0 * (spread * 0.5) * (spread * 0.5)));
    let add = u32(round(f32(baseAdd) * (0.6 + 0.8 * hash21(i32(s), i32(strokes.seed & 0xffffu), 43)) * w));
    if (add > 0u) {
      addHeight(idx, add);
      markTile(x, y);
    }
  }
}
