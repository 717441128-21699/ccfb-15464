/**
 * GpuSimUniform 的 JS 侧镜像，字段顺序/对齐必须与 common.wgsl 的 GpuSim 一致。
 * 总大小 96 字节（vec3 按 16 字节对齐）。
 */
export interface GpuSimUniformShape {
  width: number;
  height: number;
  tilesX: number;
  tilesY: number;
  reposeDelta: number;
  relaxIters: number;
  density: number;
  grain: number;
  lightColor: [number, number, number];
  lightBrightness: number;
  ambient: number;
  bodyAmount: number;
  sandTint: [number, number, number];
  pass: number;
  segmentIndex: number;
  frame: number;
  useB: number;
  flipImpulse: number;
  velocityDamping: number;
  originX: number;
  originY: number;
}

export const SIM_UNIFORM_SIZE = 112; // vec3 16 对齐，结构体向上取整到 16 的倍数
export const STROKE_UNIFORM_SIZE = 288; // 16 头 + 8 * 32
export const CLEAR_UNIFORM_SIZE = 16;
export const PRESENT_UNIFORM_SIZE = 32;

export function writeSimUniform(view: DataView, u: GpuSimUniformShape): void {
  view.setUint32(0, u.width, true);
  view.setUint32(4, u.height, true);
  view.setUint32(8, u.tilesX, true);
  view.setUint32(12, u.tilesY, true);
  view.setUint32(16, u.reposeDelta, true);
  view.setUint32(20, u.relaxIters, true);
  view.setFloat32(24, u.density, true);
  view.setFloat32(28, u.grain, true);
  // offset 32: vec3 lightColor + f32 brightness
  view.setFloat32(32, u.lightColor[0], true);
  view.setFloat32(36, u.lightColor[1], true);
  view.setFloat32(40, u.lightColor[2], true);
  view.setFloat32(44, u.lightBrightness, true);
  // offset 48: f32 ambient, f32 bodyAmount, vec3 sandTint
  view.setFloat32(48, u.ambient, true);
  view.setFloat32(52, u.bodyAmount, true);
  view.setFloat32(56, u.sandTint[0], true);
  view.setFloat32(60, u.sandTint[1], true);
  view.setFloat32(64, u.sandTint[2], true);
  // offset 68..: 标量尾部
  view.setUint32(68, u.pass, true);
  view.setUint32(72, u.segmentIndex, true);
  view.setUint32(76, u.frame, true);
  view.setUint32(80, u.useB, true);
  view.setFloat32(84, u.flipImpulse, true);
  view.setFloat32(88, u.velocityDamping, true);
  view.setUint32(92, u.originX, true);
  view.setUint32(96, u.originY, true);
}
