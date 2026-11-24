/**
 * 沙画模拟的全局常量与共享类型。
 * 高度场使用 u32 定点数：真实厚度 h（单位：沙层厚度单元）= H / FIXED_SCALE。
 * 所有涉及沙量交换的计算都在整数域完成，CPU 参考模型与 WGSL 着色器使用完全相同的公式，
 * 以保证单测结论可以代表 GPU 行为。
 */

/** 平板全屏灯箱标准沙层分辨率（4:3） */
export const SIM_WIDTH = 4096;
export const SIM_HEIGHT = 3072;

/** 定点小数位数与缩放系数 */
export const FIXED_SHIFT = 12;
export const FIXED_SCALE = 1 << FIXED_SHIFT; // 4096

/** 脏标记 tile 边长（也是 compute workgroup 尺寸） */
export const TILE_SIZE = 8;

/** 单帧最多处理的笔迹段数（快速移动时一帧可能聚合出多段） */
export const MAX_SEGMENTS = 8;

/** 工具编号，与 WGSL 中的分支判断保持一致 */
export const Tool = {
  /** 手指推沙：前方堆积 + 两侧沙脊，随后由雪崩松弛到休止角 */
  Push: 0,
  /** 指尖点孔：清出一个亮孔，沙堆到孔沿 */
  Tap: 1,
  /** 手掌侧扫：宽笔刷平推，产生大片明暗渐变 */
  Sweep: 2,
  /** 撒沙：从高处落下，按落点高斯散开 */
  Sprinkle: 3,
} as const;

export type ToolId = (typeof Tool)[keyof typeof Tool];

/** 一段笔迹（坐标均为沙层纹理像素坐标） */
export interface StrokeSegment {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  /** 笔刷半径（纹理像素） */
  radius: number;
  /** 压感 0..1（无压感设备由接触面积/速度估算） */
  pressure: number;
  tool: ToolId;
  /** 推沙/侧扫时沙被向前推送的距离（纹理像素） */
  pushDist: number;
  /** 单次经过时被带走的沙量比例 0..1 */
  pickup: number;
  /** 撒沙等随机过程的种子 */
  seed: number;
}

/** 灯箱与沙子的可调参数（四块设置面板中的“灯箱和沙子设置”） */
export interface SimParams {
  /** 沙子休止角（度），超过该坡度自动塌落 */
  reposeDeg: number;
  /** 每帧雪崩松弛迭代次数 */
  relaxIters: number;
  /** 光学厚度系数：h * density 进入 Beer-Lambert 指数 */
  density: number;
  /** 灯箱暖光颜色（线性 sRGB 0..1） */
  lightColor: [number, number, number];
  /** 灯箱亮度 0..1.5 */
  lightBrightness: number;
  /** 无沙处环境暗度 0..1（灯箱漏光/反光） */
  ambient: number;
  /** 颗粒感强度 0..1 */
  grain: number;
  /** 沙体本色（厚沙处微微透出的暖棕） */
  sandTint: [number, number, number];
  /** 厚沙处沙体本色混合比例 */
  bodyAmount: number;
}

/** 由休止角换算定点高度差阈值：相邻格高差超过它就必须塌落 */
export function reposeDeltaFixed(reposeDeg: number): number {
  return Math.round(FIXED_SCALE * Math.tan((reposeDeg * Math.PI) / 180));
}
