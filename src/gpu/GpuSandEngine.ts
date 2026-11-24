/**
 * WebGPU 沙层引擎。
 *
 * 资源模型（全部 4096x3072 原生分辨率）：
 *   heightA/heightB  u32 storage buffer 双缓冲（定点高度，原子访问）
 *   velTex          rg16float storage texture（惯性速度，inertia pass 原地读改写）
 *   tileA/tileB     u32 storage buffer 双缓冲（脏 tile 标记，雪崩迭代间 ping-pong）
 *   outputTex        rgba8unorm storage texture（灯箱渲染结果）
 *
 * 每个 compute pass 前由引擎决定 useB（哪个是主缓冲），并把与主缓冲相反的缓冲
 * 绑定到 heightReadOnly；雪崩的活动/下一 tile 也随 useB 切换。
 */
import {
  FIXED_SCALE,
  TILE_SIZE,
  Tool,
  reposeDeltaFixed,
  type SimParams,
  type StrokeSegment,
} from '../sim/constants';
import { PRESENT_UNIFORM_SIZE, SIM_UNIFORM_SIZE, STROKE_UNIFORM_SIZE, writeSimUniform } from './uniforms';
import commonWgsl from '../shaders/common.wgsl?raw';
import interactionWgsl from '../shaders/interaction.wgsl?raw';
import tapWgsl from '../shaders/tap.wgsl?raw';
import copyGridWgsl from '../shaders/copygrid.wgsl?raw';
import avalancheWgsl from '../shaders/avalanche.wgsl?raw';
import inertiaWgsl from '../shaders/inertia.wgsl?raw';
import mirrorWgsl from '../shaders/mirror.wgsl?raw';
import clearWgsl from '../shaders/clear.wgsl?raw';
import renderWgsl from '../shaders/render.wgsl?raw';
import presentWgsl from '../shaders/present.wgsl?raw';

export interface EngineCounters {
  fps: number;
  frameMs: number;
  activeTiles: number;
}

export class GpuSandEngine {
  readonly device: GPUDevice;
  readonly context: GPUCanvasContext;
  readonly width: number;
  readonly height: number;
  readonly tilesX: number;
  readonly tilesY: number;

  params: SimParams;
  private frameNo = 0;
  private useB = false;

  private heightA!: GPUBuffer;
  private heightB!: GPUBuffer;
  private velTex!: GPUTexture;
  private tileA!: GPUBuffer;
  private tileB!: GPUBuffer;
  private outputTex!: GPUTexture;
  private simUniformBuf!: GPUBuffer;
  private strokeUniformBuf!: GPUBuffer;
  private clearUniformBuf!: GPUBuffer;
  private presentUniformBuf!: GPUBuffer;
  private stagingBuf!: GPUBuffer;
  private undoSnapshot: GPUBuffer | null = null;

  private simBindLayout!: GPUBindGroupLayout;
  private simBindGroup!: GPUBindGroup;
  private clearBindGroup!: GPUBindGroup;
  private renderBindGroup!: GPUBindGroup;
  private presentBindGroup!: GPUBindGroup;

  private interactionPipeline!: GPUComputePipeline;
  private tapPipeline!: GPUComputePipeline;
  private copyGridPipeline!: GPUComputePipeline;
  private avalanchePipeline!: GPUComputePipeline;
  private inertiaPipeline!: GPUComputePipeline;
  private mirrorPipeline!: GPUComputePipeline;
  private clearPipeline!: GPUComputePipeline;
  private renderPipeline!: GPUComputePipeline;
  private presentPipeline!: GPURenderPipeline;
  private sampler!: GPUSampler;

  private lastInteractionAt = -1000;
  private counters: EngineCounters = { fps: 0, frameMs: 0, activeTiles: 0 };
  private fpsAccumMs = 0;
  private fpsFrames = 0;

  private constructor(
    device: GPUDevice,
    context: GPUCanvasContext,
    width: number,
    height: number,
    params: SimParams,
  ) {
    this.device = device;
    this.context = context;
    this.width = width;
    this.height = height;
    this.tilesX = Math.ceil(width / TILE_SIZE);
    this.tilesY = Math.ceil(height / TILE_SIZE);
    this.params = params;
  }

  static async create(
    canvas: HTMLCanvasElement,
    width: number,
    height: number,
    params: SimParams,
  ): Promise<GpuSandEngine> {
    if (!navigator.gpu) {
      throw new Error('NO_WEBGPU');
    }
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) {
      throw new Error('NO_ADAPTER');
    }
    const cellCount = width * height;
    const heightBytes = cellCount * 4;
    // 全精度 4096x3072 高度缓冲约 50MB，超过部分驱动的默认上限，显式请求
    const need = Math.max(heightBytes, 256 * 256 * 4);
    const limits: Record<string, number> = {};
    const pick = (key: keyof GPUSupportedLimits): void => {
      const supported = adapter.limits[key];
      if (typeof supported === 'number' && supported >= need) limits[key] = supported;
    };
    pick('maxStorageBufferBindingSize');
    pick('maxBufferSize');
    // tap.wgsl 的 workgroup 共享环槽只需 24*4=96 字节，默认上限足够，这里仍显式确认
    if (adapter.limits.maxComputeWorkgroupStorageSize < 96) {
      throw new Error('WORKGROUP_STORAGE_TOO_SMALL');
    }
    const device = await adapter.requestDevice({ requiredLimits: limits });
    const context = canvas.getContext('webgpu');
    if (!context) {
      throw new Error('NO_CONTEXT');
    }
    const engine = new GpuSandEngine(device, context, width, height, params);
    engine.initResources();
    engine.initPipelines();
    engine.configureContext(canvas);
    return engine;
  }

  private initResources(): void {
    const cellCount = this.width * this.height;
    const tileCount = this.tilesX * this.tilesY;
    const d = this.device;

    this.heightA = d.createBuffer({ size: cellCount * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    this.heightB = d.createBuffer({ size: cellCount * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    this.tileA = d.createBuffer({ size: tileCount * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.tileB = d.createBuffer({ size: tileCount * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });

    this.velTex = d.createTexture({
      size: { width: this.width, height: this.height },
      format: 'rg16float',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST,
    });

    this.outputTex = d.createTexture({
      size: { width: this.width, height: this.height },
      format: 'rgba8unorm',
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
    });

    this.simUniformBuf = d.createBuffer({ size: SIM_UNIFORM_SIZE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.strokeUniformBuf = d.createBuffer({ size: STROKE_UNIFORM_SIZE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.clearUniformBuf = d.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.presentUniformBuf = d.createBuffer({ size: PRESENT_UNIFORM_SIZE, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    // 回读缓冲（练习评测用），按 256x256 区域上限预分配
    this.stagingBuf = d.createBuffer({ size: 256 * 256 * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });

    this.sampler = d.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
    });
  }

  private initPipelines(): void {
    const d = this.device;
    this.simBindLayout = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'read-write', format: 'rg16float' } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        // 仅 binding 7 是真正的只读源（Jacobi 快照）
        { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      ],
    });
    this.rebuildSimBindGroup();

    const clearLayout = d.createBindGroupLayout({
      entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }],
    });
    this.clearBindGroup = d.createBindGroup({
      layout: clearLayout,
      entries: [{ binding: 0, resource: { buffer: this.clearUniformBuf } }],
    });

    const renderLayout = d.createBindGroupLayout({
      entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba8unorm' } }],
    });
    this.renderBindGroup = d.createBindGroup({
      layout: renderLayout,
      entries: [{ binding: 0, resource: this.outputTex.createView() }],
    });

    const presentLayout = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: {} },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
      ],
    });
    this.presentBindGroup = d.createBindGroup({
      layout: presentLayout,
      entries: [
        { binding: 0, resource: { buffer: this.presentUniformBuf } },
        { binding: 1, resource: this.outputTex.createView() },
        { binding: 2, resource: this.sampler },
      ],
    });

    const simPipelineLayout = d.createPipelineLayout({
      bindGroupLayouts: [this.simBindLayout],
    });
    const clearPipelineLayout = d.createPipelineLayout({
      bindGroupLayouts: [this.simBindLayout, clearLayout],
    });
    const renderPipelineLayout = d.createPipelineLayout({
      bindGroupLayouts: [this.simBindLayout, renderLayout],
    });

    this.interactionPipeline = d.createComputePipeline({
      layout: simPipelineLayout,
      compute: { module: d.createShaderModule({ code: commonWgsl + interactionWgsl }), entryPoint: 'main' },
    });
    this.copyGridPipeline = d.createComputePipeline({
      layout: simPipelineLayout,
      compute: { module: d.createShaderModule({ code: commonWgsl + copyGridWgsl }), entryPoint: 'main' },
    });
    this.tapPipeline = d.createComputePipeline({
      layout: simPipelineLayout,
      compute: { module: d.createShaderModule({ code: commonWgsl + tapWgsl }), entryPoint: 'main' },
    });
    this.avalanchePipeline = d.createComputePipeline({
      layout: simPipelineLayout,
      compute: { module: d.createShaderModule({ code: commonWgsl + avalancheWgsl }), entryPoint: 'main' },
    });
    this.inertiaPipeline = d.createComputePipeline({
      layout: simPipelineLayout,
      compute: { module: d.createShaderModule({ code: commonWgsl + inertiaWgsl }), entryPoint: 'main' },
    });
    this.mirrorPipeline = d.createComputePipeline({
      layout: simPipelineLayout,
      compute: { module: d.createShaderModule({ code: commonWgsl + mirrorWgsl }), entryPoint: 'main' },
    });
    this.clearPipeline = d.createComputePipeline({
      layout: clearPipelineLayout,
      compute: { module: d.createShaderModule({ code: commonWgsl + clearWgsl }), entryPoint: 'main' },
    });
    this.renderPipeline = d.createComputePipeline({
      layout: renderPipelineLayout,
      compute: { module: d.createShaderModule({ code: commonWgsl + renderWgsl }), entryPoint: 'main' },
    });
    this.presentPipeline = d.createRenderPipeline({
      layout: d.createPipelineLayout({ bindGroupLayouts: [presentLayout] }),
      vertex: { module: d.createShaderModule({ code: presentWgsl }), entryPoint: 'vs' },
      fragment: {
        module: d.createShaderModule({ code: presentWgsl }),
        entryPoint: 'fs',
        targets: [{ format: navigator.gpu.getPreferredCanvasFormat() }],
      },
      primitive: { topology: 'triangle-list' },
    });
  }

  /** 根据当前 useB 重建模拟 bind group（切换只读源缓冲；tile 活动/下一组由着色器按 useB 选择） */
  private rebuildSimBindGroup(): void {
    const source = this.useB ? this.heightA : this.heightB;
    this.simBindGroup = this.device.createBindGroup({
      layout: this.simBindLayout,
      entries: [
        { binding: 0, resource: { buffer: this.simUniformBuf } },
        { binding: 1, resource: { buffer: this.strokeUniformBuf } },
        { binding: 2, resource: { buffer: this.heightA } },
        { binding: 3, resource: { buffer: this.heightB } },
        { binding: 4, resource: this.velTex.createView() },
        { binding: 5, resource: { buffer: this.tileA } },
        { binding: 6, resource: { buffer: this.tileB } },
        { binding: 7, resource: { buffer: source } },
      ],
    });
  }

  private configureContext(canvas: HTMLCanvasElement): void {
    this.context.configure({
      device: this.device,
      format: navigator.gpu.getPreferredCanvasFormat(),
      alphaMode: 'opaque',
    });
    void canvas;
  }

  private writeSimUniform(extra: {
    pass?: number;
    segmentIndex?: number;
    useB?: boolean;
    flipImpulse?: number;
    origin?: { x: number; y: number };
  }): void {
    const buf = new ArrayBuffer(SIM_UNIFORM_SIZE);
    const view = new DataView(buf);
    writeSimUniform(view, {
      width: this.width,
      height: this.height,
      tilesX: this.tilesX,
      tilesY: this.tilesY,
      reposeDelta: reposeDeltaFixed(this.params.reposeDeg),
      relaxIters: this.params.relaxIters,
      density: this.params.density,
      grain: this.params.grain,
      lightColor: this.params.lightColor,
      lightBrightness: this.params.lightBrightness,
      ambient: this.params.ambient,
      bodyAmount: this.params.bodyAmount,
      sandTint: this.params.sandTint,
      pass: extra.pass ?? 0,
      segmentIndex: extra.segmentIndex ?? 0,
      frame: this.frameNo,
      useB: (extra.useB ?? this.useB) ? 1 : 0,
      flipImpulse: extra.flipImpulse ?? 0,
      velocityDamping: 0.86,
      originX: extra.origin?.x ?? 0,
      originY: extra.origin?.y ?? 0,
    });
    this.device.queue.writeBuffer(this.simUniformBuf, 0, buf);
  }

  private writeStrokes(segments: StrokeSegment[], seed: number): void {
    const buf = new ArrayBuffer(STROKE_UNIFORM_SIZE);
    const view = new DataView(buf);
    view.setUint32(0, Math.min(segments.length, 8), true);
    view.setUint32(4, seed >>> 0, true);
    segments.slice(0, 8).forEach((seg, i) => {
      const o = 16 + i * 32;
      view.setFloat32(o, seg.x0, true);
      view.setFloat32(o + 4, seg.y0, true);
      view.setFloat32(o + 8, seg.x1, true);
      view.setFloat32(o + 12, seg.y1, true);
      view.setFloat32(o + 16, seg.radius, true);
      view.setFloat32(o + 20, seg.pressure, true);
      view.setUint32(o + 24, seg.tool, true);
      view.setFloat32(o + 28, seg.pushDist, true);
    });
    this.device.queue.writeBuffer(this.strokeUniformBuf, 0, buf);
  }

  private groupsX(): number {
    return Math.ceil(this.width / TILE_SIZE);
  }

  private groupsY(): number {
    return Math.ceil(this.height / TILE_SIZE);
  }

  /** 段包围盒（含 push 沉积半径），按 workgroup 对齐 */
  private segmentBBox(seg: StrokeSegment): { x: number; y: number; w: number; h: number } {
    const reach = seg.radius + seg.pushDist + 4;
    const minX = Math.max(0, Math.floor(Math.min(seg.x0, seg.x1) - reach));
    const maxX = Math.min(this.width, Math.ceil(Math.max(seg.x0, seg.x1) + reach));
    const minY = Math.max(0, Math.floor(Math.min(seg.y0, seg.y1) - reach));
    const maxY = Math.min(this.height, Math.ceil(Math.max(seg.y0, seg.y1) + reach));
    const x = Math.floor(minX / TILE_SIZE) * TILE_SIZE;
    const y = Math.floor(minY / TILE_SIZE) * TILE_SIZE;
    const x2 = Math.ceil(maxX / TILE_SIZE) * TILE_SIZE;
    const y2 = Math.ceil(maxY / TILE_SIZE) * TILE_SIZE;
    return { x, y, w: Math.max(TILE_SIZE, x2 - x), h: Math.max(TILE_SIZE, y2 - y) };
  }

  private dispatchBox(
    pass: GPUComputePassEncoder,
    pipeline: GPUComputePipeline,
    box: { x: number; y: number; w: number; h: number },
  ): void {
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, this.simBindGroup);
    pass.dispatchWorkgroups(Math.ceil(box.w / TILE_SIZE), Math.ceil(box.h / TILE_SIZE), 1);
  }

  /** 点孔包围盒：以孔中心所在的 2x2 tile 为单 workgroup（16x16），覆盖孔与孔沿 */
  private tapBBox(seg: StrokeSegment): { x: number; y: number; w: number; h: number } {
    const TAP_BOX = TILE_SIZE * 2; // 16
    const x = Math.floor(seg.x1 / TAP_BOX) * TAP_BOX;
    const y = Math.floor(seg.y1 / TAP_BOX) * TAP_BOX;
    return {
      x: Math.max(0, Math.min(x, this.width - TAP_BOX)),
      y: Math.max(0, Math.min(y, this.height - TAP_BOX)),
      w: TAP_BOX,
      h: TAP_BOX,
    };
  }

  /**
   * 应用一帧内的笔迹段（最多 8 段）。
   * 每段：interaction 的 PASS_COPY 在包围盒内把主缓冲复制到源缓冲并清活动 tile，
   * 随后交换主源（useB 翻转），PASS_APPLY 从只读源读、对新主缓冲原子 scatter。
   * 段与段逐段串联，与 CPU 参考模型的 applyStroke 序列等价。
   */
  applyStrokes(segments: StrokeSegment[], seed: number): void {
    if (segments.length === 0) return;
    this.lastInteractionAt = this.frameNo;
    this.writeStrokes(segments, seed);
    const encoder = this.device.createCommandEncoder();

    segments.slice(0, 8).forEach((seg, i) => {
      const box = seg.tool === Tool.Tap ? this.tapBBox(seg) : this.segmentBBox(seg);
      // 1) copy：当前主 -> 当前源，清空范围内活动 tile
      this.writeSimUniform({ pass: 0, segmentIndex: i, origin: box });
      const copyPass = encoder.beginComputePass();
      this.dispatchBox(copyPass, this.interactionPipeline, box);
      copyPass.end();
      // 2) 交换主源，apply 写新主缓冲（readSource 仍指向旧主，即只读源）
      this.useB = !this.useB;
      this.rebuildSimBindGroup();
      this.writeSimUniform({ pass: 1, segmentIndex: i, origin: box });
      const applyPass = encoder.beginComputePass();
      if (seg.tool === Tool.Tap) {
        // 点孔：单 workgroup + workgroup 共享环槽，包围盒已按 8 对齐且不超过 8x8
        applyPass.setPipeline(this.tapPipeline);
        applyPass.setBindGroup(0, this.simBindGroup);
        applyPass.dispatchWorkgroups(1, 1, 1);
      } else {
        this.dispatchBox(applyPass, this.interactionPipeline, box);
      }
      applyPass.end();
    });

    this.device.queue.submit([encoder.finish()]);
  }

  /**
   * 雪崩松弛 relaxIters 轮。
   * 每轮：copygrid 让主=源（同值快照）-> 交换 -> avalanche 从源读、写新主，
   * 活动/下一 tile 随 useB 自动 ping-pong（下一轮开始前清空“下一”缓冲）。
   */
  runAvalanche(): void {
    if (this.frameNo - this.lastInteractionAt > 60) return;
    const encoder = this.device.createCommandEncoder();
    for (let iter = 0; iter < this.params.relaxIters; iter++) {
      // 本轮开始前，清空将作为“下一 tile”的缓冲（与当前活动 tile 相反的那组）
      const nextTiles = this.useB ? this.tileA : this.tileB;
      encoder.clearBuffer(nextTiles);

      // 主 <- 源，建立同值快照
      this.writeSimUniform({ pass: 1 });
      let pass = encoder.beginComputePass();
      pass.setPipeline(this.copyGridPipeline);
      pass.setBindGroup(0, this.simBindGroup);
      pass.dispatchWorkgroups(this.groupsX(), this.groupsY());
      pass.end();

      // 交换：旧主变只读源，新主承接 Jacobi 结果
      this.useB = !this.useB;
      this.rebuildSimBindGroup();
      this.writeSimUniform({ pass: 1 });
      pass = encoder.beginComputePass();
      pass.setPipeline(this.avalanchePipeline);
      pass.setBindGroup(0, this.simBindGroup);
      pass.dispatchWorkgroups(this.groupsX(), this.groupsY());
      pass.end();
    }
    this.device.queue.submit([encoder.finish()]);
  }

  /** 惯性平流（翻转后多帧调用，由外层控制）：copygrid 建源 -> 交换 -> 半拉格朗日平流 */
  runInertia(): void {
    const encoder = this.device.createCommandEncoder();
    this.writeSimUniform({ pass: 1 });
    let pass = encoder.beginComputePass();
    pass.setPipeline(this.copyGridPipeline);
    pass.setBindGroup(0, this.simBindGroup);
    pass.dispatchWorkgroups(this.groupsX(), this.groupsY());
    pass.end();

    this.useB = !this.useB;
    this.rebuildSimBindGroup();
    this.writeSimUniform({ pass: 1 });
    pass = encoder.beginComputePass();
    pass.setPipeline(this.inertiaPipeline);
    pass.setBindGroup(0, this.simBindGroup);
    pass.dispatchWorkgroups(this.groupsX(), this.groupsY());
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  /** 快速翻转画面：copygrid 建源 -> 交换 -> 180° 旋转 + 注入反向惯性 */
  flip(impulse = 6): void {
    const encoder = this.device.createCommandEncoder();
    this.writeSimUniform({ pass: 1, flipImpulse: impulse });
    let pass = encoder.beginComputePass();
    pass.setPipeline(this.copyGridPipeline);
    pass.setBindGroup(0, this.simBindGroup);
    pass.dispatchWorkgroups(this.groupsX(), this.groupsY());
    pass.end();

    this.useB = !this.useB;
    this.rebuildSimBindGroup();
    this.writeSimUniform({ pass: 1, flipImpulse: impulse });
    pass = encoder.beginComputePass();
    pass.setPipeline(this.mirrorPipeline);
    pass.setBindGroup(0, this.simBindGroup);
    pass.dispatchWorkgroups(this.groupsX(), this.groupsY());
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    this.lastInteractionAt = this.frameNo;
  }

  /**
   * 在一次交互前保存当前主高度缓冲快照（单步撤销）。
   * 全精度快照约 50MB，因此只保留最近一次；新快照覆盖并销毁旧的。
   */
  pushUndoSnapshot(): void {
    const main = this.useB ? this.heightB : this.heightA;
    if (this.undoSnapshot) this.undoSnapshot.destroy();
    const snap = this.device.createBuffer({
      size: main.size,
      usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    });
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(main, 0, snap, 0, main.size);
    this.device.queue.submit([encoder.finish()]);
    this.undoSnapshot = snap;
  }

  get canUndo(): boolean {
    return this.undoSnapshot !== null;
  }

  /** 撤销：把快照恢复到两个高度缓冲，随后销毁快照 */
  undo(): boolean {
    const snap = this.undoSnapshot;
    if (!snap) return false;
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(snap, 0, this.heightA, 0, this.heightA.size);
    encoder.copyBufferToBuffer(snap, 0, this.heightB, 0, this.heightB.size);
    encoder.clearBuffer(this.tileA);
    encoder.clearBuffer(this.tileB);
    this.device.queue.submit([encoder.finish()]);
    snap.destroy();
    this.undoSnapshot = null;
    return true;
  }

  /** 清场：mode 1 同时清速度与 tile */
  clearAll(): void {
    const buf = new ArrayBuffer(16);
    new DataView(buf).setUint32(0, 1, true);
    this.device.queue.writeBuffer(this.clearUniformBuf, 0, buf);
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.clearPipeline);
    pass.setBindGroup(0, this.simBindGroup);
    pass.setBindGroup(1, this.clearBindGroup);
    pass.dispatchWorkgroups(this.groupsX(), this.groupsY());
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  /** 灯箱渲染 + 上屏。viewport 为画布 CSS 像素视口（contain） */
  render(canvas: HTMLCanvasElement): void {
    const t0 = performance.now();
    this.writeSimUniform({ pass: 1 });
    const encoder = this.device.createCommandEncoder();

    // compute 渲染高度场 -> outputTex
    const cpass = encoder.beginComputePass();
    cpass.setPipeline(this.renderPipeline);
    cpass.setBindGroup(0, this.simBindGroup);
    cpass.setBindGroup(1, this.renderBindGroup);
    cpass.dispatchWorkgroups(this.groupsX(), this.groupsY());
    cpass.end();

    // 上屏
    const vp = this.computeViewport(canvas);
    const pbuf = new ArrayBuffer(PRESENT_UNIFORM_SIZE);
    const pv = new DataView(pbuf);
    pv.setFloat32(0, vp.x, true);
    pv.setFloat32(4, vp.y, true);
    pv.setFloat32(8, vp.w, true);
    pv.setFloat32(12, vp.h, true);
    pv.setFloat32(16, canvas.width, true);
    pv.setFloat32(20, canvas.height, true);
    this.device.queue.writeBuffer(this.presentUniformBuf, 0, pbuf);

    const rpass = encoder.beginRenderPass({
      colorAttachments: [
        { view: this.context.getCurrentTexture().createView(), clearValue: { r: 0.012, g: 0.01, b: 0.008, a: 1 }, loadOp: 'clear', storeOp: 'store' },
      ],
    });
    rpass.setPipeline(this.presentPipeline);
    rpass.setBindGroup(0, this.presentBindGroup);
    rpass.draw(3);
    rpass.end();
    this.device.queue.submit([encoder.finish()]);

    const dt = performance.now() - t0;
    this.fpsAccumMs += dt;
    this.fpsFrames++;
    if (this.fpsAccumMs >= 500) {
      this.counters.frameMs = this.fpsAccumMs / this.fpsFrames;
      this.counters.fps = Math.round(1000 / Math.max(0.01, this.counters.frameMs));
      this.fpsAccumMs = 0;
      this.fpsFrames = 0;
    }
    this.frameNo++;
  }

  private computeViewport(canvas: HTMLCanvasElement): { x: number; y: number; w: number; h: number } {
    const simAspect = this.width / this.height;
    const cw = canvas.width;
    const ch = canvas.height;
    const canvasAspect = cw / ch;
    let w: number;
    let h: number;
    if (canvasAspect > simAspect) {
      h = ch;
      w = h * simAspect;
    } else {
      w = cw;
      h = w / simAspect;
    }
    return { x: (cw - w) / 2, y: (ch - h) / 2, w, h };
  }

  getCounters(): EngineCounters {
    return { ...this.counters };
  }

  /**
   * 回读一块高度场区域（练习评测用）。返回定点 u32。
   * 区域自动裁到 256x256 以内与画面边界内。
   */
  async readbackRegion(x: number, y: number, w: number, h: number): Promise<Uint32Array> {
    const rw = Math.min(256, w, this.width - x);
    const rh = Math.min(256, h, this.height - y);
    const main = this.useB ? this.heightB : this.heightA;
    const encoder = this.device.createCommandEncoder();
    const bytesPerRow = Math.ceil((rw * 4) / 256) * 256;
    encoder.copyBufferToBuffer(main, (y * this.width + x) * 4, this.stagingBuf, 0, bytesPerRow * rh);
    this.device.queue.submit([encoder.finish()]);
    await this.stagingBuf.mapAsync(GPUMapMode.READ);
    const mapped = new Uint8Array(this.stagingBuf.getMappedRange());
    const out = new Uint32Array(rw * rh);
    for (let row = 0; row < rh; row++) {
      out.set(new Uint32Array(mapped.buffer, row * bytesPerRow, rw), row * rw);
    }
    this.stagingBuf.unmap();
    return out;
  }

  get fixedScale(): number {
    return FIXED_SCALE;
  }
}

export { Tool };
