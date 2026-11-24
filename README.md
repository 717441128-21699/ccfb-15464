# 沙画手法训练台（WebGPU）

面向沙画培训班的平板应用：在灯箱上用**真实的沙层厚度透光模型**练手法，而不是黑色笔刷涂抹。
灯箱从下往上打光，沙层越厚透光越少；手指推沙会在两侧堆脊，超过沙子休止角自动塌落；
掌侧横扫拉出明暗渐变，指尖点出亮孔，撒沙按落点散开，翻转画面带惯性滑动。

## 运行

```bash
npm install
npm run dev        # 开发服务器，平板/电脑浏览器打开（需支持 WebGPU：Chrome/Edge 124+）
npm test           # 物理模型单元测试（CPU 参考实现，无需 GPU）
npm run build      # 类型检查 + 生产构建
node --experimental-strip-types scripts/check-wgsl.ts   # WGSL 结构自检
```

浏览器需要开启硬件加速。不支持 WebGPU 时页面会给出明确提示。

## 四大模块

| 模块 | 入口 | 能力 |
| --- | --- | --- |
| 手法练习 | 顶部「手法练习」 | 推沙起脊 / 指尖点孔 / 掌侧渐变 / 撒沙四课，GPU 回读沙层厚度自动评分 |
| 自由创作 | 顶部「自由创作」 | 不限手法，自动识别手势，可撤销/翻转/清场 |
| 老师示范 | 顶部「老师示范」 | 录制笔迹时间线、回放、描红引导，录像存 localStorage |
| 灯箱和沙子设置 | 顶部「灯箱和沙子设置」 | 4 组预设 + 休止角/密度/灯色/颗粒等滑杆，3 档模拟分辨率 |

## 手势映射（TypeScript，`src/input/gestures.ts`）

- 静止轻点（位移小、时长短）→ **指尖点孔**
- 接触面积大且移动（掌侧）→ **侧扫**
- 普通慢速单指移动 → **推沙**
- 双指或长按停顿后移动 → **撒沙**
- 工具栏翻转按钮 / 摇平板（加速度计）→ **翻转 + 惯性**
- 也可在底部工具栏手动锁定单一工具

## 物理与渲染模型

### 沙层数据

- 4096×3072 高度场，`u32` 定点数（小数 12 位）存于 storage buffer，双缓冲 A/B
- 速度场用 `rg16float` storage texture 双缓冲；脏 tile 标记用两组 u32 buffer ping-pong
- 定点整数运算让 CPU 参考实现与 WGSL 着色器公式完全一致，单测结论可代表 GPU 行为

### Compute 管线（`src/shaders/`，WGSL）

1. `interaction` / `tap`：手势散射。推沙/侧扫用原子 scatter（本格抹走 → 60% 前缘 + 两侧沙脊），
   点孔用 workgroup 共享的 24 环槽累加器两阶段成环，撒沙用确定性哈希高斯落点
2. `avalanche`：雪崩松弛。Jacobi 迭代，相邻高差超过休止角阈值就产生通量，
   只处理脏 tile（含一圈 halo），稳定 tile 自动淘汰
3. `inertia`：守恒的整数 scatter 平流，速度衰减 + 绝对截止
4. `mirror`：180° 翻转并注入反向惯性
5. `render`：Beer-Lambert 透射 `T=exp(-h·density)` + 暖光/环境漏光 + 颗粒噪声 + 坡度高光 + sRGB
6. `present`：全屏三角形 contain 缩放到画布

### 关键正确性设计

- **严格质量守恒**：推沙/点孔/惯性/翻转的沙量交换在整数域闭合，单测断言总量不变
- **休止角**：稳态相邻高差不超过 `tan(θ)·FIXED_SCALE`（含离散步长容差），四组预设对应 29°/34°/38°
- **无竞争**：所有 scatter 走 `atomicAdd/Sub`，Jacobi 通量只读源缓冲、写独立主缓冲

## 需求对照

| 需求 | 实现与验证 |
| --- | --- | 
| 推沙两侧堆脊且坡度不超过休止角，超过自塌 | `avalanche.wgsl` + CPU 同构；`sand.test.ts` 验证稳态高差、守恒、粗/细砂角度差异 |
| 侧扫渐变至少 8 个明暗层次 | 宽笔刷多帧平流 + 松弛；测试用列平均透射亮度统计分档，断言 ≥ 8 档 |
| 4096×3072 全屏沙层 | `SIM_WIDTH/HEIGHT`，存储缓冲显式请求设备上限；`performance.test.ts` 核算显存与 workgroup 预算 |
| 推沙不低于 60fps | 脏 tile 局部 dispatch + 早退、点孔单 workgroup、双缓冲；性能预算测试约束单 pass 规模 |
| 灯箱/沙子参数自调几组 | 4 组预设（标准暖黄/高亮强对比/粗砂/细砂柔光）+ 全滑杆手动调 |
| 老师示范一笔让学生描 | 录像回放 + SVG 描红引导，归一化坐标跨分辨率可用 |

## 目录结构

```
src/
  sim/        定点常量、数学、4 组预设、CPU 参考沙层模型（与 WGSL 同公式）
  shaders/    8 个 compute pass + 1 个 render pass（WGSL）
  gpu/        WebGPU 引擎：缓冲/管线/双缓冲调度/回读
  input/      Pointer Events 手势识别
  practice/   四课练习评测（GPU 回读 + 量化标准）
  demo/       老师示范录放
  ui/         中文界面（无框架 DOM）
test/         物理与性能测试（vitest）
scripts/      WGSL 结构自检
```

## 已知边界

- 真机 WebGPU 帧率需在目标平板上实测；当前在无显卡 CI 中用 workgroup/显存预算 + CPU 物理测试保证算法可行性。
- 撤销为单步快照（全精度下约 50MB，恢复后即销毁，不长期占显存）。
- 示范录像存浏览器 localStorage（最多 20 条），不跨设备同步。
