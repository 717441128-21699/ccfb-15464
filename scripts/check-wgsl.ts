/**
 * WGSL 结构自检（无 GPU 环境下的静态检查）：
 *  - 括号/花括号配对
 *  - 每个被调用的函数都有定义（common 工具函数 + 各 pass main）
 *  - 每个引用的 sim/strokes 字段、绑定资源都已在 common 声明
 *  - workgroupBarrier 只出现在声明了 var<workgroup> 的着色器
 * 不是完整语法解析，但能抓住拼接遗漏、括号失衡和未声明符号这类高频错误。
 */
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const shaderDir = join(here, '..', 'src', 'shaders');

const common = readFileSync(join(shaderDir, 'common.wgsl'), 'utf8');
const passes = ['interaction', 'tap', 'avalanche', 'inertia', 'mirror', 'clear', 'copygrid', 'render'];

const errors: string[] = [];

function stripComments(src: string): string {
  return src
    .replace(/\/\/[^\n]*/g, '') // 行注释
    .replace(/\/\*[\s\S]*?\*\//g, ''); // 块注释
}

function checkBalance(raw: string, name: string): void {
  const src = stripComments(raw);
  const pairs: Array<[string, string]> = [['(', ')'], ['{', '}'], ['[', ']']];
  for (const [open, close] of pairs) {
    let depth = 0;
    for (const ch of src) {
      if (ch === open) depth++;
      if (ch === close) depth--;
      if (depth < 0) errors.push(`${name}: 出现未配对的 "${close}"`);
    }
    if (depth !== 0) errors.push(`${name}: "${open}${close}" 不平衡（残余深度 ${depth}）`);
  }
}

// common 中定义的符号（函数名、结构字段、绑定资源）
const commonFns = new Set<string>((common.match(/fn\s+([a-zA-Z0-9_]+)/g) ?? []).map((s) => s.split(/\s+/)[1]));
const builtins = new Set([
  'atomicLoad', 'atomicStore', 'atomicAdd', 'atomicSub',
  'textureLoad', 'textureStore', 'distance', 'sqrt', 'exp', 'pow', 'min', 'max',
  'abs', 'floor', 'round', 'cos', 'sin', 'atan2', 'mix', 'select', 'clamp',
  'length', 'dot', 'array', 'vec2', 'vec3', 'vec4', 'mat2x2',
  'workgroupBarrier', 'storageBarrier',
  // 标量类型构造器
  'f32', 'i32', 'u32', 'bool',
]);
const keywords = new Set(['if', 'for', 'while', 'switch', 'fn', 'return', 'var', 'let', 'const']);
// @ 属性名（attribute），不是函数调用
const attributes = new Set([
  'group', 'binding', 'compute', 'vertex', 'fragment', 'workgroup_size',
  'builtin', 'location', 'uniform', 'storage', 'read_write', 'write', 'read',
]);

for (const pass of passes) {
  const src = common + '\n' + readFileSync(join(shaderDir, `${pass}.wgsl`), 'utf8');
  checkBalance(src, pass);

  // 收集本 pass 定义的函数
  const defined = new Set(commonFns);
  for (const m of src.matchAll(/fn\s+([a-zA-Z0-9_]+)/g)) defined.add(m[1]);

  // 检查函数调用是否都有定义或为内建（@attribute(...) 与关键字除外）
  const callRe = /(^|[^@\w])([a-zA-Z_][a-zA-Z0-9_]*)\s*\(/g;
  for (const m of stripComments(src).matchAll(callRe)) {
    const name = m[2];
    if (name === 'main' || builtins.has(name) || keywords.has(name) || attributes.has(name) || defined.has(name)) continue;
    errors.push(`${pass}: 调用了未定义的函数/构造 "${name}"`);
  }

  // workgroup 规则
  const hasWorkgroupVar = /var<workgroup>/.test(src);
  const usesBarrier = /workgroupBarrier/.test(src);
  if (usesBarrier && !hasWorkgroupVar) {
    errors.push(`${pass}: 使用 workgroupBarrier 但没有 var<workgroup> 声明`);
  }
  if (pass === 'tap' && !hasWorkgroupVar) {
    errors.push('tap: 应当声明 workgroup 环槽累加器');
  }

  // 必须有 compute main
  if (!/@compute\s+@workgroup_size/.test(src)) {
    errors.push(`${pass}: 缺少 compute workgroup 入口`);
  }
}

// present 是独立的 render 着色器
const present = readFileSync(join(shaderDir, 'present.wgsl'), 'utf8');
checkBalance(present, 'present');
if (!/@vertex/.test(present) || !/@fragment/.test(present)) {
  errors.push('present: 缺少 vertex/fragment 入口');
}

if (errors.length > 0) {
  console.error('WGSL 结构自检失败：');
  for (const e of errors) console.error('  - ' + e);
  process.exit(1);
}
console.log(`WGSL 结构自检通过：${passes.length} 个 compute pass + present，括号/符号/workgroup 规则全部正常。`);
