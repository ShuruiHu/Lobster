# Lobster 修复记录

源码基线：`76a151f01a81b902606a7423bc0772d2e337532c`。
公共构建调整和 CPU 初始化修复始于本 fork 的 `5b17581dff78bd2ed690280fd7d6eafd2c1c694b`。
以下修改直接保存在本仓库源码中。

## CUDA 初始化入口

- 原 `incremental.rs` 在读取 GPU 选择前调用 `libsclgpu_init()`。该函数调用 CUDA、启动 dummy kernel 并同步，CPU 首次 `ctx.run()` 因此使用 GPU。
- 现在先执行 `GpuExecutionContext::new_from_env`，仅在返回 `Some` 后初始化 CUDA。
- 删除 `integrate/interpret.rs` 的重复初始化。普通 Python 执行沿 `InterpretContext::run → execute → incremental_execute_helper` 进入统一入口。
- CUDA 初始化函数和 GPU 算子保留。monitor 执行沿独立 CPU 路径运行。

## GPU 概率输出

| 源码位置 | 原因与修改 |
|---|---|
| `core/src/gpu_runtime/libsclgpu/lib.cu` | 非 unit 输出的 tag 转换循环被注释。恢复实际 `host_tags` 回传。 |
| `core/src/gpu_runtime/libsclgpu/provenance.h` | `C_Proof.literals` 声明为 `Array<int32_t>`，原 placement-new 构造 `Array<float>`。统一为 `Array<int32_t>`。 |
| `core/src/gpu_runtime/mod.rs` | 原自检使用只比较 tuple 的集合相等，`CHECK_TAGS` 仅打印。增加同一 tuple 的 provenance weight 比较，容差为 `1e-7 + 1e-5 × abs(CPU weight)`，拒绝非有限值。 |

tag 丢失的独立复现：两条边的概率为 `0.8, 0.7`，闭包输出应为 `0.8, 0.7, 0.56`，输出总和对输入的梯度应为 `1.7, 1.8`。原 GPU 返回全 `1` 和全 `0` 梯度。这个例子只使用符号程序。

等权 proof 会影响后续结果。例如 `{a,b}` 和 `{b,c}` 当前均为 `α²`，再与 `a` 合并时分别成为 `α²` 和 `α³`。恢复 tag 后，Pacman 的 32 个场景中有 23 个场景存在部分分数差异，动作和 BFS 评价一致。

曾测试把 GPU `add` 从 `>` 改为 `>=`。这让 32 场景输出一致，但另一个三事实程序出现相反偏差：`p=a∧b; p=b∧c; q=p∧a`，事实概率均为 `0.5`，CPU `q=0.125`，候选 GPU `q=0.25`。局部左优先并不能统一 CPU/GPU 整体求值顺序，因此最终源码保留原来的 `>`。这项差异属于现有 top-1 proof 剪枝与求值顺序的限制，单独记录。

上述 CUDA 文件在原始提交与 `5b17581` 中相同。CPU provenance、规则、CNN 和权重沿用现有实现。FFI 类型修正解决对象类型错误，tag 丢失有单独的复现证据。

`NO_CHECK=1` 仍用于 GPU-only 执行。正常自检仍执行 CPU 参考，新增检查比较数值权重。proof 身份和梯度通过独立回归检查。

## 公共构建调整

- `core/build.rs` 使用源码相对路径，支持 `SCLGPU_LIB_DIR`、`CUDA_HOME`，并生成缺失的 parser。
- CMake 支持显式 CUDA 架构；A30 使用 `80`。
- 补齐原提交缺失的四个 Scallop 模块，内容来自 `da21ec89518debffe9069f4e982570bd0b299d4e`。适配当前 Rust `extract_if(..)` API。
- 仓库保存 `Cargo.lock`，release 构建使用 `--locked`。

实际构建组合：Linux x86_64、A30、驱动 `565.57.01`、CUDA `12.5.82`、GCC `9.4.0`、Python `3.11.14`、Rust `nightly-2026-09-24`、CMake `3.31.0-rc2`。
CCCL 固定为 `5a87e5d5706efa2d59e8d8281c89f093cd48280f`，Catch2 固定为 `6e79e682b726f524310d55dec8ddac4e9c52fb5f`。这组版本记录本次实际构建环境。

## 回归范围

- Boolean unit TC：空图、DAG、环、自环、重复边、不连通图和 u32 边界值，完整输出对照独立 BFS。
- 概率 DAG：CPU/GPU 完整概率和输入梯度；新 context 重复执行；正常自检与 `NO_CHECK=1` 分别运行。
- Pacman-Maze：固定 32 个 5×5 场景，预训练 CNN 识别格子，再由符号规则选择下一步避敌、接近目标的动作。比较全部概率、分数及动作。
- CUDA 跟踪：CPU import、context 创建、run；GPU 实际 offload；同进程 CUDA context 同步并检查返回值。

GPU 使用 f32 权重、并行规约和现有收敛条件。Boolean TC 与概率推理分别验证；Pacman 正式实验使用 CPU 符号路径，GPU 完整概率差异保存在诊断记录中。论文实验的精确提交和逐项概率日志需另行核对。
