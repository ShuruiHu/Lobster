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
| `core/src/gpu_runtime/tag.rs`、两种 top-k provenance 及 CUDA 两处概率查表 | 原编码 `±id` 将 `Neg(0)` 和 `Pos(0)` 都写成 `0`。统一编码为 `±(id+1)`，查表使用 `abs(literal)-1`，恢复时保留符号。 |
| `core/src/gpu_runtime/mod.rs` | 原自检使用只比较 tuple 的集合相等，`CHECK_TAGS` 仅打印。增加同一 tuple 的 provenance weight 比较，容差为 `1e-7 + 1e-5 × abs(CPU weight)`，拒绝非有限值。 |

tag 丢失的独立复现：两条边的概率为 `0.8, 0.7`，闭包输出应为 `0.8, 0.7, 0.56`，输出总和对输入的梯度应为 `1.7, 1.8`。原 GPU 返回全 `1` 和全 `0` 梯度。这个例子只使用符号程序。

独立检查还发现：Pacman 的中间结果 `(0,5)`，CPU proof 含 `Neg(0)`，GPU 回传成 `Pos(0)`，权重从 `0.8589723666` 变成 `0.04352761075`。普通 provenance 的事实表只追加，检查使用同一个 provenance 上下文。该差异来自 signed literal 编码。

可微和普通 top-k 共用此协议；设备和常量投影的 host 概率查表同步调整。设备仍使用原 `int16_t` literal，输入事实 ID 范围为 `0..32766`。Rust 和 CUDA 在窄化前校验范围，拒绝零编码和溢出。

等权 proof 会影响后续结果。例如 `{a,b}` 和 `{b,c}` 当前均为 `α²`，再与 `a` 合并时分别成为 `α²` 和 `α³`。tag 回传修复阶段记录过 Pacman 23/32 场景的部分分数差异，动作和 BFS 评价一致。

曾测试把 GPU `add` 从 `>` 改为 `>=`。这让当时的 32 场景输出一致，但另一个三事实程序出现相反偏差：`p=a∧b; p=b∧c; q=p∧a`，事实概率均为 `0.5`，CPU `q=0.125`，候选 GPU `q=0.25`。该反例说明局部左优先不能统一整体 proof 顺序。最终源码保留原来的 `>`。

上述 CUDA 文件在原始提交与 `5b17581` 中相同。CPU provenance、规则、CNN 和权重沿用现有实现。FFI 类型修正解决对象类型错误，tag 丢失有单独的复现证据。

`NO_CHECK=1` 仍用于 GPU-only 执行。正常自检仍执行 CPU 参考，新增检查比较数值权重。proof 身份和梯度通过独立回归检查。

## 递归 proof 与算子修复

逐项对照 CPU `dynamic/relation.rs`、`schedulers/lfp.rs`、`schedulers/utils.rs` 和 GPU 实现，使用独立小程序及实际 proof 树确认以下问题。

| 位置 | 确认的问题 | 修改 |
|---|---|---|
| `libsclgpu/dataflow.cu` | 只看 stable 元组数量，遗漏最后 recent 中的更优 proof；更差候选也继续传播。 | 按 CPU 调度合同筛选有效前沿：严格改善进入 recent，同权保留 stable，移除被替换的旧 stable；有效 recent 非空时继续。筛选与搬运在 GPU 执行。 |
| `libsclgpu/merge_specialized.cu` | 最后一个 key 直接复制首个 tag，漏合并重复项。 | 以输入长度作为最后一组的右边界，统一执行 tag 合并。 |
| `gpu_runtime/tag.rs`、`libsclgpu/bindings.h`、`provenance.*`、`project_constant.cu` | f32 评分把真实不等权 proof 舍入为同权，后续共享事实去重放大偏差。 | 概率表、FFI、设备和 host 评分统一使用 CPU 的 f64 精度。 |
| `provenance.cu`、`project_constant.cu` | 逻辑零在乘法中成为真元，在加法中覆盖概率为零的有效 proof。 | 恢复乘法零吸收和加法零身份，保留普通同权比较 `>`。 |

实际反例：晚到三跳路径应为 `0.9×0.8×0.7=0.504`，原 GPU 返回 shortcut `0.12`；末尾 key 更新应为 `0.9×0.8=0.72`，原 GPU 保留 `0.1`；更差候选传播使原 GPU 返回 `0.3`，CPU 为 `0.2`。这些例子分别隔离收敛、merge 和前沿筛选。

Pacman 首场景 `path_(0,20)` 的实际旧 stable 路径为 `0→5→10→15→20`，权重 `0.11874668435883295`。GPU 第 8 轮 recent 已含权重 `0.1789995377939982` 的较优路径，原循环此时停止并只返回 stable。逐轮记录明确定位到丢失前沿。

`>=` 反例的调用顺序也已核对：CPU 从 `to_add_batches.pop()` 先处理最后一条规则，再由稳定 top-1 保留该 proof；GPU 顺序 append 后以原 `>` 保留后者。两端在该小例中原本一致。稳定前沿更新则使用 CPU 的 `add(stable, candidate)` 合同，同权保留旧 proof。两个调用点分别处理。

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
- signed literal：事实 0 的补集、概率及梯度，常量投影；正负 ID 与设备边界往返检查。
- Pacman-Maze：固定 32 个 5×5 场景，预训练 CNN 识别格子，再由符号规则选择下一步避敌、接近目标的动作。比较全部概率、分数及动作。
- CUDA 跟踪：CPU import、context 创建、run；GPU 实际 offload；同进程 CUDA context 同步并检查返回值。

Boolean TC 与概率推理分别验证。回归比较完整概率、梯度和 proof 传播，记录 GPU-only 输出及原始跟踪。CPU 实现作为兼容参考。论文实验的精确提交和逐项概率日志需另行核对。
