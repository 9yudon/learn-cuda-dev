# CUDA Operator Engineering Lab

一个面向真实部署流程的独立 CUDA 算子工程。它以框架无关的 C++ API 为核心，当前实现了 Transformer 中常见的
**fused RMSNorm forward**（可选残差、FP16/FP32、非紧凑行间距、可选 `inv_rms` 输出）。目录与流程为后续算子、
PyTorch 插件或推理引擎适配层预留了位置。

## 工程组成

| 路径 | 职责 |
| --- | --- |
| `include/cuda_ops/` | 稳定的公共 API、张量契约和 CUDA 错误检查 |
| `src/` | 实际 CUDA 内核与按 dtype/功能组合的调度 |
| `tests/` | 不依赖第三方测试框架的 GPU 正确性/参数契约测试 |
| `benchmarks/` | CUDA Event 计时和 JSON 性能输出，便于 CI 收集 |
| `tools/autotune.py` | 在 128/256/512 线程候选项间构建、压测并保存最优配置 |
| `scripts/profile.ps1` | Nsight Compute 全指标采样入口 |
| `.github/workflows/` | CUDA 容器编译与 self-hosted GPU 测试分层 CI |

构建产物可通过 `cmake --install build/release --prefix <install-prefix>` 安装；下游 CMake 项目可以使用
`find_package(cuda_ops CONFIG REQUIRED)` 和目标 `cuda_ops::cuda_ops`。可直接验证随附的
`examples/consumer`：`cmake -S examples/consumer -B build/consumer -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 -Dcuda_ops_DIR=<install-prefix>/lib/cmake/cuda_ops`。
使用静态库的下游可执行文件需启用 CUDA language 并设置 `CUDA_RESOLVE_DEVICE_SYMBOLS ON`；示例已完整演示。

## 快速开始

需要 CUDA Toolkit、CMake 3.26+、Ninja，以及可用的 NVIDIA GPU。Windows 下先调用 `tools\\devshell.cmd`，它会激活
MSVC 主机编译器并将当前用户安装的 CMake/Ninja 加入这个 shell。将架构列表替换为部署 GPU 的 Compute Capability，避免无用 fatbin：

```powershell
cmd /d /c "call tools\devshell.cmd && cmake -S . -B build/release -G Ninja -DCMAKE_BUILD_TYPE=Release -DCUDA_OPS_CUDA_ARCHITECTURES=86 && cmake --build build/release --parallel && ctest --test-dir build/release --output-on-failure"
# 或在已激活的 Developer Command Prompt 中逐条运行：
cmake -S . -B build/release -G Ninja -DCMAKE_BUILD_TYPE=Release -DCUDA_OPS_CUDA_ARCHITECTURES=89
cmake --build build/release --parallel
ctest --test-dir build/release --output-on-failure
.\build\release\rmsnorm_bench.exe --rows 4096 --cols 4096 --iterations 300
```

Linux 下将可执行文件扩展名 `.exe` 去掉。容器化构建可运行：

```bash
docker build -t cuda-ops .
docker run --rm --gpus all cuda-ops
```

## 性能工作流

先执行对应真实模型 hidden size、batch/token 数和 dtype 的基准，再把线程数调优结果纳入部署配置：

```powershell
python tools/autotune.py --architectures 89 --rows 4096 --cols 4096
.\scripts\profile.ps1 -BuildDir build\release -Rows 4096 -Cols 4096
```

`rmsnorm_bench` 的 JSON 中 `effective_gbps` 是基于最少读写流量估算的有效带宽，而非峰值带宽。更完整的算子契约、
性能模型和扩展方式见 [docs/operators/rmsnorm.md](docs/operators/rmsnorm.md)。

### Windows 工具链与限制

`tools\devshell.cmd` 会选取本机已安装的 MSVC Build Tools，并为本 shell 配置 CMake/Ninja；可先运行
`cmd /k tools\devshell.cmd` 后再执行任意构建命令。Nsight Compute 需要账户具有 NVIDIA GPU Performance Counters
权限；若出现 `ERR_NVGPUCTRPERM`，请由设备管理员按错误提示启用权限后再运行 `scripts\profile.ps1`。若组织的
Device Guard 拦截新生成的可执行文件，则需要管理员将本项目的构建输出加入信任策略；不要试图绕过该策略。

## 生产接入建议

1. 在调用框架适配层中显式校验 device、dtype、shape、stride 和当前 stream。
2. 用生产请求的 shape bucket 建立基准门槛，并保留同 GPU/CUDA 版本的 JSON 结果。
3. 每条新优化路径必须有 CPU/框架参考实现的误差测试，并覆盖边界维度和非连续行间距。
4. 在 GPU runner 上运行正确性与性能门禁；普通 CI 只做 CUDA 编译，避免无 GPU 的伪测试。
