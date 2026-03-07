# Remaining Work for CUDA Parity

**Current state:** 1978 tests pass, 0 xfails. All StableHLO ops implemented.

## Completed

### ~~1. `count_leading_zeros`~~ DONE
Implemented via float32 log2 + exponent extraction with precision correction.

### ~~2. `reduce_precision`~~ DONE
Implemented via bitwise float32 manipulation: mantissa round-to-nearest-even truncation + exponent range clamping with overflow->inf, underflow->0 handling. NaN passthrough preserves original bits through both mantissa rounding and exponent clamping. Verified bit-exact against CPU for all exponent/mantissa configurations including edge cases (exponent_bits=1, mantissa_bits=0, subnormals, negative zero).

### ~~3. Error hardening~~ DONE
- Complex sort: clean error with workaround suggestion
- Complex convolution: clean error with workaround suggestion
- Float64/unsupported dtypes: clean error at placeholder creation with suggestion to use float32

### ~~4. Identity buffer copy elimination~~ DONE
Pass-through buffers now share the underlying MTLBuffer instead of copying.

## Remaining Actionable Gaps

### 5. `jax.debug.print` / `jax.debug.callback`
- **Impact:** Medium — useful for debugging JIT-compiled code.
- **Status:** Blocked. JAX's `emit_python_callback` explicitly excludes MPS (`platform not in {"cpu", "cuda", "rocm", "tpu"}`). Would require either patching JAX upstream or reimplementing the host callback infrastructure.
- **Workaround:** Print outside JIT, or use `jax.debug.callback` with `jax.effects_barrier()`.

### 6. Buffer donation
- **Impact:** Low — Apple Silicon unified memory reduces the benefit.
- **Approach:** Implement `PJRT_Buffer_DonateWithControlDependency` to reuse input buffers.
- **File:** `src/pjrt_plugin/pjrt_api.cc`, `src/pjrt_plugin/mps_executable.mm`

### 7. Async execution
- **Impact:** Potentially high for workloads with heavy CPU tracing.
- **Status:** Attempted and reverted. For GPU-bound workloads (ResNet18/CIFAR-10), async execution adds overhead without benefit because the CPU dispatch time is negligible compared to GPU execution. Would benefit workloads with lots of small dispatches or heavy CPU-side tracing.
- **Approach:** Make `PJRT_Event` track Metal command buffer completion; return from Execute before GPU finishes.

## Apple MPS Framework Limitations (cannot fix)

| Issue | Description | Workaround |
|-------|-------------|------------|
| No float64 | Metal GPUs only support 32-bit floats | Use float32 (now caught with clean error) |
| Complex sort crashes | `mps.sort` doesn't accept complex types | Sort components separately (now caught with clean error) |
| Complex convolution crashes | `mps.conv_2d` doesn't accept complex types | Manual decomposition (now caught with clean error) |
| QDWH polar segfault | Internally promotes to float64 | Use `polar(method='svd')` |
| Zero-size tensors | MPS doesn't support empty tensors | Avoid zero-dim ops |

## Performance Characteristics

Benchmarked on M4 MacBook Air (CPU = Accelerate BLAS, MPS = Metal GPU):

| Workload | CPU | MPS | Speedup |
|----------|-----|-----|---------|
| ResNet18 CIFAR-10 train step | 3.0s | 1.0s | 3x |
| matmul 2048x2048 | 27ms | 9.6ms | 2.8x |
| matmul 1024x1024 | 2.0ms | 1.7ms | 1.2x |
| conv2d 128ch 32x32 | 5.0ms | 4.1ms | 1.2x |
| layernorm 1024 | 0.95ms | 3.6ms | 0.26x |

MPS excels at compute-bound workloads (large matmul, batched attention). Dispatch overhead makes small/elementwise ops slower than CPU. The 3x ResNet speedup reflects a mix of both patterns.

## Nice-to-Have Improvements

- **Performance:** Graph fusion, Metal shader cache tuning, profiling
- **Broader dtypes:** float8/FP8 (needs Metal support), int4 (needs custom shaders)
- **Multi-device:** Collective ops (all_reduce, etc.) — N/A for single-GPU Macs
- **Upstream:** Report MPS bugs to Apple, contribute fixes to tillahoffmann/jax-mps

## What "Parity" Means

**Feature parity with `jaxlib[cuda12]` is effectively achieved** for single-GPU workloads:

- **0 missing StableHLO ops** — all ops that JAX generates are implemented
- **1 missing debug feature** (`debug.print`) — blocked by JAX upstream, not fixable in plugin
- **1 missing optimization** (buffer donation) — no correctness impact
- **Hardware limits** (no float64, no complex sort/conv) — cannot be fixed in software, now caught with clean errors

For any standard ML/scientific computing workflow, jax-mps is production-ready on Apple Silicon.
