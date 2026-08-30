# Fused RMSNorm forward

The kernel targets the transformer hot path `y = RMSNorm(x + residual) * gamma`. It fuses residual addition,
sum-of-squares reduction, inverse RMS, and affine scaling into one launch, avoiding intermediate global-memory writes.

## Contract

- Input, optional residual, and output are two-dimensional CUDA tensors of the same dtype (`fp16` or `fp32`).
- `row_stride` is in elements and may exceed `cols`; the final dimension is contiguous.
- `weight` is a `[1, cols]` tensor with `row_stride >= cols`.
- `epsilon` is finite and positive. The optional `inv_rms` output contains `rows` `float` values.
- Calls are stream ordered and asynchronous. A returned `kSuccess` means launch submission succeeded; synchronize the
  stream to observe asynchronous device errors.

## Performance model

This is normally memory-bandwidth limited. With residual enabled, each element performs four element-sized global
memory transactions: input read, residual read, gamma read, and output write. Benchmark output reports this effective
bandwidth, which makes regressions portable across GPUs. Compare only the same workload, dtype, GPU clocks, and CUDA
version.

## Extension points

Backward, BF16, vectorized `half2`, persistent multi-row blocks, and framework adapters should be separate dispatch
paths. Keep the public `TensorView` contract stable, add a reference test before each path, and tune per architecture
instead of assuming a universal launch configuration.

