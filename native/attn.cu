// magpie's attention for sm_120 as a torch extension: out = softmax(q k^T * scale) v for already quantised operands,
// the kernel in attn_sm120.cuh (a fork of SageAttention 2's). all tensors are (batch, tokens, heads, 128) except the
// values, which are (batch, 128, heads, keys padded to ATTN_CTA_K) fp8 with the tokens shuffled in 16s:
//   q, k   int8, with one fp32 scale per ATTN_WARP_Q queries and per ATTN_WARP_K keys
//   v      fp8, one fp32 scale per (head, channel), applied to the output. with ATTN_PV4=1: nvfp4 instead, as
//          (batch, heads, key tiles, 4608) bytes, and `v_scale` is the per (head, channel) mean added to the output
//   out    bf16
// compile-time settings, searched by tools/tune_attn.py: queries and keys per thread block, queries and keys per warp
// (a block runs (CTA_Q / WARP_Q) x (CTA_K / WARP_K) warps), and ATTN_INST_BUF (1: each key tile's products are summed in
// a fresh accumulator and then added, which upstream needs for accuracy on ada).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#ifndef ATTN_CTA_Q
#define ATTN_CTA_Q 128
#endif
#ifndef ATTN_CTA_K
#define ATTN_CTA_K 64
#endif
#ifndef ATTN_WARP_Q
#define ATTN_WARP_Q 32
#endif
#ifndef ATTN_WARP_K
#define ATTN_WARP_K 64
#endif
#ifndef ATTN_INST_BUF
#define ATTN_INST_BUF 1
#endif
#ifndef ATTN_PV4
#define ATTN_PV4 0
#endif
#include "attn_sm120.cuh"

constexpr int HEAD_DIM = 128;

void attend(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor out, torch::Tensor q_scale, torch::Tensor k_scale,
            torch::Tensor v_scale, float sm_scale) {
  const int batch = q.size(0), queries = q.size(1), heads = q.size(2), keys = k.size(1);
  TORCH_CHECK(q.size(3) == HEAD_DIM && k.size(3) == HEAD_DIM && v.is_contiguous(), "attn: bad operand layout");
  TORCH_CHECK(ATTN_PV4 ? (v.size(2) >= div_ceil(keys, 64) && v.size(3) == 4608 && ATTN_CTA_K == 64)
                       : (v.size(1) == HEAD_DIM && v.size(3) >= div_ceil(keys, ATTN_CTA_K) * ATTN_CTA_K), "attn: values are not laid out for this build");
  TORCH_CHECK(q_scale.size(2) == div_ceil(queries, ATTN_CTA_Q) * (ATTN_CTA_Q / ATTN_WARP_Q), "attn: wrong number of query scales");
  TORCH_CHECK(k_scale.size(2) == div_ceil(keys, ATTN_CTA_K) * (ATTN_CTA_K / ATTN_WARP_K), "attn: wrong number of key scales");
  size_t smem = std::max(ATTN_CTA_Q * HEAD_DIM + 2 * ATTN_CTA_K * HEAD_DIM, ATTN_CTA_Q * HEAD_DIM * 2);
  auto kernel = attn_sm120_kernel<ATTN_CTA_Q, ATTN_CTA_K, ATTN_WARP_Q, ATTN_WARP_K, HEAD_DIM, DataType::kInt8, QuantGranularity::kPerWarp,
                                  QuantGranularity::kPerWarp, float, ATTN_INST_BUF == 1, nv_bfloat16, ComputeUnit::kCudaCore, MaskMode::kNone, false, !ATTN_PV4, ATTN_PV4 == 1, false>;
  cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  dim3 grid(div_ceil(queries, ATTN_CTA_Q), heads, batch), block(32, (ATTN_CTA_Q / ATTN_WARP_Q) * (ATTN_CTA_K / ATTN_WARP_K));
  kernel<<<grid, block, smem, at::cuda::getCurrentCUDAStream()>>>(
      q.data_ptr<int8_t>(), k.data_ptr<int8_t>(), reinterpret_cast<int8_t *>(v.data_ptr()), reinterpret_cast<nv_bfloat16 *>(out.data_ptr()), nullptr,
      q_scale.data_ptr<float>(), k_scale.data_ptr<float>(), v_scale.data_ptr<float>(), v_scale.data_ptr<float>(), queries, keys, 1,
      q.stride(0), q.stride(1), q.stride(2), k.stride(0), k.stride(1), k.stride(2), v.stride(0), v.stride(ATTN_PV4 ? 1 : 2), v.stride(1),
      out.stride(0), out.stride(1), out.stride(2), sm_scale);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) { module.def("attend", &attend); }
