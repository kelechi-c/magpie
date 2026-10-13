// magpie's attention kernel for sm_120, forked from SageAttention 2's sm89 kernel (csrc/qattn/qk_int_sv_f8_cuda_sm89.cuh
// at d1a57a5, Apache-2.0, notice below): int8 q x k and fp8 weights x values on the tensor cores, fp32 accumulators.
// it builds against the rest of SageAttention's csrc/ headers. what differs from upstream:
//   ATTN_LAZY    1: the output and denominator are rescaled only when some row's running maximum moved (the scale is
//                exactly 1 otherwise, so the result is bit-identical); upstream multiplies them on every key tile.
//   ATTN_PV4     1: the weights x values matmul runs on blackwell's block-scaled 4-bit instruction (mxf4nvf4, 64 keys
//                per instruction instead of 32). the values arrive as nvfp4 (see `load_fp4_values`), the weights are
//                rounded to nvfp4 in registers (`weights_to_fp4`), and the denominator sums the rounded weights.
//                the values are rounded in a hadamard basis of the 128 channels (`values_fp4_kernel`), which spreads
//                a channel's outliers; the output is turned back before it is stored (`turn_back`).
//   ATTN_ABLATE  profiling only: 1 skips the q x k matmul, 2 the softmax update, 3 the weights x values matmul,
//                5 / 6 / 7 run the weights x values matmul / the q x k matmul / both at the 4-bit instruction's cost,
//                4 the key and value loads, in the main loop. the output is then wrong by design.
#ifndef ATTN_LAZY
#define ATTN_LAZY 0
#endif
#ifndef ATTN_PV4
#define ATTN_PV4 0
#endif
#ifndef ATTN_ABLATE
#define ATTN_ABLATE 0
#endif
/*
 * Copyright (c) 2024 by SageAttention team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "../utils.cuh"
#include <cuda_fp16.h>
#include <cuda_pipeline_primitives.h>
#include <torch/extension.h>

#include "cp_async.cuh"
#include "mma.cuh"
#include "permuted_smem.cuh"
#include "math.cuh"
#include "dispatch_utils.h"

#include "attn_utils.cuh"

// blackwell's block-scaled 4-bit matmul: C (16 x 8, fp32) += A (16 x 64, e2m1) x B (8 x 64, e2m1), one e4m3 scale per
// 16 values along the contraction for each row of A and each column of B. A and B are packed nibbles in the layout of
// the 8-bit instruction with twice the values per register; `sfa` holds a row's 4 scales (even lanes: row lane / 4,
// odd lanes: 8 further), `sfb` a column's (column lane / 4).
__device__ __forceinline__ void mma_fp4(float *C, const uint32_t *A, const uint32_t *B, uint32_t sfa, uint32_t sfb)
{
  static constexpr uint16_t zero = 0;
  asm volatile(
      "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
      : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
      : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]),
        "r"(sfa), "h"(zero), "h"(zero), "r"(sfb), "h"(zero), "h"(zero));
}

constexpr uint32_t FP4_ONES = 0x38383838; // four e4m3 scales of 1.0

constexpr uint32_t FP4_TILE_LINES = 288; // 16-byte lines per 64-key tile of nvfp4 values: 256 of nibbles, 32 of scales

// one 64-key tile of nvfp4 values, global -> shared, asynchronously. a tile is 128 channel rows of 32 bytes (64 keys,
// two per byte, in slot order: see `weights_to_fp4`), then one word per channel holding its 4 block scales. the two
// 16-byte halves of a row swap places every 4 rows so the 8 rows one ldmatrix reads fall in 8 different banks.
__device__ __forceinline__ void load_fp4_values(const int8_t *tile, b128_t *smem_v, uint32_t threads)
{
  for (uint32_t line = threadIdx.y * 32 + threadIdx.x; line < FP4_TILE_LINES; line += threads)
  {
    const uint32_t row = line / 2;
    b128_t *target = smem_v + (line < 256 ? row * 2 + ((line % 2) ^ ((row >> 2) & 1)) : line);
    cp_async::load_128b<cp_async::PrefetchMode::kPrefetch>(target, reinterpret_cast<const b128_t *>(tile) + line);
  }
}

// 8 floats -> 8 e2m1 nibbles in one register (first value lowest), round to nearest, saturating at 6: blackwell's
// own conversion instruction.
__device__ __forceinline__ uint32_t fp4_pack(const float *x)
{
  uint32_t packed;
  asm volatile("{\n.reg .b8 b0, b1, b2, b3;\n"
               "cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\ncvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n"
               "cvt.rn.satfinite.e2m1x2.f32 b2, %6, %5;\ncvt.rn.satfinite.e2m1x2.f32 b3, %8, %7;\n"
               "mov.b32 %0, {b0, b1, b2, b3};\n}"
               : "=r"(packed) : "f"(x[0]), "f"(x[1]), "f"(x[2]), "f"(x[3]), "f"(x[4]), "f"(x[5]), "f"(x[6]), "f"(x[7]));
  return packed;
}

// the softmax weights of one warp (32 queries x 64 keys, fp32, in the accumulator layout of the q x k matmul) ->
// the A operand of the 4-bit matmul, and their row sums added to the denominator.
// a thread holds, for each of its 2 rows per 16-row tile, keys 8i + 2t + {0, 1} (t = lane % 4, i = 0..7); the 4-bit
// instruction wants it to hold slots 8t..8t+7 and 32+8t..32+8t+7. so slot s of a 64-key tile is key
// 32h + 8(j / 2) + 2t + j % 2 with h = s / 32, t = (s % 32) / 8, j = s % 8, and the values are stored in that order.
// one scale covers 16 slots, the 8 of this thread and the 8 of lane ^ 1: the largest weight over 6, rounded up to
// e4m3, at least 2^-6. each lane ends up with the 4 scales of one row (even lanes the first row, odd the second).
template <uint32_t num_tiles_q, uint32_t num_tiles_k>
__device__ __forceinline__ void weights_to_fp4(float RS[][num_tiles_k][8], uint32_t RA[][4], uint32_t SF[], float d[][2])
{
  static_assert(num_tiles_k == 4, "the 4-bit weights x values matmul needs 64 keys per warp");
  const uint32_t lane = get_lane_id();
  uint32_t ones[2] = {0x22222222, 0x22222222}; // e2m1 1.0 in every slot
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++)
  {
    uint32_t words[2];
#pragma unroll
    for (uint32_t k = 0; k < 2; k++)
    {
      uint32_t word = 0;
#pragma unroll
      for (uint32_t h = 0; h < 2; h++)
      {
        float x[8];
        float top = 0.f;
#pragma unroll
        for (uint32_t j = 0; j < 8; j++)
        {
          x[j] = RS[fq][2 * h + j / 4][k * 2 + 4 * ((j / 2) % 2) + j % 2];
          top = max(top, x[j]);
        }
        top = max(top, __shfl_xor_sync(0xffffffff, top, 0x1));
        const uint32_t bits = (__float_as_uint(max(top * (1.f / 6.f), 0.015625f)) + 0xFFFFF) & 0xFFF00000;
        const float inverse = math::ptx_rcp(__uint_as_float(bits));
#pragma unroll
        for (uint32_t j = 0; j < 8; j++)
          x[j] *= inverse;
        RA[fq][k + 2 * h] = fp4_pack(x);
        word |= ((bits >> 20) - 960) << (8 * (2 * h + (lane % 4) / 2));
      }
      words[k] = word | __shfl_xor_sync(0xffffffff, word, 0x2);
    }
    SF[fq] = words[lane % 2];
    float sum[4] = {0.f, 0.f, 0.f, 0.f};
    mma_fp4(sum, RA[fq], ones, SF[fq], FP4_ONES);
    d[fq][0] += 0.25f * sum[0]; // every lane of a row gets the whole row sum, and the 4 are added at the end
    d[fq][1] += 0.25f * sum[2];
  }
}

// output (32 queries x 128 channels per warp) += nvfp4 weights x nvfp4 values for one 64-key tile.
template <uint32_t num_tiles_q, uint32_t num_tiles_v>
__device__ __forceinline__ void values_fp4(b128_t *smem_v, uint32_t RA[][4], uint32_t SF[], float RO[][num_tiles_v][8])
{
  const uint32_t lane = get_lane_id();
  const uint32_t row = lane % 8 + (lane / 16) * 8, half = ((lane / 8) % 2) ^ ((lane >> 2) & 1);
  const uint32_t *scales = reinterpret_cast<const uint32_t *>(smem_v + 256);
#pragma unroll
  for (uint32_t fv = 0; fv < num_tiles_v; fv++)
  {
    uint32_t RV[4];
    mma::ldmatrix_m8n8x4(RV, smem_v + (fv * 16 + row) * 2 + half);
    const uint32_t first = scales[fv * 16 + lane / 4], second = scales[fv * 16 + 8 + lane / 4];
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++)
    {
      mma_fp4(RO[fq][fv], RA[fq], RV, SF[fq], first);
      mma_fp4(RO[fq][fv] + 4, RA[fq], RV + 2, SF[fq], second);
    }
  }
}

// one warp's output (32 queries x 128 channels in registers) times the 128 x 128 hadamard matrix over 128^0.5, in
// place: the values were rounded in that basis, and the matrix is its own inverse. channel c of a row sits in tile
// fv = c / 16, element (c / 8) % 2 * 4 + c % 2 of lane % 4 == (c / 2) % 4, so five of the seven butterfly stages
// stay inside a thread and two exchange with lane ^ 1 and lane ^ 2. once per thread block, not per key tile.
template <uint32_t num_tiles_q, uint32_t num_tiles_v>
__device__ __forceinline__ void turn_back(float RO[][num_tiles_v][8])
{
  static_assert(num_tiles_v == 8, "the hadamard turn is written for 128 channels");
  const uint32_t lane = get_lane_id();
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++)
  {
#pragma unroll
    for (uint32_t fv = 0; fv < num_tiles_v; fv++)
    {
#pragma unroll
      for (uint32_t i = 0; i < 8; i += 2) // channel bit 0
      {
        const float a = RO[fq][fv][i], b = RO[fq][fv][i + 1];
        RO[fq][fv][i] = a + b;
        RO[fq][fv][i + 1] = a - b;
      }
#pragma unroll
      for (uint32_t i = 0; i < 4; i++) // channel bit 3
      {
        const float a = RO[fq][fv][i], b = RO[fq][fv][i + 4];
        RO[fq][fv][i] = a + b;
        RO[fq][fv][i + 4] = a - b;
      }
#pragma unroll
      for (uint32_t i = 0; i < 8; i++) // channel bits 1 and 2: across the 4 lanes of a row
      {
        float other = __shfl_xor_sync(0xffffffff, RO[fq][fv][i], 0x1);
        RO[fq][fv][i] = (lane & 1) ? other - RO[fq][fv][i] : RO[fq][fv][i] + other;
        other = __shfl_xor_sync(0xffffffff, RO[fq][fv][i], 0x2);
        RO[fq][fv][i] = (lane & 2) ? other - RO[fq][fv][i] : RO[fq][fv][i] + other;
      }
    }
#pragma unroll
    for (uint32_t span = 1; span < num_tiles_v; span *= 2) // channel bits 4 to 6
#pragma unroll
      for (uint32_t fv = 0; fv < num_tiles_v; fv++)
        if ((fv & span) == 0)
#pragma unroll
          for (uint32_t i = 0; i < 8; i++)
          {
            const float a = RO[fq][fv][i], b = RO[fq][fv + span][i];
            RO[fq][fv][i] = a + b;
            RO[fq][fv + span][i] = a - b;
          }
#pragma unroll
    for (uint32_t fv = 0; fv < num_tiles_v; fv++)
#pragma unroll
      for (uint32_t i = 0; i < 8; i++)
        RO[fq][fv][i] *= 0.08838834764831845f;
  }
}

#ifndef ATTN_TRIES
#define ATTN_TRIES 4
#endif

// an e2m1 nibble -> its value: sign, then 0, 0.5, 1, 1.5, 2, 3, 4, 6 (from 1 up: a float's exponent and top
// mantissa bit).
__device__ __forceinline__ float fp4_value(uint32_t nibble)
{
  const uint32_t code = nibble & 7;
  const float size = code < 2 ? 0.5f * code : __uint_as_float(((126 + (code >> 1)) << 23) | ((code & 1) << 22));
  return __uint_as_float(__float_as_uint(size) | ((nibble & 8) << 28));
}

// bf16 values (tokens, heads, 128) -> the nvfp4 tiles `load_fp4_values` reads, one thread block of 128 threads per
// (64-key tile, head): row 0 of `v` is the first key of tile `first_tile`. per tile: every token's 128 channels are turned by the hadamard
// matrix over 128^0.5, `mean` (heads, 128, in the turned basis) is subtracted, and each channel's 64 keys go out in
// slot order (see `weights_to_fp4`) as e2m1 nibbles with one e4m3 scale per 16 slots. the scale is the block's
// largest magnitude over 6, or that times a shrink where clipping the largest costs less squared error than a
// coarser grid for the rest (ATTN_TRIES candidates). keys past `tokens` are zeros.
// first every thread turns half of one token's channels in registers (six butterfly stages over 64 values read
// straight from global memory) and leaves them in shared memory; then every thread owns one channel, and the
// seventh stage is the sum or difference of the two halves as it reads its 64 keys.
__global__ void values_fp4_kernel(const nv_bfloat16 *__restrict__ v, const float *__restrict__ mean, uint8_t *__restrict__ out,
                                  const uint32_t tokens, const uint32_t heads, const uint32_t first_tile, const uint32_t tiles)
{
  constexpr float shrinks[4] = {1.f, 0.7f, 0.85f, 0.55f};
  __shared__ float x[64][128];
  const uint32_t tile = first_tile + blockIdx.x, head = blockIdx.y, t = threadIdx.x;
  {
    const uint32_t key = blockIdx.x * 64 + t % 64, half = t / 64;
    float r[64];
    if (key < tokens)
    {
      const uint4 *row = reinterpret_cast<const uint4 *>(v + (size_t(key) * heads + head) * 128 + half * 64);
#pragma unroll
      for (uint32_t i = 0; i < 8; i++)
      {
        const uint4 w = row[i];
        const uint32_t words[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
        for (uint32_t j = 0; j < 4; j++)
        {
          r[i * 8 + j * 2] = __uint_as_float(words[j] << 16); // a bf16 is the top half of an fp32
          r[i * 8 + j * 2 + 1] = __uint_as_float(words[j] & 0xFFFF0000);
        }
      }
#pragma unroll
      for (uint32_t span = 1; span < 64; span *= 2)
#pragma unroll
        for (uint32_t c = 0; c < 64; c++)
          if ((c & span) == 0)
          {
            const float a = r[c], b = r[c + span];
            r[c] = a + b;
            r[c + span] = a - b;
          }
    }
#pragma unroll
    for (uint32_t c = 0; c < 64; c++)
      x[t % 64][half * 64 + c] = key < tokens ? r[c] : 0.f;
  }
  __syncthreads();
  const float centre = mean[head * 128 + t];
  const uint32_t low = t % 64, valid = tokens - min(tokens, blockIdx.x * 64);
  uint32_t *target = reinterpret_cast<uint32_t *>(out + (size_t(head) * tiles + tile) * (FP4_TILE_LINES * 16));
  uint32_t scales = 0;
#pragma unroll
  for (uint32_t block = 0; block < 4; block++)
  {
    float a[16];
    float top = 0.f;
#pragma unroll
    for (uint32_t i = 0; i < 16; i++)
    {
      const uint32_t s = block * 16 + i, key = 32 * (s / 32) + 8 * ((s % 8) / 2) + 2 * ((s % 32) / 8) + s % 2;
      const float first = x[key][low], second = x[key][low + 64];
      a[i] = key < valid ? (t < 64 ? first + second : first - second) * 0.08838834764831845f - centre : 0.f;
      top = max(top, fabsf(a[i]));
    }
    float best = 3.4e38f;
    uint32_t words[2] = {0, 0}, byte = 0;
#pragma unroll
    for (uint32_t n = 0; n < ATTN_TRIES; n++)
    {
      const uint32_t bits = (__float_as_uint(min(max(top * shrinks[n] * (1.f / 6.f), 0.015625f), 448.f)) + 0xFFFFF) & 0xFFF00000;
      const float scale = __uint_as_float(bits), inverse = 1.f / scale;
      float scaled[16];
#pragma unroll
      for (uint32_t i = 0; i < 16; i++)
        scaled[i] = a[i] * inverse;
      const uint32_t tried[2] = {fp4_pack(scaled), fp4_pack(scaled + 8)};
      float error = 0.f;
#pragma unroll
      for (uint32_t i = 0; i < 16; i++)
      {
        const float miss = fp4_value(tried[i / 8] >> (4 * (i % 8))) - scaled[i];
        error += miss * miss;
      }
      error *= scale * scale;
      if (error < best)
      {
        best = error;
        words[0] = tried[0];
        words[1] = tried[1];
        byte = (bits >> 20) - 960;
      }
    }
    target[t * 8 + block * 2] = words[0];
    target[t * 8 + block * 2 + 1] = words[1];
    scales |= byte << (8 * block);
  }
  target[1024 + t] = scales;
}

// profiling only (ATTN_ABLATE 5, 7): the weights x values matmul at the 4-bit instruction's cost, half the
// instructions and half the value loads of the 8-bit one, on whatever bytes are in the registers.
template <uint32_t num_tiles_q, uint32_t num_tiles_k, uint32_t num_tiles_v, SwizzleMode swizzle_mode, uint32_t stride>
__device__ __forceinline__ void mock_fp4_sv(const smem_t<swizzle_mode, stride> &smem_V, uint32_t RS_f8[][num_tiles_k / 2][4], float RO[][num_tiles_v][8])
{
  uint32_t offset_V = smem_V.get_permuted_offset(get_lane_id() % 8 + (get_lane_id() / 16) * 8, (get_lane_id() / 8) % 2);
#pragma unroll
  for (uint32_t fv = 0; fv < num_tiles_v; fv++)
  {
    uint32_t RV[4];
    smem_V.ldmatrix_m8n8x4(offset_V, RV);
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++)
    {
      mma_fp4(RO[fq][fv], RS_f8[fq][0], RV, FP4_ONES, FP4_ONES);
      mma_fp4(RO[fq][fv] + 4, RS_f8[fq][0], RV + 2, FP4_ONES, FP4_ONES);
    }
    offset_V = smem_V.advance_offset_by_row<16>(offset_V);
  }
}

// profiling only (ATTN_ABLATE 6, 7): the q x k matmul at the 4-bit instruction's cost, two passes over the head
// dimension instead of four.
template <uint32_t num_tiles_q, uint32_t num_tiles_k, SwizzleMode swizzle_mode, uint32_t stride>
__device__ __forceinline__ void mock_fp4_qk(const smem_t<swizzle_mode, stride> &smem_Q, const smem_t<swizzle_mode, stride> &smem_K, float RS[][num_tiles_k][8], uint32_t offset_Q, uint32_t offset_K)
{
  uint32_t RQ[num_tiles_q][4];
  uint32_t RK[4];
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++)
#pragma unroll
    for (uint32_t fk = 0; fk < num_tiles_k; fk++)
#pragma unroll
      for (uint32_t k = 0; k < 8; k++)
        RS[fq][fk][k] = 0.f;
#pragma unroll
  for (uint32_t iter = 0; iter < 2; iter++)
  {
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++)
    {
      smem_Q.ldmatrix_m8n8x4(offset_Q, RQ[fq]);
      offset_Q = smem_Q.advance_offset_by_row<16>(offset_Q);
    }
    offset_Q = smem_Q.template advance_offset_by_column<2>(offset_Q - (num_tiles_q * 16 * stride), iter);
#pragma unroll
    for (uint32_t fk = 0; fk < num_tiles_k; fk++)
    {
      smem_K.ldmatrix_m8n8x4(offset_K, RK);
      offset_K = smem_K.advance_offset_by_row<16>(offset_K);
#pragma unroll
      for (uint32_t fq = 0; fq < num_tiles_q; fq++)
      {
        mma_fp4(RS[fq][fk], RQ[fq], RK, FP4_ONES, FP4_ONES);
        mma_fp4(RS[fq][fk] + 4, RQ[fq], RK + 2, FP4_ONES, FP4_ONES);
      }
    }
    offset_K = smem_K.template advance_offset_by_column<2>(offset_K - (num_tiles_k * 16 * stride), iter);
  }
}

// `update_mdo` for fp32 accumulators and fp8 weights (new maxima, rescale, exponentiate), with the rescale skipped
// for a warp none of whose rows saw a new maximum.
template <uint32_t num_tiles_q, uint32_t num_tiles_k, uint32_t num_tiles_v>
__device__ __forceinline__ void update_mdo_lazy(float RS[][num_tiles_k][8], float RO[][num_tiles_v][8], float m[][2], float d[][2], const float &sm_scale)
{
  float m_prev[num_tiles_q][2];
  bool moved = false;
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++)
  {
#pragma unroll
    for (uint32_t k = 0; k < 2; k++)
    {
      float m_temp = -5000000.0f;
#pragma unroll
      for (uint32_t fk = 0; fk < num_tiles_k; fk++)
      {
        m_temp = max(m_temp, max(max(RS[fq][fk][k * 2 + 0], RS[fq][fk][k * 2 + 1]), max(RS[fq][fk][k * 2 + 4], RS[fq][fk][k * 2 + 5])));
      }
      m_temp = fmaf(m_temp, sm_scale, -S_FP8_OFFSET);
      m_temp = max(m_temp, __shfl_xor_sync(0xffffffff, m_temp, 0x1));
      m_temp = max(m_temp, __shfl_xor_sync(0xffffffff, m_temp, 0x2));
      m_prev[fq][k] = m[fq][k];
      m[fq][k] = max(m[fq][k], m_temp);
      moved |= m[fq][k] != m_prev[fq][k];
    }
  }
  if (__any_sync(0xffffffff, moved))
  {
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++)
    {
#pragma unroll
      for (uint32_t k = 0; k < 2; k++)
      {
        float o_scale = math::ptx_exp2(m_prev[fq][k] - m[fq][k]);
        d[fq][k] *= o_scale;
#pragma unroll
        for (uint32_t fv = 0; fv < num_tiles_v; fv++)
        {
          RO[fq][fv][k * 2 + 0] *= o_scale;
          RO[fq][fv][k * 2 + 1] *= o_scale;
          RO[fq][fv][k * 2 + 4] *= o_scale;
          RO[fq][fv][k * 2 + 5] *= o_scale;
        }
      }
    }
  }
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++)
  {
#pragma unroll
    for (uint32_t k = 0; k < 2; k++)
    {
      float negative_m = -m[fq][k];
#pragma unroll
      for (uint32_t fk = 0; fk < num_tiles_k; fk++)
      {
        RS[fq][fk][k * 2 + 0] = math::ptx_exp2(fmaf(RS[fq][fk][k * 2 + 0], sm_scale, negative_m));
        RS[fq][fk][k * 2 + 1] = math::ptx_exp2(fmaf(RS[fq][fk][k * 2 + 1], sm_scale, negative_m));
        RS[fq][fk][k * 2 + 4] = math::ptx_exp2(fmaf(RS[fq][fk][k * 2 + 4], sm_scale, negative_m));
        RS[fq][fk][k * 2 + 5] = math::ptx_exp2(fmaf(RS[fq][fk][k * 2 + 5], sm_scale, negative_m));
      }
    }
  }
}

template <uint32_t num_tiles_q, uint32_t num_tiles_k, uint32_t num_tiles_v>
__device__ __forceinline__ void ATTN_UPDATE(float RS[][num_tiles_k][8], float RO[][num_tiles_v][8], float m[][2], float d[][2], const float &sm_scale)
{
#if ATTN_LAZY
  update_mdo_lazy<num_tiles_q, num_tiles_k, num_tiles_v>(RS, RO, m, d, sm_scale);
#else
  update_mdo<num_tiles_q, num_tiles_k, num_tiles_v, false, true, false>(RS, RO, m, d, sm_scale);
#endif
}

#define PACK_SIZE_QK 16 // as if it is int8
#define PACK_SIZE_V 16  // fp8
#define PACK_SIZE_O 8   // fp16

// treat as if int8 tensor core
#define MMA_QK_M 16
#define MMA_QK_N 16
#define MMA_QK_K 32

// fp8 tensor core
#define MMA_SV_M 16
#define MMA_SV_N 16
#define MMA_SV_K 32

template<uint32_t CTA_Q, uint32_t CTA_K, uint32_t WARP_Q, uint32_t WARP_K, uint32_t head_dim, DataType DTypeQK, QuantGranularity Q_GRAN, QuantGranularity K_GRAN,
        typename DTypeSVAccum = float, bool use_inst_buffer = false, typename DTypeOut = half, ComputeUnit DenominatorAccumUnit, MaskMode mask_mode = MaskMode::kNone, bool return_lse = false, bool fuse_v_scale=false, bool fuse_v_mean=false, bool use_pv_fp16_accu=false>
__global__ void attn_sm120_kernel(int8_t *__restrict__ Q, int8_t *__restrict__ K, int8_t *__restrict__ V, DTypeOut *__restrict__ O, float *__restrict__ Lse,
                      float *__restrict__ Q_scale, float *__restrict__ K_scale, float *__restrict__ V_scale, float *__restrict__ V_mean,
                      const uint32_t qo_len, const uint32_t kv_len, const uint32_t num_kv_groups,
                      const uint32_t stride_bz_q, const uint32_t stride_seq_q, const uint32_t stride_h_q, 
                      const uint32_t stride_bz_k, const uint32_t stride_seq_k, const uint32_t stride_h_k,
                      const uint32_t stride_bz_v, const uint32_t stride_h_v, const uint32_t stride_d_v,
                      const uint32_t stride_bz_o, const uint32_t stride_seq_o, const uint32_t stride_h_o,
                      float sm_scale)
{
  // compile time check
  static_assert(DTypeQK == DataType::kInt8 || DTypeQK == DataType::kInt4, "DTypeQK must be int8 or int4");
  static_assert(Q_GRAN == QuantGranularity::kPerBlock || Q_GRAN == QuantGranularity::kPerWarp || Q_GRAN == QuantGranularity::kPerThread, "Q_GRAN must be kPerBlock, kPerWarp or kPerThread");
  static_assert(K_GRAN == QuantGranularity::kPerBlock || K_GRAN == QuantGranularity::kPerWarp || K_GRAN == QuantGranularity::kPerThread, "K_GRAN must be kPerBlock, kPerWarp or kPerThread");
  static_assert(head_dim % 64 == 0, "head_dim must be a multiple of 64");
  static_assert(std::is_same<DTypeSVAccum, float>::value, "DTypeSVAccum must be float, half is WIP");
  static_assert(std::is_same<DTypeOut, half>::value || std::is_same<DTypeOut, nv_bfloat16>::value, "DTypeOut must be half or nv_bfloat16");
  static_assert(CTA_K % 64 == 0);
  static_assert(CTA_Q / CTA_K <= 2); // for efficient causal implementation

  constexpr uint32_t num_warps_q = CTA_Q / WARP_Q;
  constexpr uint32_t num_warps_k = CTA_K / WARP_K;
  constexpr uint32_t num_warps = num_warps_q * num_warps_k;
  constexpr uint32_t num_tiles_q = WARP_Q / MMA_QK_M;
  constexpr uint32_t num_tiles_k = WARP_K / MMA_QK_N;
  constexpr uint32_t num_tiles_qk_inner = (DTypeQK == DataType::kInt8) ? (head_dim / MMA_QK_K) : (head_dim / 2 / MMA_QK_K);
  constexpr uint32_t num_tiles_v = head_dim / MMA_SV_N;

  constexpr uint32_t QK_SMEM_STRIDE = (DTypeQK == DataType::kInt8) ? (head_dim) : (head_dim / 2);
  constexpr uint32_t O_SMEM_STRIDE = head_dim;
  //                       for fp16: head_dim
  constexpr uint32_t V_SMEM_STRIDE = CTA_K;

  extern __shared__ int8_t smem[];

  const uint32_t lane_id = get_lane_id();
  const uint32_t warp_id = get_warp_id();

  // maximize L2 hit rate
  const uint32_t batch_id = blockIdx.z;
  const uint32_t bx = blockIdx.x;
  const uint32_t num_qo_heads = gridDim.y;
  const uint32_t head_id = blockIdx.y;

  // transfer to base 2 instead of base e with better numerical efficiency
  sm_scale *= math::log2e;

  // RS holds the fragment of S
  int32_t RS[num_tiles_q][num_tiles_k][8];
  DTypeSVAccum RO[num_tiles_q][num_tiles_v][8];
  float m[num_tiles_q][2]; // max
  float d[num_tiles_q][2]; // denominator

  uint32_t q_scale_idx, k_scale_idx;

  if constexpr (Q_GRAN == QuantGranularity::kPerBlock)
  {
    const uint32_t num_block_q = gridDim.x;
    q_scale_idx = batch_id * num_qo_heads * num_block_q + head_id * num_block_q + bx;
  }
  else if constexpr (Q_GRAN == QuantGranularity::kPerWarp)
  {
    const uint32_t num_warp_block_q = gridDim.x * num_warps_q;
    q_scale_idx = batch_id * num_qo_heads * num_warp_block_q + head_id * num_warp_block_q + bx * num_warps_q + get_warp_idx_q<num_warps_q, num_warps_k>();
  }
  else if constexpr (Q_GRAN == QuantGranularity::kPerThread)
  {
    const uint32_t num_warp_block_q = gridDim.x * num_warps_q;
    q_scale_idx = batch_id * num_qo_heads * (num_warp_block_q * 8) + head_id * (num_warp_block_q * 8) + bx * (num_warps_q * 8) + get_warp_idx_q<num_warps_q, num_warps_k>() * 8 + lane_id / 4;
  }

  if constexpr (K_GRAN == QuantGranularity::kPerBlock)
  {
    const uint32_t num_block_k = div_ceil(kv_len, CTA_K);
    k_scale_idx = batch_id * (num_qo_heads / num_kv_groups) * num_block_k + (head_id / num_kv_groups) * num_block_k;
  }
  else if constexpr (K_GRAN == QuantGranularity::kPerWarp)
  {
    const uint32_t num_warp_block_k = div_ceil(kv_len, CTA_K) * (CTA_K / WARP_K);
    k_scale_idx = batch_id * (num_qo_heads / num_kv_groups) * num_warp_block_k + (head_id / num_kv_groups) * num_warp_block_k + get_warp_idx_k<num_warps_q, num_warps_k>();
  }
  else if constexpr (K_GRAN == QuantGranularity::kPerThread)
  {
    const uint32_t num_warp_block_k = div_ceil(kv_len, CTA_K) * (CTA_K / WARP_K);
    k_scale_idx = batch_id * (num_qo_heads / num_kv_groups) * (num_warp_block_k * 4) + (head_id / num_kv_groups) * (num_warp_block_k * 4) + get_warp_idx_k<num_warps_q, num_warps_k>() * 4 + lane_id % 4;
  }

  constexpr uint32_t k_scale_advance_offset = (K_GRAN == QuantGranularity::kPerBlock) ? 1 : (K_GRAN == QuantGranularity::kPerWarp) ? (CTA_K / WARP_K) : (CTA_K / WARP_K) * 4;

  // initialize o, m, d
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++)
  {
#pragma unroll
    for (uint32_t fv = 0; fv < num_tiles_v; fv++)
    {
      if constexpr (std::is_same<DTypeSVAccum, float>::value)
      {
#pragma unroll
        for (uint32_t k = 0; k < 8; k++)
        {        
          RO[fq][fv][k] = 0.0f;
        }
      }
      else if constexpr (std::is_same<DTypeSVAccum, half>::value)
      {
#pragma unroll
        for (uint32_t k = 0; k < 4; k++)
        {
          ((int32_t*)RO[fq][fv])[k] = 0;
        }
      }
    }
  }
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++)
  {
#pragma unroll
    for (uint32_t k = 0; k < 2; k++)
    {
      m[fq][k] = -5000000.0f;
      d[fq][k] = 1.0f;
    }
  }

  constexpr uint32_t K_smem_idx_offset = CTA_Q;
  constexpr uint32_t V_smem_idx_offset = CTA_Q + CTA_K;

  constexpr SwizzleMode swizzle_mode_QK = (QK_SMEM_STRIDE == 32) ? SwizzleMode::k32B : (QK_SMEM_STRIDE == 64) ? SwizzleMode::k64B : SwizzleMode::k128B;
  smem_t<swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK> smem_Q(smem);
  smem_t<swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK> smem_K(smem + K_smem_idx_offset * QK_SMEM_STRIDE);
  //                                             for fp16: 32
  constexpr SwizzleMode swizzle_mode_V = (V_SMEM_STRIDE == 64) ? SwizzleMode::k64B : SwizzleMode::k128B;
  smem_t<swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V> smem_V(smem + V_smem_idx_offset * QK_SMEM_STRIDE);
  constexpr SwizzleMode swizzle_mode_O = (O_SMEM_STRIDE == 32) ? SwizzleMode::k64B : SwizzleMode::k128B;
  smem_t<swizzle_mode_O, O_SMEM_STRIDE / PACK_SIZE_O> smem_O(smem);

  constexpr uint32_t global_to_shared_line_lanes_QK = (QK_SMEM_STRIDE == 32) ? 2 : (QK_SMEM_STRIDE == 64) ? 4 : 8;
  constexpr uint32_t global_to_shared_copy_lines_per_warp_QK = (QK_SMEM_STRIDE == 32) ? 16 : (QK_SMEM_STRIDE == 64) ? 8 : 4;
  //                                                         for fp16: 32
  constexpr uint32_t global_to_shared_line_lanes_V = (V_SMEM_STRIDE == 64) ? 4 : 8;
  //                                                                  for fp16: 32
  constexpr uint32_t global_to_shared_copy_lines_per_warp_V = (V_SMEM_STRIDE == 64) ? 8 : 4;
  constexpr uint32_t global_to_shared_line_lanes_O = (O_SMEM_STRIDE == 32) ? 4 : 8;
  constexpr uint32_t global_to_shared_copy_lines_per_warp_O = (O_SMEM_STRIDE == 32) ? 8 : 4;

  constexpr uint32_t QK_smem_iters_row = QK_SMEM_STRIDE / (global_to_shared_line_lanes_QK * PACK_SIZE_QK);
  constexpr uint32_t Q_smem_iters_col = CTA_Q / (num_warps * global_to_shared_copy_lines_per_warp_QK);
  constexpr uint32_t K_smem_iters_col = CTA_K / (num_warps * global_to_shared_copy_lines_per_warp_QK);
  constexpr uint32_t V_smem_iters_row = V_SMEM_STRIDE / (global_to_shared_line_lanes_V * PACK_SIZE_V);
  //                          for fp16: CTA_K
  constexpr uint32_t V_smem_iters_col = head_dim / (num_warps * global_to_shared_copy_lines_per_warp_V);
  constexpr uint32_t O_smem_iters_row = O_SMEM_STRIDE / (global_to_shared_line_lanes_O * PACK_SIZE_O);
  constexpr uint32_t O_smem_iters_col = CTA_Q / (num_warps * global_to_shared_copy_lines_per_warp_O);

  int8_t *Q_lane_base_ptr = Q + batch_id * stride_bz_q + head_id * stride_h_q + (bx * CTA_Q + CTA_Q / num_warps * warp_id + lane_id / global_to_shared_line_lanes_QK) * stride_seq_q + (lane_id % global_to_shared_line_lanes_QK) * PACK_SIZE_QK;
  int8_t *K_lane_base_ptr = K + batch_id * stride_bz_k + (head_id / num_kv_groups) * stride_h_k + (CTA_K / num_warps * warp_id + lane_id / global_to_shared_line_lanes_QK) * stride_seq_k + (lane_id % global_to_shared_line_lanes_QK) * PACK_SIZE_QK;
  //                                                                for fp16: CTA_K / num_warps * warp_id * stride_seq_v + lane_id / global_to_shared_line_lanes_V * stride_seq_v
  int8_t *V_lane_base_ptr = V + batch_id * stride_bz_v + (head_id / num_kv_groups) * stride_h_v + head_dim / num_warps * warp_id * stride_d_v + lane_id / global_to_shared_line_lanes_V * stride_d_v + (lane_id % global_to_shared_line_lanes_V) * PACK_SIZE_V;
  uint32_t Q_smem_offset_load = smem_Q.get_permuted_offset(warp_id * global_to_shared_copy_lines_per_warp_QK * Q_smem_iters_col + lane_id / global_to_shared_line_lanes_QK, lane_id % global_to_shared_line_lanes_QK);
  uint32_t K_smem_offset_load = smem_K.get_permuted_offset(warp_id * global_to_shared_copy_lines_per_warp_QK * K_smem_iters_col + lane_id / global_to_shared_line_lanes_QK, lane_id % global_to_shared_line_lanes_QK);
  uint32_t V_smem_offset_load = smem_V.get_permuted_offset(warp_id * global_to_shared_copy_lines_per_warp_V * V_smem_iters_col + lane_id / global_to_shared_line_lanes_V, lane_id % global_to_shared_line_lanes_V);

  uint32_t Q_smem_offset_mma = smem_Q.get_permuted_offset(get_warp_idx_q<num_warps_q, num_warps_k>() * WARP_Q + lane_id % 16, lane_id / 16);
  uint32_t K_smem_offset_mma = smem_K.get_permuted_offset(get_warp_idx_k<num_warps_q, num_warps_k>() * WARP_K + lane_id % 8 + (lane_id / 16) * 8, (lane_id / 8) % 2);
  // for fp 16:
  // uint32_t V_smem_offset_mma = smem_V.get_permuted_offset(get_warp_idx_k<num_warps_q, num_warps_k>() * WARP_K + lane_id % 16, lane_id / 16);
  b128_t *smem_V4 = reinterpret_cast<b128_t *>(smem + V_smem_idx_offset * QK_SMEM_STRIDE);
  const int8_t *V_tile = V + batch_id * stride_bz_v + (head_id / num_kv_groups) * stride_h_v; // nvfp4 values: (batch, heads, key tiles, 4608 bytes)
  uint32_t V_smem_offset_mma = smem_V.get_permuted_offset(lane_id % 8 + (lane_id / 16) * 8, get_warp_idx_k<num_warps_q, num_warps_k>() * WARP_K / PACK_SIZE_V + (lane_id / 8) % 2);

  // for causal masking
  uint32_t Q_idx_lane_base = bx * CTA_Q + get_warp_idx_q<num_warps_q, num_warps_k>() * WARP_Q + lane_id / 4;
  uint32_t K_idx_lane_base = get_warp_idx_k<num_warps_q, num_warps_k>() * WARP_K + 2 * (lane_id % 4);

  // for loading
  uint32_t Q_load_idx_lane_base = bx * CTA_Q + CTA_Q / num_warps * warp_id + lane_id / global_to_shared_line_lanes_QK;
  uint32_t K_load_idx_lane_base = CTA_K / num_warps * warp_id + lane_id / global_to_shared_line_lanes_QK;

  const uint32_t num_iterations = div_ceil(
      mask_mode == MaskMode::kCausal
          ? min(kv_len, (bx + 1) * CTA_Q)
          : kv_len,
      CTA_K);

  // load Q with predicate
  load_global_to_share<global_to_shared_line_lanes_QK, global_to_shared_copy_lines_per_warp_QK, QK_smem_iters_row, Q_smem_iters_col, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, CTA_Q>(
    &Q_lane_base_ptr, Q_smem_offset_load, stride_seq_q, smem_Q, Q_load_idx_lane_base, qo_len);
  cp_async::commit_group();
  cp_async::wait_group<0>();
  __syncthreads();

  // for num_tiles_qk_inner = 1, we load all Qs in register
  uint32_t RQ[num_tiles_q][4];
  if constexpr (num_tiles_qk_inner == 1)
  {
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++)
    {
      smem_Q.ldmatrix_m8n8x4(Q_smem_offset_mma, RQ[fq]);
      Q_smem_offset_mma = smem_Q.advance_offset_by_row<16>(Q_smem_offset_mma);
    }
  }

  // load K with predicate
  load_global_to_share<global_to_shared_line_lanes_QK, global_to_shared_copy_lines_per_warp_QK, QK_smem_iters_row, K_smem_iters_col, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, CTA_K>(
    &K_lane_base_ptr, K_smem_offset_load, stride_seq_k, smem_K, K_load_idx_lane_base, kv_len);
  cp_async::commit_group();

  float q_scale = Q_scale[q_scale_idx];

  float original_sm_scale = sm_scale;
  float dequant_scale = q_scale * K_scale[k_scale_idx + 0 * k_scale_advance_offset];

  sm_scale = original_sm_scale * dequant_scale;

  // load V
  // ! we assume that V is padded. If not, there might be illegal memory access or nan issue.
  // for fp16: 
  // load_global_to_share                stride_seq_v
#if ATTN_PV4
    load_fp4_values(V_tile, smem_V4, num_warps * 32);
    V_tile += FP4_TILE_LINES * 16;
#else
  load_fp8_V_global_to_share<global_to_shared_line_lanes_V, global_to_shared_copy_lines_per_warp_V, V_smem_iters_row, V_smem_iters_col, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V, CTA_K>(
    &V_lane_base_ptr, V_smem_offset_load, stride_d_v, smem_V);
#endif
  cp_async::commit_group();

  K_load_idx_lane_base += CTA_K;

#pragma unroll
  for (uint32_t iter = 1; iter < num_iterations - 1; iter++)
  {
    // ensure K is ready
    cp_async::wait_group<1>();
    __syncthreads();

#if ATTN_ABLATE == 6 || ATTN_ABLATE == 7
    float RS_f32[num_tiles_q][num_tiles_k][8];
    mock_fp4_qk<num_tiles_q, num_tiles_k, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK>(smem_Q, smem_K, RS_f32, Q_smem_offset_mma, K_smem_offset_mma);
#else
#if ATTN_ABLATE != 1
    // compute QK^T
    if constexpr (num_tiles_qk_inner == 1)
    {
      compute_int_qk<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_qk_inner, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, DTypeQK>(
        smem_K, RS, RQ, K_smem_offset_mma);
    }
    else
    {
      compute_int_qk<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_qk_inner, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, DTypeQK>(
        smem_Q, smem_K, RS, Q_smem_offset_mma, K_smem_offset_mma);
    }
#endif
    float RS_f32[num_tiles_q][num_tiles_k][8];

#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++)
    {
#pragma unroll
      for (uint32_t fk = 0; fk < num_tiles_k; fk++)
      {
#pragma unroll
        for (uint32_t k = 0; k < 8; k++)
        {
          RS_f32[fq][fk][k] = __int2float_rz(RS[fq][fk][k]);
        }
      }
    }

#endif
    K_idx_lane_base += CTA_K;

#if ATTN_ABLATE != 2
    if constexpr (std::is_same<DTypeSVAccum, float>::value)
    {
      ATTN_UPDATE<num_tiles_q, num_tiles_k, num_tiles_v>(RS_f32, RO, m, d, sm_scale);
    }
#endif

    if constexpr (DenominatorAccumUnit == ComputeUnit::kCudaCore)
    {
#if !ATTN_PV4
      accumulate_d<num_tiles_q, num_tiles_k, ComputeUnit::kCudaCore>(RS_f32, d);
#endif
    }

#if ATTN_PV4
    uint32_t RS_f4[num_tiles_q][4], RS_sf[num_tiles_q];
    weights_to_fp4<num_tiles_q, num_tiles_k>(RS_f32, RS_f4, RS_sf, d);
    uint32_t RS_f8[num_tiles_q][num_tiles_k / 2][4];
#else
    uint32_t RS_f8[num_tiles_q][num_tiles_k / 2][4];
    RS_32_to_8<num_tiles_q, num_tiles_k>(RS_f32, RS_f8);
#endif

    if constexpr (DenominatorAccumUnit == ComputeUnit::kTensorCore)
    {
      accumulate_d_f8<num_tiles_q, num_tiles_k>(RS_f8, d);
    }

    __syncthreads();

#if ATTN_ABLATE != 4
    // load K without predicate
    load_global_to_share<global_to_shared_line_lanes_QK, global_to_shared_copy_lines_per_warp_QK, QK_smem_iters_row, K_smem_iters_col, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, CTA_K>(
      &K_lane_base_ptr, K_smem_offset_load, stride_seq_k, smem_K);
    cp_async::commit_group();
#endif

    dequant_scale = q_scale * K_scale[k_scale_idx + iter * k_scale_advance_offset];
    sm_scale = original_sm_scale * dequant_scale;
    
    // ensure V is ready
    cp_async::wait_group<1>();
    __syncthreads();

    // for fp16:
    // compute_fp16_sv_permuted<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_v, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V, 4>(
    //   smem_V, RS_f16, RO, d, V_smem_offset_mma);
#if ATTN_ABLATE == 5 || ATTN_ABLATE == 7
    mock_fp4_sv<num_tiles_q, num_tiles_k, num_tiles_v, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V>(smem_V, RS_f8, RO);
#elif ATTN_ABLATE != 3
#if ATTN_PV4
    values_fp4<num_tiles_q, num_tiles_v>(smem_V4, RS_f4, RS_sf, RO);
#else
    if constexpr (!use_inst_buffer)
    {
      compute_fp8_sv<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_v, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V>(
        smem_V, RS_f8, RO, d);
    }
    else
    {
      if constexpr (!use_pv_fp16_accu){
        compute_fp8_sv_inst_buf<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_v, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V>(
          smem_V, RS_f8, RO, d);   
      }
      else{
        compute_fp8_sv_inst_buf_fp16_accu<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_v, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V>(
          smem_V, RS_f8, RO, d);   
      }
    }
#endif
#endif
    __syncthreads();
#if ATTN_ABLATE != 4
    // load V
    // for fp16: 
    // load_global_to_share                stride_seq_v
#if ATTN_PV4
    load_fp4_values(V_tile, smem_V4, num_warps * 32);
    V_tile += FP4_TILE_LINES * 16;
#else
    load_fp8_V_global_to_share<global_to_shared_line_lanes_V, global_to_shared_copy_lines_per_warp_V, V_smem_iters_row, V_smem_iters_col, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V, CTA_K>(
      &V_lane_base_ptr, V_smem_offset_load, stride_d_v, smem_V);
#endif
    cp_async::commit_group();
#endif
  
    K_load_idx_lane_base += CTA_K;
  }

  // second last iter, apply causal mask
  if (num_iterations > 1)
  {
    // ensure K is ready
    cp_async::wait_group<1>();
    __syncthreads();

    // compute QK^T
    if constexpr (num_tiles_qk_inner == 1)
    {
      compute_int_qk<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_qk_inner, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, DTypeQK>(
        smem_K, RS, RQ, K_smem_offset_mma);
    }
    else
    {
      compute_int_qk<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_qk_inner, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, DTypeQK>(
        smem_Q, smem_K, RS, Q_smem_offset_mma, K_smem_offset_mma);
    }

    float RS_f32[num_tiles_q][num_tiles_k][8];

#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++)
    {
#pragma unroll
      for (uint32_t fk = 0; fk < num_tiles_k; fk++)
      {
#pragma unroll
        for (uint32_t k = 0; k < 8; k++)
        {
          RS_f32[fq][fk][k] = __int2float_rz(RS[fq][fk][k]) * dequant_scale;
        }
      }
    }

    if constexpr (mask_mode == MaskMode::kCausal)
    {
      apply_causal_mask<num_tiles_q, num_tiles_k>(Q_idx_lane_base, K_idx_lane_base, RS_f32);
    }
    // apply_out_of_bound_mask<num_tiles_q, num_tiles_k>(K_idx_lane_base, RS_f32, kv_len);
    K_idx_lane_base += CTA_K;

    if constexpr (std::is_same<DTypeSVAccum, float>::value)
    {
      ATTN_UPDATE<num_tiles_q, num_tiles_k, num_tiles_v>(RS_f32, RO, m, d, original_sm_scale);
    }
    else if constexpr (std::is_same<DTypeSVAccum, half>::value)
    {
      update_mdo<num_tiles_q, num_tiles_k, num_tiles_v, true, true, false>(RS_f32, RO, m, d, original_sm_scale);
    }

    if constexpr (DenominatorAccumUnit == ComputeUnit::kCudaCore)
    {
#if !ATTN_PV4
      accumulate_d<num_tiles_q, num_tiles_k, ComputeUnit::kCudaCore>(RS_f32, d);
#endif
    }

#if ATTN_PV4
    uint32_t RS_f4[num_tiles_q][4], RS_sf[num_tiles_q];
    weights_to_fp4<num_tiles_q, num_tiles_k>(RS_f32, RS_f4, RS_sf, d);
    uint32_t RS_f8[num_tiles_q][num_tiles_k / 2][4];
#else
    uint32_t RS_f8[num_tiles_q][num_tiles_k / 2][4];
    RS_32_to_8<num_tiles_q, num_tiles_k>(RS_f32, RS_f8);
#endif

    if constexpr (DenominatorAccumUnit == ComputeUnit::kTensorCore)
    {
      accumulate_d_f8<num_tiles_q, num_tiles_k>(RS_f8, d);
    }

    __syncthreads();

    // load K with predicate
    load_global_to_share<global_to_shared_line_lanes_QK, global_to_shared_copy_lines_per_warp_QK, QK_smem_iters_row, K_smem_iters_col, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, CTA_K>(
      &K_lane_base_ptr, K_smem_offset_load, stride_seq_k, smem_K, K_load_idx_lane_base, kv_len);
    cp_async::commit_group();

    dequant_scale = q_scale * K_scale[k_scale_idx + (num_iterations - 1) * k_scale_advance_offset];
    sm_scale = original_sm_scale * dequant_scale;

    // ensure V is ready
    cp_async::wait_group<1>();
    __syncthreads();

    // for fp16:
    // compute_fp16_sv_permuted<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_v, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V, 4>(
    //   smem_V, RS_f16, RO, d, V_smem_offset_mma);
#if ATTN_PV4
    values_fp4<num_tiles_q, num_tiles_v>(smem_V4, RS_f4, RS_sf, RO);
#else
    if constexpr (!use_inst_buffer)
    {
      compute_fp8_sv<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_v, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V>(
        smem_V, RS_f8, RO, d);
    }
    else
    {
      if constexpr (!use_pv_fp16_accu){
        compute_fp8_sv_inst_buf<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_v, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V>(
          smem_V, RS_f8, RO, d);
      }
      else{
        compute_fp8_sv_inst_buf_fp16_accu<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_v, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V>(
          smem_V, RS_f8, RO, d);   
      }
    }
#endif

    __syncthreads();
    // load V
    // for fp16: 
    // load_global_to_share                stride_seq_v
#if ATTN_PV4
    load_fp4_values(V_tile, smem_V4, num_warps * 32);
    V_tile += FP4_TILE_LINES * 16;
#else
    load_fp8_V_global_to_share<global_to_shared_line_lanes_V, global_to_shared_copy_lines_per_warp_V, V_smem_iters_row, V_smem_iters_col, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V, CTA_K>(
      &V_lane_base_ptr, V_smem_offset_load, stride_d_v, smem_V);
#endif
    cp_async::commit_group();
    K_load_idx_lane_base += CTA_K;
  }

  // last iter, apply causal mask and out of bound mask
  {
    // ensure K is ready
    cp_async::wait_group<1>();
    __syncthreads();

    // compute QK^T
    if constexpr (num_tiles_qk_inner == 1)
    {
      compute_int_qk<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_qk_inner, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, DTypeQK>(
        smem_K, RS, RQ, K_smem_offset_mma);
    }
    else
    {
      compute_int_qk<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_qk_inner, swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK, DTypeQK>(
        smem_Q, smem_K, RS, Q_smem_offset_mma, K_smem_offset_mma);
    }

    float RS_f32[num_tiles_q][num_tiles_k][8];

#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++)
    {
#pragma unroll
      for (uint32_t fk = 0; fk < num_tiles_k; fk++)
      {
#pragma unroll
        for (uint32_t k = 0; k < 8; k++)
        {
          RS_f32[fq][fk][k] = __int2float_rz(RS[fq][fk][k]) * dequant_scale;
        }
      }
    }

    if constexpr (mask_mode == MaskMode::kCausal)
    {
      apply_causal_mask<num_tiles_q, num_tiles_k>(Q_idx_lane_base, K_idx_lane_base, RS_f32);
    }
    apply_out_of_bound_mask<num_tiles_q, num_tiles_k>(K_idx_lane_base, RS_f32, kv_len);
    K_idx_lane_base += CTA_K;

    if constexpr (std::is_same<DTypeSVAccum, float>::value)
    {
      ATTN_UPDATE<num_tiles_q, num_tiles_k, num_tiles_v>(RS_f32, RO, m, d, original_sm_scale);
    }
    else if constexpr (std::is_same<DTypeSVAccum, half>::value)
    {
      update_mdo<num_tiles_q, num_tiles_k, num_tiles_v, true, true, false>(RS_f32, RO, m, d, original_sm_scale);
    }

    if constexpr (DenominatorAccumUnit == ComputeUnit::kCudaCore)
    {
#if !ATTN_PV4
      accumulate_d<num_tiles_q, num_tiles_k, ComputeUnit::kCudaCore>(RS_f32, d);
#endif
    }

#if ATTN_PV4
    uint32_t RS_f4[num_tiles_q][4], RS_sf[num_tiles_q];
    weights_to_fp4<num_tiles_q, num_tiles_k>(RS_f32, RS_f4, RS_sf, d);
    uint32_t RS_f8[num_tiles_q][num_tiles_k / 2][4];
#else
    uint32_t RS_f8[num_tiles_q][num_tiles_k / 2][4];
    RS_32_to_8<num_tiles_q, num_tiles_k>(RS_f32, RS_f8);
#endif

    if constexpr (DenominatorAccumUnit == ComputeUnit::kTensorCore)
    {
      accumulate_d_f8<num_tiles_q, num_tiles_k>(RS_f8, d);
    }

    // ensure V is ready
    cp_async::wait_group<0>();
    __syncthreads();

    // for fp16:
    // compute_fp16_sv_permuted<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_v, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V, 4>(
    //   smem_V, RS_f16, RO, d, V_smem_offset_mma);
#if ATTN_PV4
    values_fp4<num_tiles_q, num_tiles_v>(smem_V4, RS_f4, RS_sf, RO);
#else
    if constexpr (!use_inst_buffer)
    {
      compute_fp8_sv<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_v, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V>(
        smem_V, RS_f8, RO, d);
    }
    else
    {
      if constexpr (!use_pv_fp16_accu){
        compute_fp8_sv_inst_buf<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_v, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V>(
          smem_V, RS_f8, RO, d);
      }
      else{
        compute_fp8_sv_inst_buf_fp16_accu<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k, num_tiles_v, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V>(
          smem_V, RS_f8, RO, d);   
      }
    }
#endif

    __syncthreads();

  }

  // TODO: thread block sync mdo state for num_warps_k > 0. Then only one thread block needs to do the final saving.

  normalize_d<num_tiles_q, num_tiles_v, ComputeUnit::kCudaCore>(RO, m, d);

  // ! here we just implement the case for fp32 acumulation
  if constexpr (fuse_v_scale)
  {
    float v_scale[4];
    float *V_scale_base_ptr = V_scale + batch_id * (num_qo_heads / num_kv_groups) * head_dim + (head_id / num_kv_groups) * head_dim + (lane_id % 4 ) * 2;
#pragma unroll
    for (uint32_t fv = 0; fv < num_tiles_v; fv++)
    {
      ((float2*)v_scale)[0] = *((float2*)(V_scale_base_ptr + fv * 16));
      ((float2*)v_scale)[1] = *((float2*)(V_scale_base_ptr + fv * 16 + 8));
#pragma unroll
      for (uint32_t fq = 0; fq < num_tiles_q; fq++)
      {
        RO[fq][fv][0] *= v_scale[0];
        RO[fq][fv][1] *= v_scale[1];
        RO[fq][fv][2] *= v_scale[0];
        RO[fq][fv][3] *= v_scale[1];
        RO[fq][fv][4] *= v_scale[2];
        RO[fq][fv][5] *= v_scale[3];
        RO[fq][fv][6] *= v_scale[2];
        RO[fq][fv][7] *= v_scale[3];
      }
    }
  }

  if constexpr (fuse_v_mean)
  {
    float v_mean[4];
    float *V_mean_base_ptr = V_mean + batch_id * (num_qo_heads / num_kv_groups) * head_dim + (head_id / num_kv_groups) * head_dim + (lane_id % 4 ) * 2;
#pragma unroll
    for (uint32_t fv = 0; fv < num_tiles_v; fv++)
    {
      ((float2*)v_mean)[0] = *((float2*)(V_mean_base_ptr + fv * 16));
      ((float2*)v_mean)[1] = *((float2*)(V_mean_base_ptr + fv * 16 + 8));
#pragma unroll
      for (uint32_t fq = 0; fq < num_tiles_q; fq++)
      {
        RO[fq][fv][0] += v_mean[0];
        RO[fq][fv][1] += v_mean[1];
        RO[fq][fv][2] += v_mean[0];
        RO[fq][fv][3] += v_mean[1];
        RO[fq][fv][4] += v_mean[2];
        RO[fq][fv][5] += v_mean[3];
        RO[fq][fv][6] += v_mean[2];
        RO[fq][fv][7] += v_mean[3];
      }
    }
  }

#if ATTN_PV4
  turn_back<num_tiles_q, num_tiles_v>(RO);
#endif
  // save the result to shared memory
  uint32_t smem_O_row_base = get_warp_idx_q<num_warps_q, num_warps_k>() * WARP_Q + lane_id / 4;
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++)
  {
#pragma unroll
    for (uint32_t fv = 0; fv < num_tiles_v; fv++)
    {
      uint32_t offset_O = smem_O.get_permuted_offset(smem_O_row_base + fq * MMA_QK_M, fv * (MMA_SV_N / PACK_SIZE_O));

      if constexpr (std::is_same<DTypeSVAccum, float>::value)
      {
        // convert RO to half
        uint32_t RO_f16[4];
#pragma unroll
        for (uint32_t k = 0; k < 4; k++)
        {
          if constexpr (std::is_same<DTypeOut, half>::value)
          {
            ((half2*)RO_f16)[k] = __float22half2_rn(((float2*)RO[fq][fv])[k]);
          }
          else
          {
            ((nv_bfloat162*)RO_f16)[k] = __float22bfloat162_rn(((float2*)RO[fq][fv])[k]);
          }
        }

        ((uint32_t*)(smem_O.base + offset_O))[lane_id % 4] = RO_f16[0];
        ((uint32_t*)(smem_O.base + offset_O + 8 * (O_SMEM_STRIDE / PACK_SIZE_O)))[lane_id % 4] = RO_f16[1];

        offset_O = smem_O.get_permuted_offset(smem_O_row_base + fq * MMA_QK_M, fv * (MMA_SV_N / PACK_SIZE_O) + 1);
        ((uint32_t*)(smem_O.base + offset_O))[lane_id % 4] = RO_f16[2];
        ((uint32_t*)(smem_O.base + offset_O + 8 * (O_SMEM_STRIDE / PACK_SIZE_O)))[lane_id % 4] = RO_f16[3];
      }
      else if constexpr (std::is_same<DTypeSVAccum, half>::value)
      { 
        // TODO: not implement
      }
    }
  }

  // ! do we need to sync here?
  __syncwarp();

  // shared memory to global memory
  DTypeOut *O_lane_ptr = O + batch_id * stride_bz_o + head_id * stride_h_o + (bx * CTA_Q + WARP_Q * get_warp_idx_q<num_warps_q, num_warps_k>() + lane_id / global_to_shared_line_lanes_O) * stride_seq_o + lane_id % global_to_shared_line_lanes_O * PACK_SIZE_O;
  uint32_t offset_O = smem_O.get_permuted_offset(get_warp_idx_q<num_warps_q, num_warps_k>() * WARP_Q + lane_id / global_to_shared_line_lanes_O, lane_id % global_to_shared_line_lanes_O);
  uint32_t O_load_idx_lane_base = bx * CTA_Q + CTA_Q / num_warps * warp_id + lane_id / global_to_shared_line_lanes_O;

#pragma unroll
  for (uint32_t i = 0; i < O_smem_iters_col; i++)
  {
#pragma unroll
    for (uint32_t j = 0; j < O_smem_iters_row; j++)
    {
      if (O_load_idx_lane_base < qo_len)
      {
        smem_O.store_128b(offset_O, O_lane_ptr);
      }
      O_lane_ptr += (global_to_shared_line_lanes_O * PACK_SIZE_O);
      offset_O = smem_O.advance_offset_by_column<global_to_shared_line_lanes_O>(offset_O);
    }

    offset_O = smem_O.advance_offset_by_row<global_to_shared_copy_lines_per_warp_O>(offset_O - (O_smem_iters_row * global_to_shared_line_lanes_O));
    O_lane_ptr += ((global_to_shared_copy_lines_per_warp_O * stride_seq_o) - (O_smem_iters_row * global_to_shared_line_lanes_O * PACK_SIZE_O));
    O_load_idx_lane_base += global_to_shared_copy_lines_per_warp_O;
  }

  if constexpr (return_lse)
  { 
    // ! this only works for num_tiles_q = 2
    uint32_t lse_idx = bx * CTA_Q + lane_id / 4 + 8 * (lane_id % 4) + WARP_Q * get_warp_idx_q<num_warps_q, num_warps_k>();
    float *lse_lane_ptr = Lse + batch_id * (qo_len * num_qo_heads) + head_id * qo_len + lse_idx;
    uint32_t fq = (lane_id % 4) / 2;
    uint32_t k = (lane_id % 4) % 2;

    if (lse_idx < qo_len)
    {
      lse_lane_ptr[0] = (math::ptx_log2(d[fq][k]) + m[fq][k]);
    }
  }
}






