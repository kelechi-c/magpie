// what the card's tensor cores do when nothing else is in the way: every warp multiplies operands that already sit in
// registers, with no loads, no stores and no scheduling. the rate of a real matmul kernel can only be below this, so
// it is the ceiling to hold the gemm and attention kernels against (not the datasheet figure).
// then "streamed ...": see `streamed`.
//   nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a native/peak.cu -o /tmp/peak && /tmp/peak
// prints "peak <format> warps=<per block> tiles=<independent accumulators per thread> <tflop/s>", then "fed <format>
// warp=<rows>x<columns> warps=<per block> <tflop/s>": the same instruction when every operand fragment is first read from
// shared memory with ldmatrix, as the inner loop of a tiled matmul does (a warp tile of M x N outputs reads M/16 + N/8
// fragments for M/16 * N/8 instructions), still with no global loads and no stores.
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

constexpr uint32_t ITERS = 20000;
constexpr uint16_t ZERO = 0;

// 16 x 8 output += 16 x K times K x 8, one warp. K is 16 (bf16), 32 (fp8), 64 (fp4).
__device__ __forceinline__ void mma_bf16(float *C, const uint32_t *A, const uint32_t *B)
{
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
               : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
               : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));
}

__device__ __forceinline__ void mma_fp8(float *C, const uint32_t *A, const uint32_t *B)
{
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
               : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
               : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));
}

// the int8 form sageattention's q x k uses; the accumulators are integers, passed here as raw bits.
__device__ __forceinline__ void mma_s8(float *C, const uint32_t *A, const uint32_t *B)
{
  int32_t *c = reinterpret_cast<int32_t *>(C);
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
               : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
               : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
}

// the block-scaled forms cutlass's sm_120 kernels use: one power-of-two scale per 32 values (mxfp8), one e4m3 scale per 16 (nvfp4).
__device__ __forceinline__ void mma_mxfp8(float *C, const uint32_t *A, const uint32_t *B)
{
  const uint32_t one = 0x7f7f7f7f;
  asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
               "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
               : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
               : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]),
                 "r"(one), "h"(ZERO), "h"(ZERO), "r"(one), "h"(ZERO), "h"(ZERO));
}

__device__ __forceinline__ void mma_nvfp4(float *C, const uint32_t *A, const uint32_t *B)
{
  const uint32_t one = 0x38383838;
  asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
               "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
               : "=f"(C[0]), "=f"(C[1]), "=f"(C[2]), "=f"(C[3])
               : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]), "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]),
                 "r"(one), "h"(ZERO), "h"(ZERO), "r"(one), "h"(ZERO), "h"(ZERO));
}

// every thread: TILES accumulators, each fed ITERS times by the same instruction on its own register operands.
template <int FORMAT, int TILES>
__global__ void spin(float *out)
{
  uint32_t A[TILES][4], B[TILES][2];
  float C[TILES][4];
  for (int t = 0; t < TILES; t++)
  {
    for (int i = 0; i < 4; i++) A[t][i] = FORMAT == 0 ? 0x3c003c00u + 0x00010001u * (t + i) : 0x11111111u * (1 + (threadIdx.x + t + i) % 3), C[t][i] = 0.f;
    for (int i = 0; i < 2; i++) B[t][i] = FORMAT == 0 ? 0x3c003c00u + 0x00010001u * (t + 2 * i) : 0x11111111u * (1 + (blockIdx.x + t + i) % 3);
  }
  for (uint32_t iter = 0; iter < ITERS; iter++)
#pragma unroll
    for (int t = 0; t < TILES; t++)
    {
      if (FORMAT == 0) mma_bf16(C[t], A[t], B[t]);
      if (FORMAT == 1) mma_fp8(C[t], A[t], B[t]);
      if (FORMAT == 2) mma_mxfp8(C[t], A[t], B[t]);
      if (FORMAT == 3) mma_nvfp4(C[t], A[t], B[t]);
      if (FORMAT == 4) mma_s8(C[t], A[t], B[t]);
    }
  float sum = 0.f;
  for (int t = 0; t < TILES; t++) sum += C[t][0] + C[t][3];
  if (threadIdx.x == 0) out[blockIdx.x] = sum;
}

// one warp's inner loop of a tiled matmul: MT fragments of A and NT of B from shared memory, then MT * NT instructions.
template <int FORMAT, int MT, int NT>
__global__ void fed(float *out)
{
  extern __shared__ uint32_t smem[];
  const uint32_t warp = threadIdx.x / 32, lane = threadIdx.x % 32, words = 1024; // 4 kb of operands per warp
  for (uint32_t i = lane; i < words; i += 32) smem[warp * words + i] = FORMAT == 0 ? 0x3c003c00u : 0x11111111u * (1 + i % 3);
  __syncthreads();
  float C[MT][NT][4];
  for (int i = 0; i < MT; i++) for (int j = 0; j < NT; j++) for (int c = 0; c < 4; c++) C[i][j][c] = 0.f;
  const uint32_t *base = smem + warp * words + (lane % 16) * 4; // ldmatrix: 16 lanes each point at one 16-byte row
  for (uint32_t iter = 0; iter < ITERS; iter++)
  {
    uint32_t A[MT][4], B[NT][2];
    const uint32_t *at = base + (iter % 4) * 64;
#pragma unroll
    for (int i = 0; i < MT; i++)
      asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n" : "=r"(A[i][0]), "=r"(A[i][1]), "=r"(A[i][2]), "=r"(A[i][3]) : "l"(at + i * 64));
#pragma unroll
    for (int j = 0; j < NT; j++)
      asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n" : "=r"(B[j][0]), "=r"(B[j][1]) : "l"(at + (MT + j) * 64));
#pragma unroll
    for (int i = 0; i < MT; i++)
#pragma unroll
      for (int j = 0; j < NT; j++)
      {
        if (FORMAT == 0) mma_bf16(C[i][j], A[i], B[j]);
        if (FORMAT == 2) mma_mxfp8(C[i][j], A[i], B[j]);
        if (FORMAT == 3) mma_nvfp4(C[i][j], A[i], B[j]);
      }
  }
  float sum = 0.f;
  for (int i = 0; i < MT; i++) for (int j = 0; j < NT; j++) sum += C[i][j][0] + C[i][j][3];
  if (threadIdx.x == 0) out[blockIdx.x] = sum;
}

template <int FORMAT, int MT, int NT>
void fed_rate(const char *name, float *out, int sms)
{
  const int depth[4] = {16, 32, 32, 64};
  for (int warps : {4, 8})
  {
    const int blocks = sms * 32 / warps;
    cudaEvent_t start, stop;
    cudaEventCreate(&start), cudaEventCreate(&stop);
    fed<FORMAT, MT, NT><<<blocks, warps * 32, warps * 1024 * 4>>>(out);
    cudaEventRecord(start);
    fed<FORMAT, MT, NT><<<blocks, warps * 32, warps * 1024 * 4>>>(out);
    cudaEventRecord(stop), cudaEventSynchronize(stop);
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    printf("fed %s warp=%dx%d warps=%d %.0f\n", name, MT * 16, NT * 8, warps, 2.0 * 16 * 8 * depth[FORMAT] * MT * NT * ITERS * double(blocks) * warps / (ms * 1e-3) / 1e12);
  }
}

template <int FORMAT>
void fed_report(const char *name, float *out, int sms)
{
  fed_rate<FORMAT, 1, 1>(name, out, sms);
  fed_rate<FORMAT, 2, 2>(name, out, sms);
  fed_rate<FORMAT, 2, 4>(name, out, sms);
  fed_rate<FORMAT, 2, 8>(name, out, sms);
  fed_rate<FORMAT, 4, 4>(name, out, sms);
  fed_rate<FORMAT, 4, 8>(name, out, sms);
  fed_rate<FORMAT, 8, 4>(name, out, sms);
}

// a whole tiled matmul with nothing left out but the arithmetic's meaning: TM x TN output tiles, warps of 64 x (8 NT)
// outputs, operands streamed from global memory with cp.async into two shared-memory stages (rows padded by 16 bytes
// so the 8 rows of one ldmatrix land in 8 different banks), ldmatrix into registers, then the instruction. `a` is
// (rows, depth) bytes and `b` (columns, depth) bytes; the bytes are not laid out as a real kernel would need, so the
// result is meaningless, but every load, wait and instruction of a real kernel is there. this is what a hand-written
// kernel could reach, to hold against cutlass (whose sm_120 kernels only build with 128 x 128 and 128 x 64 tiles).
constexpr uint32_t DEEP = 128, STRIDE = 144;

template <int FORMAT, int TM, int TN, int NT>
__global__ void streamed(const uint8_t *a, const uint8_t *b, float *out, uint32_t depth)
{
  extern __shared__ uint8_t bytes[];
  constexpr uint32_t stage_bytes = (TM + TN) * STRIDE, chunks = (TM + TN) * 8, threads = (TM / 64) * (TN / (8 * NT)) * 32;
  const uint32_t warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const uint8_t *rows[2] = {a + size_t(blockIdx.y) * TM * depth, b + size_t(blockIdx.x) * TN * depth};
  auto fetch = [&](uint32_t tile) { // one 128-byte-deep slice of both operands into stage tile % 2
    uint8_t *stage = bytes + (tile % 2) * stage_bytes;
    for (uint32_t chunk = threadIdx.x; chunk < chunks; chunk += threads)
    {
      const uint32_t which = chunk >= TM * 8, row = (chunk - which * TM * 8) / 8, col = chunk % 8;
      const uint32_t target = static_cast<uint32_t>(__cvta_generic_to_shared(stage + (which * TM + row) * STRIDE + col * 16));
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(target), "l"(rows[which] + size_t(row) * depth + tile * DEEP + col * 16));
    }
    asm volatile("cp.async.commit_group;\n" ::);
  };
  float C[4][NT][4];
  for (int i = 0; i < 4; i++) for (int j = 0; j < NT; j++) for (int c = 0; c < 4; c++) C[i][j][c] = 0.f;
  const uint32_t tiles = depth / DEEP, row_a = (warp % (TM / 64)) * 64, row_b = (warp / (TM / 64)) * 8 * NT;
  fetch(0);
  for (uint32_t tile = 0; tile < tiles; tile++)
  {
    if (tile + 1 < tiles) fetch(tile + 1);
    if (tile + 1 < tiles) asm volatile("cp.async.wait_group 1;\n" ::); else asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();
    const uint8_t *stage = bytes + (tile % 2) * stage_bytes;
    for (uint32_t step = 0; step < 4; step++)
    {
      uint32_t A[4][4], B[NT][2];
#pragma unroll
      for (int i = 0; i < 4; i++)
        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n" : "=r"(A[i][0]), "=r"(A[i][1]), "=r"(A[i][2]), "=r"(A[i][3])
                     : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(stage + (row_a + i * 16 + lane % 8 + 8 * ((lane / 8) % 2)) * STRIDE + step * 32 + 16 * (lane / 16)))));
#pragma unroll
      for (int j = 0; j < NT; j++)
        asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n" : "=r"(B[j][0]), "=r"(B[j][1])
                     : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(stage + (TM + row_b + j * 8 + lane % 8) * STRIDE + step * 32 + 16 * ((lane / 8) % 2)))));
#pragma unroll
      for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < NT; j++)
        {
          if (FORMAT == 2) mma_mxfp8(C[i][j], A[i], B[j]);
          if (FORMAT == 3) mma_nvfp4(C[i][j], A[i], B[j]);
        }
    }
    __syncthreads();
  }
  float sum = 0.f;
  for (int i = 0; i < 4; i++) for (int j = 0; j < NT; j++) sum += C[i][j][0] + C[i][j][3];
  if (threadIdx.x == 0) out[blockIdx.y * gridDim.x + blockIdx.x] = sum;
}

// rows x columns x depth bytes; an fp8 byte is one value, an nvfp4 byte two.
template <int FORMAT, int TM, int TN, int NT>
void streamed_rate(const char *name, uint32_t rows, uint32_t columns, uint32_t depth)
{
  const uint32_t smem = 2 * (TM + TN) * STRIDE, threads = (TM / 64) * (TN / (8 * NT)) * 32;
  uint8_t *a, *b;
  float *out;
  cudaMalloc(&a, size_t(rows) * depth), cudaMalloc(&b, size_t(columns) * depth), cudaMalloc(&out, sizeof(float) * (rows / TM) * (columns / TN));
  cudaMemset(a, 0x11, size_t(rows) * depth), cudaMemset(b, 0x11, size_t(columns) * depth);
  cudaFuncSetAttribute(streamed<FORMAT, TM, TN, NT>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  cudaEvent_t start, stop;
  cudaEventCreate(&start), cudaEventCreate(&stop);
  const dim3 grid(columns / TN, rows / TM);
  streamed<FORMAT, TM, TN, NT><<<grid, threads, smem>>>(a, b, out, depth);
  cudaEventRecord(start);
  for (int run = 0; run < 10; run++) streamed<FORMAT, TM, TN, NT><<<grid, threads, smem>>>(a, b, out, depth);
  cudaEventRecord(stop), cudaEventSynchronize(stop);
  float ms;
  cudaEventElapsedTime(&ms, start, stop);
  const double values = FORMAT == 3 ? 2.0 * depth : depth;
  printf("streamed %s tile=%dx%d warp=64x%d %ux%ux%.0f %.3f ms %.0f (%s)\n", name, TM, TN, 8 * NT, rows, columns, values, ms / 10,
         2.0 * rows * columns * values * 10 / (ms * 1e-3) / 1e12, cudaGetErrorString(cudaGetLastError()));
  cudaFree(a), cudaFree(b), cudaFree(out);
}

// the dit's matmuls as (rows of activations, weight rows, depth in bytes): mlp hidden, q/k/v/out, mlp out.
template <int TM, int TN, int NT>
void streamed_report()
{
  streamed_rate<2, TM, TN, NT>("mxfp8", 4096, 24576, 4096), streamed_rate<2, TM, TN, NT>("mxfp8", 4096, 4096, 4096);
  streamed_rate<3, TM, TN, NT>("nvfp4", 4096, 24576, 2048), streamed_rate<3, TM, TN, NT>("nvfp4", 4096, 4096, 6144);
}

template <int FORMAT, int TILES>
double rate(float *out, int blocks, int warps)
{
  const int depth[5] = {16, 32, 32, 64, 32};
  cudaEvent_t start, stop;
  cudaEventCreate(&start), cudaEventCreate(&stop);
  spin<FORMAT, TILES><<<blocks, warps * 32>>>(out);
  cudaEventRecord(start);
  spin<FORMAT, TILES><<<blocks, warps * 32>>>(out);
  cudaEventRecord(stop), cudaEventSynchronize(stop);
  float ms;
  cudaEventElapsedTime(&ms, start, stop);
  return 2.0 * 16 * 8 * depth[FORMAT] * TILES * ITERS * double(blocks) * warps / (ms * 1e-3) / 1e12;
}

template <int FORMAT>
void report(const char *name, float *out, int sms)
{
  for (int warps : {4, 8, 16})
  {
    const int blocks = sms * 32 / warps;
    printf("peak %s warps=%d tiles=1 %.0f\n", name, warps, rate<FORMAT, 1>(out, blocks, warps));
    printf("peak %s warps=%d tiles=2 %.0f\n", name, warps, rate<FORMAT, 2>(out, blocks, warps));
    printf("peak %s warps=%d tiles=4 %.0f\n", name, warps, rate<FORMAT, 4>(out, blocks, warps));
    printf("peak %s warps=%d tiles=8 %.0f\n", name, warps, rate<FORMAT, 8>(out, blocks, warps));
  }
}

int main(int argc, char **argv)
{
  cudaDeviceProp card;
  cudaGetDeviceProperties(&card, 0);
  printf("peak card %s sms=%d\n", card.name, card.multiProcessorCount);
  streamed_report<128, 128, 4>(), streamed_report<128, 128, 8>(), streamed_report<256, 128, 8>(), streamed_report<128, 256, 8>(), streamed_report<256, 256, 8>();
  if (argc > 1) return 0; // any argument: the streamed matmuls only
  float *out;
  cudaMalloc(&out, sizeof(float) * card.multiProcessorCount * 64);
  report<0>("bf16", out, card.multiProcessorCount);
  report<1>("fp8", out, card.multiProcessorCount);
  report<2>("mxfp8", out, card.multiProcessorCount);
  report<3>("nvfp4", out, card.multiProcessorCount);
  report<4>("int8", out, card.multiProcessorCount);
  fed_report<0>("bf16", out, card.multiProcessorCount);
  fed_report<2>("mxfp8", out, card.multiProcessorCount);
  fed_report<3>("nvfp4", out, card.multiProcessorCount);
  return 0;
}
