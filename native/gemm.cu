// block-scaled matmuls on sm_120 tensor cores through cutlass, as a torch extension: d = a @ b^T in bf16, where a is
// (m, k) row-major activations and b is (n, k) row-major weights (torch's linear layout), both in a block-scaled
// format with one scale per group of elements along k:
//   mxfp8: fp8 e4m3 values, one power-of-two scale (ue8m0 byte) per 32 elements
//   nvfp4: e2m1 values packed two per byte, one fp8 (ue4m3) scale per 16 elements
// the tensor-core instruction applies the scales itself, so nothing scalar runs inside the loop. the scale tensors
// use cutlass's interleaved layout (see magpie.kernels.scale_slots). kernel settings follow cutlass example 79.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

// compile-time settings, searched by tools/tune_gemm.py: the tile one thread block computes, the kernel schedule per
// format (0: cooperative, 1: ping-pong) and the number of tiles loaded ahead (0: as many as fit). on sm_120 only
// 128 x {64, 128} x 128 tiles compile; ping-pong is 4 % faster for mxfp8 and no faster for nvfp4.
#ifndef TILE_M
#define TILE_M 128
#endif
#ifndef TILE_N
#define TILE_N 128
#endif
#ifndef TILE_K
#define TILE_K 128
#endif
#ifndef SCHEDULE_MX
#define SCHEDULE_MX 1
#endif
#ifndef SCHEDULE_NV
#define SCHEDULE_NV 0
#endif
#ifndef STAGES
#define STAGES 0
#endif
using Cooperative = cutlass::gemm::collective::KernelScheduleAuto;
using Pingpong = cutlass::gemm::KernelTmaWarpSpecializedPingpong;

template <class Element, int Alignment, class Schedule>
struct BlockScaled {
  using ElementD = cutlass::bfloat16_t;
  using Tile = Shape<Int<TILE_M>, Int<TILE_N>, Int<TILE_K>>;
  using Cluster = Shape<_1, _1, _1>;
  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp, Tile, Cluster,
      cutlass::epilogue::collective::EpilogueTileAuto, float, float,
      ElementD, cutlass::layout::RowMajor, 8, ElementD, cutlass::layout::RowMajor, 8,
      cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp,
      Element, cutlass::layout::RowMajor, Alignment, Element, cutlass::layout::ColumnMajor, Alignment, float, Tile, Cluster,
#if STAGES > 0
      cutlass::gemm::collective::StageCount<STAGES>,
#else
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
#endif
      Schedule>::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, Mainloop, Epilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
  using Config = typename Mainloop::Sm1xxBlkScaledConfig;

  // one matmul on the current torch stream, its result multiplied by `alpha`; the workspace is kept between calls and only ever grows.
  static void run(torch::Tensor a, torch::Tensor sfa, torch::Tensor b, torch::Tensor sfb, torch::Tensor d, int k, float alpha) {
    static torch::Tensor workspace;
    int m = d.size(0), n = d.size(1);
    auto shape = make_shape(m, n, k, 1);
    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm, {m, n, k, 1},
        {reinterpret_cast<typename Element::DataType *>(a.data_ptr()), cutlass::make_cute_packed_stride(typename Kernel::StrideA{}, {m, k, 1}),
         reinterpret_cast<typename Element::DataType *>(b.data_ptr()), cutlass::make_cute_packed_stride(typename Kernel::StrideB{}, {n, k, 1}),
         reinterpret_cast<typename Element::ScaleFactorType *>(sfa.data_ptr()), Config::tile_atom_to_shape_SFA(shape),
         reinterpret_cast<typename Element::ScaleFactorType *>(sfb.data_ptr()), Config::tile_atom_to_shape_SFB(shape)},
        {{alpha, 0.0f},
         reinterpret_cast<ElementD *>(d.data_ptr()), cutlass::make_cute_packed_stride(typename Kernel::StrideC{}, {m, n, 1}),
         reinterpret_cast<ElementD *>(d.data_ptr()), cutlass::make_cute_packed_stride(typename Kernel::StrideD{}, {m, n, 1})}};
    int64_t need = static_cast<int64_t>(Gemm::get_workspace_size(args));
    if (!workspace.defined() || workspace.numel() < need)
      workspace = torch::empty({std::max<int64_t>(need, 1)}, torch::dtype(torch::kUInt8).device(a.device()));
    Gemm gemm;
    TORCH_CHECK(gemm.can_implement(args) == cutlass::Status::kSuccess, "cutlass cannot run this shape");
    auto stream = at::cuda::getCurrentCUDAStream();
    TORCH_CHECK(gemm.initialize(args, workspace.data_ptr(), stream) == cutlass::Status::kSuccess, "cutlass initialize failed");
    TORCH_CHECK(gemm.run(stream) == cutlass::Status::kSuccess, "cutlass run failed");
  }
};

// number of scale bytes cutlass reads for `rows` rows of `k` elements with one scale per `group` elements.
template <class Element, int Alignment>
int64_t slots(int rows, int k) {
  using Config = typename BlockScaled<Element, Alignment, Cooperative>::Config;
  return size(filter_zeros(Config::tile_atom_to_shape_SFA(make_shape(rows, 1, k, 1))));
}

using Mxfp8 = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
using Nvfp4 = cutlass::nv_float4_t<cutlass::float_e2m1_t>;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("mxfp8", [](torch::Tensor a, torch::Tensor sfa, torch::Tensor b, torch::Tensor sfb, torch::Tensor d, float alpha) {
    BlockScaled<Mxfp8, 16, std::conditional_t<SCHEDULE_MX == 1, Pingpong, Cooperative>>::run(a, sfa, b, sfb, d, a.size(1), alpha);
  });
  module.def("nvfp4", [](torch::Tensor a, torch::Tensor sfa, torch::Tensor b, torch::Tensor sfb, torch::Tensor d, float alpha) {
    BlockScaled<Nvfp4, 32, std::conditional_t<SCHEDULE_NV == 1, Pingpong, Cooperative>>::run(a, sfa, b, sfb, d, a.size(1) * 2, alpha);
  });
  module.def("mxfp8_slots", [](int rows, int k) { return slots<Mxfp8, 16>(rows, k); });
  module.def("nvfp4_slots", [](int rows, int k) { return slots<Nvfp4, 32>(rows, k); });
}
