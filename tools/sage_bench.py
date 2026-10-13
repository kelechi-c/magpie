# sageattention 2 (hand-written cuda: int8 q x k, fp8 weights x values) and 3 (fp4 on blackwell) against torch's sdpa
# and this repo's flash kernel, on q, k that went through the real qk-prep.
#   python tools/sage_bench.py            (needs the gpu and both sage packages; tools/modal_sage.py builds them)
import json
import math
from typing import Any, Callable

SHAPES = [(4096, 8239), (4096, 4096), (16384, 32911)]  # (queries, keys): edit 1024 decode, t2i 1024, edit 2048 decode


# ms per call on the gpu, after two warm-up calls.
def bench(fn: Callable[[], Any], iters: int = 10) -> float:
    import torch

    for _ in range(2):
        fn()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


# sage 2's entry point split in two: the per-call quantisation of q, k, v, and the attention kernel on the result.
# q, k, v are (1, tokens, heads, dim); `accum` picks the kernel as the package does.
def sage2_parts(q: Any, k: Any, v: Any, accum: str) -> tuple[Callable[[], Any], Callable[[], Any]]:
    import torch
    from sageattention import core

    def quantised() -> tuple:
        mean = k.mean(dim=1, keepdim=True)
        ints = core.per_warp_int8_cuda(q, k, mean, tensor_layout="NHD", BLKQ=128, WARPQ=32, BLKK=64)
        return (*ints, *core.per_channel_fp8(v, tensor_layout="NHD", scale_max=2.25 if accum == "fp32+fp16" else 448.0, smooth_v=False)[:2])

    q8, q_scale, k8, k_scale, v8, v_scale = quantised()
    out = torch.empty_like(q)
    kernel = {"fp32+fp16": core.sm89_compile.qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf,
              "fp32+fp32": core.sm89_compile.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf,
              "fp32": core.sm89_compile.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn}[accum]
    return quantised, lambda: kernel(q8, k8, v8, out, q_scale, k_scale, v_scale, 0, 0, 2, q.shape[-1] ** -0.5, 0)


# the same split for sage 3; its tensors are (1, heads, tokens, dim).
def sage3_parts(q: Any, k: Any, v: Any) -> tuple[Callable[[], Any], Callable[[], Any]]:
    from sageattn3 import api

    def quantised() -> tuple:
        qp, kp, vp, delta = api.preprocess_qkv(q, k.clone(), v, True)
        return api.scale_and_quant_fp4(qp), api.scale_and_quant_fp4_permute(kp), api.scale_and_quant_fp4_transpose(vp), delta

    ready = quantised()
    return quantised, lambda: api.blockscaled_fp4_attn(*ready, k.shape[2], False, True, True)


# per shape: ms and tflop/s (4 * queries * keys * width flops per call) for sdpa, flash and each sage variant whole,
# the sage calls split into quantisation and kernel, and each result's relative l2 gap to sdpa in fp32.
def run(shapes: list[tuple[int, int]] = SHAPES) -> str:
    import torch
    import torch.nn.functional as F
    from sageattention import sageattn_qk_int8_pv_fp8_cuda
    from sageattn3 import sageattn3_blackwell

    from magpie.dit import DitConfig, rope_table
    from magpie.kernels import flash, qk_prep

    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    cfg = DitConfig()
    dim = cfg.width // cfg.heads
    report: dict[str, Any] = {"gpu": torch.cuda.get_device_name(0), "cases": {}}
    for queries, keys in shapes:
        raw = lambda count: (torch.randn(1, count, cfg.heads, dim, device="cuda") * 3).bfloat16()
        weight = (torch.rand(dim, device="cuda") * 1.5 + 0.5).bfloat16()
        rope = torch.cat([rope_table(torch.arange(keys) - keys // 2, d) for d in cfg.rope_axes], dim=-1).to("cuda")
        k, v = qk_prep(raw(keys), weight, rope, cfg.eps), raw(keys)
        q = qk_prep(raw(queries), weight, rope[-queries:], cfg.eps)
        qh, kh, vh = (x.transpose(1, 2).contiguous() for x in (q, k, v))
        want = F.scaled_dot_product_attention(qh.float(), kh.float(), vh.float())
        gap = lambda got: float((got.float() - want).norm() / want.norm())
        calls = {"sdpa": (lambda: F.scaled_dot_product_attention(qh, kh, vh), True), "flash (ours)": (lambda: flash(q, k, v, False), False),
                 **{f"sage2 {accum}": (lambda accum=accum: sageattn_qk_int8_pv_fp8_cuda(q, k, v, tensor_layout="NHD", qk_quant_gran="per_warp",
                                                                                     pv_accum_dtype=accum), False) for accum in ("fp32+fp16", "fp32+fp32", "fp32")},
                 "sage3": (lambda: sageattn3_blackwell(qh, kh.clone(), vh), True)}
        row: dict[str, Any] = {}
        for name, (call, by_head) in calls.items():
            ms = bench(call)
            row[name] = {"ms": ms, "tflops": 4 * queries * keys * cfg.width / ms / 1e9, "gap": gap(call() if by_head else call().transpose(1, 2))}
        parts = {**{f"sage2 {accum}": sage2_parts(q, k, v, accum) for accum in ("fp32+fp16", "fp32+fp32", "fp32")}, "sage3": sage3_parts(qh, kh, vh)}
        for name, (quantise, kernel) in parts.items():
            row[name].update(quantise_ms=bench(quantise), kernel_ms=bench(kernel))
        report["cases"][f"{queries}x{keys}"] = row
        print(queries, keys, json.dumps(row), flush=True)
    return json.dumps(report)


if __name__ == "__main__":
    print(run())
