# sageattention 2's attention kernel alone (quantised inputs made once) at the denoise shapes, for whatever tile
# sizes the in-place build under /sage was compiled with: `PYTHONPATH=/sage:src python tools/sage_tiles.py CTA_Q CTA_K WARP_Q WARP_K`.
# tools/sage_tiles.sh rewrites the four constants, rebuilds and calls this once per setting.
import json
import sys

import torch
import torch.nn.functional as F
from sageattention import core

from magpie.dit import DitConfig, rope_table
from magpie.kernels import qk_prep
from sage_bench import bench

SHAPES = [(4096, 8239), (16384, 32911)]  # (queries, keys): edit decode steps at 1024 and 2048

if __name__ == "__main__":
    cta_q, cta_k, warp_q, warp_k = (int(value) for value in sys.argv[1:5])
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    cfg = DitConfig()
    dim = cfg.width // cfg.heads
    for queries, keys in SHAPES:
        raw = lambda count: (torch.randn(1, count, cfg.heads, dim, device="cuda") * 3).bfloat16()
        weight = (torch.rand(dim, device="cuda") * 1.5 + 0.5).bfloat16()
        rope = torch.cat([rope_table(torch.arange(keys) - keys // 2, d) for d in cfg.rope_axes], dim=-1).to("cuda")
        k, v = qk_prep(raw(keys), weight, rope, cfg.eps), raw(keys)
        q = qk_prep(raw(queries), weight, rope[-queries:], cfg.eps)
        want = F.scaled_dot_product_attention(*(x.transpose(1, 2).float() for x in (q, k, v))).transpose(1, 2)
        q8, q_scale, k8, k_scale = core.per_warp_int8_cuda(q, k, k.mean(dim=1, keepdim=True), tensor_layout="NHD", BLKQ=cta_q, WARPQ=warp_q, BLKK=warp_k)
        v8, v_scale, _ = core.per_channel_fp8(v, tensor_layout="NHD", scale_max=448.0, smooth_v=False)
        blocks = -(-keys // cta_k)
        v8 = F.pad(v8.view(torch.uint8), (0, blocks * cta_k - v8.shape[-1])).view(torch.float8_e4m3fn)  # zero bytes are fp8 zeros
        k_scale = F.pad(k_scale, (0, blocks * (cta_k // warp_k) - k_scale.shape[-1]), value=1.0)
        out = torch.empty_like(q)
        kernel = lambda: core.sm89_compile.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf(q8, k8, v8, out, q_scale, k_scale, v_scale, 0, 0, 2, dim**-0.5, 0)
        ms = bench(kernel)
        print("tiles", json.dumps({"cta_q": cta_q, "cta_k": cta_k, "warp_q": warp_q, "warp_k": warp_k, "shape": f"{queries}x{keys}", "kernel_ms": round(ms, 3),
                                   "tflops": round(4 * queries * keys * cfg.width / ms / 1e9), "gap": round(float((out.float() - want).norm() / want.norm()), 4)}), flush=True)
