# the quantised kv cache against sageattention 2's own entry point at the edit decode shapes: ms per call, and the gap
# to exact attention on a "later step" whose target keys and values differ from the ones the cache was packed with
# (values 1.5x larger, to lean on the frozen fp8 step's headroom).   PYTHONPATH=src:tools python tools/sage_cache_bench.py
import json

import torch
import torch.nn.functional as F

from magpie.dit import DitConfig, rope_table
from magpie.kernels import qk_prep, sage, sage_cached, sage_pack, sage_write
from sage_bench import bench

SHAPES = [(4096, 8239), (16384, 32911)]  # (queries, keys): edit decode steps at 1024 and 2048

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    cfg = DitConfig()
    dim = cfg.width // cfg.heads
    for queries, keys in SHAPES:
        raw = lambda count, size=3.0: (torch.randn(1, count, cfg.heads, dim, device="cuda") * size).bfloat16()
        weight = (torch.rand(dim, device="cuda") * 1.5 + 0.5).bfloat16()
        rope = torch.cat([rope_table(torch.arange(keys) - keys // 2, d) for d in cfg.rope_axes], dim=-1).to("cuda")
        k, v = qk_prep(raw(keys), weight, rope, cfg.eps), raw(keys)
        kv = sage_pack(k, v, keys - queries)
        q = qk_prep(raw(queries), weight, rope[-queries:], cfg.eps)
        k[:, -queries:], v[:, -queries:] = qk_prep(raw(queries), weight, rope[-queries:], cfg.eps), raw(queries, 4.5)
        kv.tail[0][:, -queries:], kv.tail[1][:, -queries:] = k[:, -queries:], v[:, -queries:]
        want = F.scaled_dot_product_attention(*(x.transpose(1, 2).float() for x in (q, k, v))).transpose(1, 2)
        gap = lambda out: round(float((out.float() - want).norm() / want.norm()), 4)
        cached = lambda: sage_cached(q, sage_write(kv, *kv.tail, kv.start))
        print("sage_cache", json.dumps({"shape": f"{queries}x{keys}", "sage_ms": round(bench(lambda: sage(q, k, v)), 3), "cached_ms": round(bench(cached), 3),
                                        "write_ms": round(bench(lambda: sage_write(kv, *kv.tail, kv.start)), 3), "kernel_ms": round(bench(lambda: sage_cached(q, kv)), 3),
                                        "sage_gap": gap(sage(q, k, v)), "cached_gap": gap(cached())}), flush=True)
