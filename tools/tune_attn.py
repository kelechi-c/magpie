# profiles and hillclimbs native/attn.cu (the sm_120 attention kernel) on the card it will run on. every candidate is
# one build with its own -DATTN_* flags and module name, measured at the two edit decode shapes:
#   ms         per call, the kernel alone (operands quantised once), median of 3 x 10 calls
#   gap        relative l2 error against exact fp32 attention
#   same       whether the output equals the base build's bit for bit (where the settings share operands)
# the base is upstream's setting (128 x 64 tile, 32 x 64 per warp, instant buffer, eager rescale); sageattention's own
# three accumulator variants are timed first as the reference. ABLATE builds are profiling only (wrong output).
#   python tools/tune_attn.py            -> lines starting "attn", and /out/tune_attn.json
#   python tools/tune_attn.py build K=V  (one build, used to compile in parallel)
import json
import pathlib as pl
import statistics
import subprocess
import sys
from typing import Any

Config = dict[str, int]
FLAGS = ["-O3", "-std=c++20", "-DENABLE_BF16", "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__", "--use_fast_math",
         "-diag-suppress=174", "-gencode", "arch=compute_120a,code=sm_120a"]
INCLUDES = ["/sage/csrc", "/sage/csrc/qattn"]
SHAPES = [(4096, 8239), (16384, 32911)]  # (queries, keys): edit decode steps at 1024 and 2048
BASE: Config = {"CTA_Q": 128, "CTA_K": 64, "WARP_Q": 32, "WARP_K": 64, "INST_BUF": 1, "LAZY": 0, "ABLATE": 0, "PV4": 0}
TILES = [(128, 128, 32, 128), (128, 128, 32, 64), (128, 64, 16, 64), (64, 64, 32, 64), (64, 64, 16, 64), (64, 128, 32, 128), (64, 128, 16, 128), (64, 128, 32, 64)]
WORKERS = 4


def label(config: Config) -> str:
    return "_".join(f"{key.lower()}{value}" for key, value in sorted(config.items()) if value != BASE[key]) or "base"


# the extension built with `config` as -DATTN_ flags; a second call with the same config loads the finished build.
def compiled(config: Config) -> Any:
    from torch.utils.cpp_extension import load

    folder = pl.Path("/tmp/attn") / label(config)
    folder.mkdir(parents=True, exist_ok=True)
    return load(f"magpie_attn_{label(config)}", ["native/attn.cu"], extra_include_paths=INCLUDES,
                extra_cuda_cflags=FLAGS + [f"-DATTN_{key}={value}" for key, value in config.items()], build_directory=str(folder), verbose=False)


# builds every config in its own process, WORKERS at a time -> the labels that compiled.
def build_all(configs: list[Config]) -> set[str]:
    done: set[str] = set()
    for first in range(0, len(configs), WORKERS):
        batch = configs[first : first + WORKERS]
        pl.Path("/tmp/attn").mkdir(exist_ok=True)
        jobs = [subprocess.Popen([sys.executable, __file__, "build", *(f"{key}={value}" for key, value in config.items())],
                                 stdout=open(f"/tmp/attn/{label(config)}.log", "w"), stderr=subprocess.STDOUT) for config in batch]
        done |= {label(config) for config, job in zip(batch, jobs) if job.wait() == 0}
    return done


# bf16 q, k, v through the real qk-prep for one shape, and exact attention on them.
def inputs(queries: int, keys: int) -> tuple[Any, Any, Any, Any]:
    import torch
    import torch.nn.functional as F

    from magpie.dit import DitConfig, rope_table
    from magpie.kernels import qk_prep

    cfg = DitConfig()
    dim = cfg.width // cfg.heads
    raw = lambda count: (torch.randn(1, count, cfg.heads, dim, device="cuda") * 3).bfloat16()
    weight = (torch.rand(dim, device="cuda") * 1.5 + 0.5).bfloat16()
    rope = torch.cat([rope_table(torch.arange(keys) - keys // 2, d) for d in cfg.rope_axes], dim=-1).to("cuda")
    q, k, v = qk_prep(raw(queries), weight, rope[-queries:], cfg.eps), qk_prep(raw(keys), weight, rope, cfg.eps), raw(keys)
    return q, k, v, F.scaled_dot_product_attention(*(x.transpose(1, 2).float() for x in (q, k, v))).transpose(1, 2)


# q, k, v quantised for a tile setting with sageattention's own quantisers, the tails padded to whole key tiles.
def quantised(q: Any, k: Any, v: Any, config: Config) -> tuple[Any, ...]:
    import torch
    import torch.nn.functional as F
    from sageattention import core

    q8, q_scale, k8, k_scale = core.per_warp_int8_cuda(q, k, k.mean(dim=1, keepdim=True), tensor_layout="NHD", BLKQ=config["CTA_Q"], WARPQ=config["WARP_Q"], BLKK=config["WARP_K"])
    v8, v_scale, _ = core.per_channel_fp8(v, tensor_layout="NHD", scale_max=448.0, smooth_v=False)
    blocks = -(-k.shape[1] // config["CTA_K"])
    v8 = F.pad(v8.view(torch.uint8), (0, blocks * config["CTA_K"] - v8.shape[-1])).view(torch.float8_e4m3fn).contiguous()  # zero bytes are fp8 zeros
    k_scale = F.pad(k_scale, (0, blocks * (config["CTA_K"] // config["WARP_K"]) - k_scale.shape[-1]), value=1.0).contiguous()
    return q8, k8, v8, q_scale, k_scale, v_scale


# slot -> key inside a 64-key tile: the order the 4-bit instruction reads a warp's weights in (native/attn_sm120.cuh).
def slot_keys() -> Any:
    import torch

    s = torch.arange(64)
    return 32 * (s // 32) + 8 * ((s % 8) // 2) + 2 * ((s % 32) // 8) + s % 2


# values (1, keys, heads, 128) -> nvfp4 for the PV4 kernel: bytes (1, heads, key tiles, 4608) and the per (head,
# channel) mean (1, heads, 128) that the kernel adds back (zero unless `centre`). per tile and channel: 64 keys in
# slot order as e2m1 nibbles, then 4 e4m3 scales, one per 16 slots. `shrinks`: each block keeps the scale, among
# its largest value over 6 times each shrink, with the least squared error.
def nvfp4_values(v: Any, centre: bool, shrinks: tuple[float, ...] = (1.0,)) -> tuple[Any, Any]:
    import torch
    import torch.nn.functional as F

    x = v.float().permute(0, 2, 3, 1)  # (1, heads, 128, keys)
    mean = x.mean(-1) if centre else torch.zeros_like(x[..., 0])
    x = F.pad(x - mean[..., None], (0, -x.shape[-1] % 64)).unflatten(-1, (-1, 64))[..., slot_keys().to(x.device)].unflatten(-1, (4, 16))
    top = (x.abs().amax(-1, keepdim=True) / 6).clamp(2.0**-6, 448.0)
    grid = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=x.device)
    best_error, best_codes, best_scale = None, None, None
    for shrink in shrinks:
        bits = ((top * shrink).clamp_min(2.0**-6).view(torch.int32) + 0xFFFFF) & ~0xFFFFF  # rounded up to 3 mantissa bits: e4m3
        scale = bits.view(torch.float32)
        ratio = (x.abs() / scale).clamp_max(6.0)
        codes = torch.where(ratio < 1, (2 * ratio).round().int(), ((ratio.view(torch.int32) + 0x200000) >> 22) - 0xFC)
        error = (grid[codes] * scale - x.abs()).square().sum(-1, keepdim=True)
        keep = error < best_error if best_error is not None else torch.ones_like(error, dtype=torch.bool)
        best_error = torch.where(keep, error, best_error) if best_error is not None else error
        best_codes = torch.where(keep, codes, best_codes) if best_codes is not None else codes
        best_scale = torch.where(keep, (bits >> 20) - 960, best_scale) if best_scale is not None else (bits >> 20) - 960
    nibbles = (best_codes | ((x < 0).int() << 3)).flatten(-2)  # (1, heads, 128, tiles, 64)
    data = (nibbles[..., 0::2] | (nibbles[..., 1::2] << 4)).to(torch.uint8)
    tile = torch.cat([data.permute(0, 1, 3, 2, 4).flatten(-2), best_scale[..., 0].to(torch.uint8).permute(0, 1, 3, 2, 4).flatten(-2)], dim=-1)
    return tile.contiguous(), mean.contiguous()


# median ms per call of three runs of ten.
def timed(fn: Any) -> float:
    from sage_bench import bench

    return statistics.median(bench(fn) for _ in range(3))


def main() -> None:
    import torch

    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    one = lambda **change: {**BASE, **change}
    configs = [BASE, one(PV4=1)]
    built = build_all(configs)
    data = {shape: inputs(*shape) for shape in SHAPES}
    gap = lambda out, want: round(float((out.float() - want).norm() / want.norm()), 5)
    rows: list[dict[str, Any]] = []
    for config in configs:
        if label(config) not in built:
            print("attn", json.dumps({"config": label(config), "failed": "build"}), flush=True)
            continue
        module = compiled(config)
        for shape in SHAPES:
            q, k, v, want = data[shape]
            operands = quantised(q, k, v, config)
            for name, values in ({"plain": (False, (1.0,)), "centred": (True, (1.0,)), "fitted": (False, (1.0, 0.85, 0.7, 0.55)),
                                  "centred, fitted": (True, (1.0, 0.85, 0.7, 0.55))} if config["PV4"] else {"": None}).items():
                v_operands = nvfp4_values(v, *values) if values else (operands[2], operands[5])
                out = torch.empty_like(q)
                call = lambda: module.attend(*operands[:2], v_operands[0], out, *operands[3:5], v_operands[1], q.shape[-1] ** -0.5)
                ms = timed(call)
                torch.cuda.synchronize()
                rows.append({"config": label(config), "values": name, "shape": "x".join(map(str, shape)), "ms": round(ms, 3), "gap": gap(out, want),
                             "tflops": round(4 * shape[0] * shape[1] * 4096 / ms / 1e9)})
                print("attn", json.dumps(rows[-1]), flush=True)
    pl.Path("/out/tune_attn.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    if sys.argv[1:2] == ["build"]:
        compiled({**BASE, **{key: int(value) for key, value in (item.split("=") for item in sys.argv[2:])}})
    else:
        main()
