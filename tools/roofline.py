"""rooflines for one turbo edit: per span, the work it does and the least time that work can take on this card.

    uv run --no-project python tools/roofline.py

floor = flops / the datasheet tensor-core rate + bytes / the datasheet memory bandwidth. the bytes are those of
every elementwise stage (norms, rope, activations, residuals) fused into one read and one write of its
tensors; matmul, attention and convolution kernels are counted by flops only. the datasheet is a constant; the
same sum at the rates measured on the card (out/bench/summary.json) is kept beside it as `reached`. measured
times come from out/trace/kernels.json. writes out/roofline.json.
"""

import json
import pathlib as pl
from typing import Any, NamedTuple

ROOT = pl.Path(__file__).resolve().parent.parent
WIDTH, LAYERS, MLP_RATIO, STEPS, TEXT_TOKENS = 4096, 32, 3, 8, 47
BF16 = 2  # bytes per element
ENCODER_DIMS = (96, 96, 192, 384, 768, 768)
DECODER_DIMS = (1152, 1152, 1152, 576, 288, 144)
BIG_KINDS = ("gemm", "attention", "conv")
DIT_SPANS = ("mlp (swiglu)", "qkv + qk-norm + rope", "attention (sdpa)", "attention out")
VAE_SPANS = ("vae decode", "vae encode")
LINEAR_SPANS = ("mlp (swiglu)", "qkv + qk-norm + rope", "attention out")  # what a quantised dit would run in fp8 / fp4
# rtx pro 6000 blackwell server edition, peak figures. nvidia's datasheet gives fp4, fp32 and the bandwidth; the
# bf16 and fp8 rows are from lenovo's product guide for the same card. neither says dense or with sparsity.
DATASHEET = {"bf16_tflops": 1000.0, "fp8_tflops": 2000.0, "fp4_tflops": 4000.0, "bandwidth_gbs": 1597.0}
DATASHEET_SOURCES = ("https://www.akamai.com/site/en/documents/corporate/nvidia-rtx-pro-6000-blackwell-server-edition-datasheet.pdf",
                     "https://lenovopress.lenovo.com/lp2263-thinksystem-nvidia-rtx-pro-6000-blackwell-server-edition-pcie-gen5-gpu")


class Work(NamedTuple):
    flops: float = 0.0
    bytes: float = 0.0


def add(*parts: Work) -> Work:
    return Work(sum(p.flops for p in parts), sum(p.bytes for p in parts))


# elementwise stage fused into one kernel: `elements` counts every element read plus every element written.
def passes(elements: float) -> Work:
    return Work(bytes=BF16 * elements)


def conv(inp: int, out: int, area: float, kernel: int = 3) -> Work:
    return Work(flops=2 * kernel**2 * inp * out * area)


# the dit over one cached edit: step 1 runs prefix + target, the other seven the target only. per token and
# block: three 4096 -> 12288-class matmuls in the mlp, q/k/v and the output projection, and around them
# qk-norm + rope (read q, k; write q, k), silu * gate (read two, write one, 12288 wide), and the glue:
# two norm * modulate stages and two gated residual adds.
def dit_work(size: int) -> dict[str, Work]:
    target = (size // 16) ** 2
    prefix = TEXT_TOKENS + target
    tokens = LAYERS * (prefix + STEPS * target)
    pairs = LAYERS * (TEXT_TOKENS * (TEXT_TOKENS // 2) + target * prefix + STEPS * target * (prefix + target))
    hidden = MLP_RATIO * WIDTH
    return {
        "mlp (swiglu)": add(Work(flops=2 * 3 * WIDTH * hidden * tokens), passes(3 * hidden * tokens)),
        "qkv + qk-norm + rope": add(Work(flops=2 * 3 * WIDTH**2 * tokens), passes(4 * WIDTH * tokens)),
        "attention (sdpa)": Work(flops=4 * WIDTH * pairs),
        "attention out": Work(flops=2 * WIDTH**2 * tokens),
        "block glue": passes((2 * 2 + 2 * 3) * WIDTH * tokens),
    }


# two norm + silu stages, two 3x3 convolutions, a 1x1 shortcut when the width changes, one residual add.
def resnet(inp: int, out: int, area: float) -> Work:
    shortcut = conv(inp, out, area, 1) if inp != out else Work()
    return add(conv(inp, out, area), conv(out, out, area), shortcut, passes((2 * inp + 2 * out + 3 * out) * area))


# two resnets around single-head attention over every position.
def mid(dim: int, area: float) -> Work:
    attention = add(conv(dim, 3 * dim, area, 1), Work(flops=4 * dim * area**2), conv(dim, dim, area, 1), passes(5 * dim * area))
    return add(resnet(dim, dim, area), attention, resnet(dim, dim, area))


def vae_decode(size: int) -> Work:
    area = (size // 16) ** 2
    parts = [conv(64, 64, area, 1), conv(64, DECODER_DIMS[0], area), mid(DECODER_DIMS[0], area)]
    for index, (inp, out) in enumerate(zip(DECODER_DIMS, DECODER_DIMS[1:])):
        parts += [resnet(inp, out, area), resnet(out, out, area), resnet(out, out, area)]
        if index < len(DECODER_DIMS) - 2:  # nearest upsample, 3x3 conv, channel-repeat shortcut, add
            parts += [passes(5 * out * area), conv(out, out, 4 * area), passes((inp + 4 * out) * area), passes(12 * out * area)]
            area *= 4
    return add(*parts, passes(4 * DECODER_DIMS[-1] * area), conv(DECODER_DIMS[-1], 4, area), passes(8 * area))


def vae_encode(size: int) -> Work:
    area = float(size**2)
    parts = [conv(4, ENCODER_DIMS[0], area)]
    for index, (inp, out) in enumerate(zip(ENCODER_DIMS, ENCODER_DIMS[1:])):
        parts += [resnet(inp, out, area), resnet(out, out, area)]
        if index == len(ENCODER_DIMS) - 2:
            parts += [passes(3 * out * area)]
            continue
        parts += [conv(out, out, area / 4), passes(inp * area + out * area / 4), passes(3 * out * area / 4)]
        area /= 4
    return add(*parts, mid(ENCODER_DIMS[-1], area), passes(2 * ENCODER_DIMS[-1] * area), conv(ENCODER_DIMS[-1], 128, area),
               conv(128, 128, area, 1))


# measured (ms, ms in matmul / attention / convolution kernels) per span of one profiled request. the glue
# is every kernel outside the labelled spans: norms, modulation, residuals and the cache concatenation.
def measured(trace: dict[str, Any]) -> dict[str, tuple[float, float]]:
    def span(name: str) -> tuple[float, float]:
        row = trace["spans"][name]
        return row["gpu_ms"], sum(row["kernels_ms"].get(kind, 0.0) for kind in BIG_KINDS)

    rows = {name: span(name) for name in (*DIT_SPANS, *VAE_SPANS)}
    return {**rows, "block glue": (trace["kernel_ms"] - sum(total for total, _ in rows.values()), 0.0)}


# milliseconds for `work` at a compute rate (tflop/s) and a bandwidth (gb/s): (compute, memory).
def price(work: Work, tflops: float, gbs: float) -> tuple[float, float]:
    return work.flops / tflops / 1e9, work.bytes / gbs / 1e6


# one row per span: what it took, what its big kernels achieved, the datasheet floor split into compute and
# memory, and the same sum at the rates the card has been measured to reach. the request rows add the idle
# time between kernels; the fp8 / fp4 ones reprice only the dit's linear layers.
def roofline(size: int, trace: dict[str, Any], reached: dict[str, float]) -> dict[str, Any]:
    work = {**dit_work(size), "vae decode": vae_decode(size), "vae encode": vae_encode(size)}
    spans = {}
    for name, (total, big) in measured(trace).items():
        compute, memory = price(work[name], DATASHEET["bf16_tflops"], DATASHEET["bandwidth_gbs"])
        spans[name] = {"measured_ms": round(total, 1), "big_kernel_ms": round(big, 1), "small_kernel_ms": round(total - big, 1),
                       "tflop": round(work[name].flops / 1e12, 1), "gb_moved_fused": round(work[name].bytes / 1e9, 1),
                       "achieved_tflops": round(work[name].flops / big / 1e9, 1) if big else None,
                       "floor_compute_ms": round(compute, 1), "floor_memory_ms": round(memory, 1), "floor_ms": round(compute + memory, 1),
                       "reached_ms": round(sum(price(work[name], reached["gemm_8192_bf16_tflops"], reached["copy_gb_per_s"])), 1)}
    idle = trace["wall_ms"] - trace["kernel_ms"]
    floor = sum(row["floor_ms"] for row in spans.values()) + idle
    linear = sum(work[name].flops for name in LINEAR_SPANS) / 1e9
    saved = lambda rate: linear / DATASHEET["bf16_tflops"] - linear / DATASHEET[rate]
    return {"spans": spans, "kernel_ms": round(trace["kernel_ms"], 1), "idle_ms": round(idle, 1), "request_ms": round(trace["wall_ms"], 1),
            "floor_request_ms": round(floor, 1), "reached_request_ms": round(sum(row["reached_ms"] for row in spans.values()) + idle, 1),
            "floor_request_fp8_ms": round(floor - saved("fp8_tflops"), 1), "floor_request_fp4_ms": round(floor - saved("fp4_tflops"), 1)}


def main() -> None:
    reached = json.loads((ROOT / "out/bench/summary.json").read_text())["roofline"]
    traces = json.loads((ROOT / "out/trace/kernels.json").read_text())
    report = {"datasheet": DATASHEET, "datasheet_sources": DATASHEET_SOURCES,
              "reached": {key: reached[key] for key in ("gemm_8192_bf16_tflops", "copy_gb_per_s")},
              "cases": {case: roofline(int(case.split("_")[1]), traces[case], reached) for case in ("edit_1024", "edit_2048")}}
    (ROOT / "out/roofline.json").write_text(json.dumps(report, indent=1))
    for case, row in report["cases"].items():
        print(f"{case}: request {row['request_ms']:.0f} ms, floor {row['floor_request_ms']:.0f} ms (fp8 {row['floor_request_fp8_ms']:.0f}, "
              f"fp4 {row['floor_request_fp4_ms']:.0f}), at reached rates {row['reached_request_ms']:.0f} ms")
        for name, span in row["spans"].items():
            print(f"  {name:22s} {span['measured_ms']:7.0f} ms (small {span['small_kernel_ms']:6.0f})  {span['tflop']:7.1f} tflop at "
                  f"{span['achieved_tflops'] or 0:5.1f}/s  floor {span['floor_ms']:6.0f} = {span['floor_compute_ms']:.0f} + "
                  f"{span['floor_memory_ms']:.0f}  reached {span['reached_ms']:6.0f}")


if __name__ == "__main__":
    main()
