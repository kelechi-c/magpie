# hillclimbs the cutlass matmul's compile-time settings on the card it will run on. every candidate is one build of
# native/gemm.cu with its own -D flags and module name, checked in three tiers, cheapest first:
#   1. same bits: its output on fixed operands must equal the base build's exactly (a tile or schedule change must
#      not change the arithmetic); `exact` and the largest difference are recorded.
#   2. rate: ms at the shapes a 1024 edit calls, weighted by how often it calls them -> `score_ms`, the matmul time
#      of one request. idle gpu, median of 3 x 10 calls.
#   3. the gate (tools/gate.py), run by hand on the winner once native/gemm.cu's defaults are set to it.
# round 1 tries every tile shape under both schedules per format; round 2 tries stage counts on the two best. a build
# that does not compile or cannot start is recorded and skipped.
# found 2026-10-13: only 128 x {64, 128} x 128 compiles; ping-pong for mxfp8 is the one gain (658 -> 632 ms).
#   python tools/tune_gemm.py            -> /out/tune_gemm.json
#   python tools/tune_gemm.py build K=V  (one build, used by the search to compile in parallel)
import json
import pathlib as pl
import statistics
import subprocess
import sys
import time
from typing import Any

Config = dict[str, int]
FLAGS = ["-O3", "-std=c++20", "-DNDEBUG", "--expt-relaxed-constexpr", "-gencode", "arch=compute_120a,code=sm_120a"]
INCLUDES = ["/cutlass/include", "/cutlass/tools/util/include"]
# (rows, out features, in features) and calls per 1024 edit: step 0 runs 8,239 rows, the other seven 4,096.
SHAPES = {"mxfp8": {(4096, 24576, 4096): 224, (8239, 24576, 4096): 32, (4096, 4096, 4096): 896, (8239, 4096, 4096): 128},
          "nvfp4": {(4096, 4096, 12288): 224, (8239, 4096, 12288): 32}}
TILES = [(128, 128, 128), (256, 128, 128), (128, 256, 128), (256, 256, 128), (128, 128, 256), (128, 128, 64), (64, 128, 128), (128, 64, 128)]
WORKERS = 4


def label(config: Config) -> str:
    return "_".join(f"{key.lower()}{value}" for key, value in sorted(config.items())) or "base"


# the extension built with `config` as -D flags; a second call with the same config loads the finished build.
def compiled(config: Config) -> Any:
    from torch.utils.cpp_extension import load

    folder = pl.Path("/tmp/gemm") / label(config)
    folder.mkdir(parents=True, exist_ok=True)
    return load(f"magpie_gemm_{label(config)}", ["native/gemm.cu"], extra_include_paths=INCLUDES,
                extra_cuda_cflags=FLAGS + [f"-D{key}={value}" for key, value in config.items()], build_directory=str(folder), verbose=False)


# builds every config in its own process, WORKERS at a time -> the labels that compiled.
def build_all(configs: list[Config]) -> set[str]:
    done: set[str] = set()
    for first in range(0, len(configs), WORKERS):
        batch = configs[first : first + WORKERS]
        jobs = [subprocess.Popen([sys.executable, __file__, "build", *(f"{key}={value}" for key, value in config.items())],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) for config in batch]
        done |= {label(config) for config, job in zip(batch, jobs) if job.wait() == 0}
    return done


# fixed block-scaled operands for every shape: {format: {shape: (a, sfa, b, sfb)}}.
def operands() -> dict[str, dict[tuple[int, int, int], tuple[Any, ...]]]:
    import torch

    from magpie.kernels import mx_rows, nv_rows

    torch.manual_seed(0)
    made: dict[str, dict[tuple[int, int, int], tuple[Any, ...]]] = {}
    for name, pack in (("mxfp8", mx_rows), ("nvfp4", nv_rows)):
        made[name] = {}
        for rows, out, width in SHAPES[name]:
            x = (torch.randn(rows, width, device="cuda") * torch.exp2(torch.randint(-3, 4, (rows, 1), device="cuda").float())).bfloat16()
            w = torch.randn(out, width, device="cuda").bfloat16()
            made[name][(rows, out, width)] = (*pack(x), *pack(w))
    return made


# tiers 1 and 2 for one build: bit agreement with `reference` outputs (filled in when empty) and weighted ms.
def measure(module: Any, inputs: dict[str, dict[tuple[int, int, int], tuple[Any, ...]]], reference: dict[str, Any]) -> dict[str, Any]:
    import torch

    report: dict[str, Any] = {"exact": True, "largest_gap": 0.0, "score_ms": 0.0, "shapes": {}}
    for name, shapes in SHAPES.items():
        run = getattr(module, name)
        for shape, calls in shapes.items():
            args = inputs[name][shape]
            out = torch.empty(shape[0], shape[1], device="cuda", dtype=torch.bfloat16)
            run(*args, out, 1.0)
            want = reference.setdefault(f"{name}{shape}", out.clone())
            report["exact"] &= bool(torch.equal(out, want))
            report["largest_gap"] = max(report["largest_gap"], float((out.float() - want.float()).abs().max() / want.float().abs().max()))
            times = []
            for _ in range(3):
                torch.cuda.synchronize()
                start = time.perf_counter()
                for _ in range(10):
                    run(*args, out, 1.0)
                torch.cuda.synchronize()
                times.append((time.perf_counter() - start) / 10 * 1e3)
            ms = statistics.median(times)
            report["shapes"][f"{name} {shape[0]}x{shape[1]}x{shape[2]}"] = {"ms": round(ms, 4), "tflops": round(2 * shape[0] * shape[1] * shape[2] / ms / 1e9, 1)}
            report["score_ms"] += calls * ms
    return report


def main() -> None:
    inputs, reference, results = operands(), {}, {}
    out = pl.Path("/out/tune_gemm.json")

    def round_of(configs: list[Config]) -> None:
        built = build_all(configs)
        for config in configs:
            name = label(config)
            try:
                results[name] = {"config": config, **(measure(compiled(config), inputs, reference) if name in built else {"compiled": False})}
            except RuntimeError as error:  # a stage count the kernel cannot start with
                results[name] = {"config": config, "failed": str(error)}
            print(name, json.dumps({key: value for key, value in results[name].items() if key != "shapes"}), flush=True)
            out.write_text(json.dumps(results, indent=1))

    round_of([{}] + [{"TILE_M": m, "TILE_N": n, "TILE_K": k, "SCHEDULE_MX": mx, "SCHEDULE_NV": nv} for mx in (0, 1) for nv in (0, 1) for m, n, k in TILES
                     if (m, n, k, mx, nv) != (128, 128, 128, 1, 0)])
    best = sorted((name for name in results if "score_ms" in results[name]), key=lambda name: results[name]["score_ms"])[:2]
    round_of([{**results[name]["config"], "STAGES": stages} for name in best for stages in (2, 3, 4, 5, 6, 8)])
    ranked = sorted((name for name in results if "score_ms" in results[name]), key=lambda name: results[name]["score_ms"])
    print("best", ranked[0], results[ranked[0]]["score_ms"], "base", results["base"]["score_ms"], flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["build"]:
        compiled({key: int(value) for key, value in (pair.split("=") for pair in sys.argv[2:])})
    else:
        main()
