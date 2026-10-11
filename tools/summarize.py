import json
import pathlib as pl

ROOT = pl.Path(__file__).resolve().parent.parent
BLOCK_PARAMS = 32 * (4 * 4096**2 + 3 * 4096 * 12288)  # weights every token passes through
TEXT_TOKENS = {"t2i": 61, "edit": 47}  # measured prompt lengths after the system prefix is dropped
GROUPS = {"aten::mm": "matmul (linear layers)", "aten::_flash_attention_forward": "attention",
          "aten::cudnn_convolution": "convolution (vae)"}


# dit work for one step in tflop: 2 flops per weight per query token, plus qk^t and pv in 32 layers.
def step_tflop(queries: list[tuple[int, int]]) -> float:
    linear = 2 * BLOCK_PARAMS * sum(q for q, _ in queries)
    attention = 32 * 4 * 4096 * sum(q * k for q, k in queries)
    return (linear + attention) / 1e12


# gpu ms per operation group from the profiler's top-30 `aten::` rows (kernel rows repeat their parent op).
def op_groups(ops: list[dict]) -> dict[str, int]:
    rows = [op for op in ops if op["op"].startswith("aten::")]
    names = [*GROUPS.values(), "elementwise, casts and copies"]
    return {name: round(sum(op["self_gpu_ms"] for op in rows if GROUPS.get(op["op"], names[-1]) == name)) for name in names}


# (prefill, cached) work for a case: which query blocks attend to how many keys.
def work(case: str) -> tuple[float, float]:
    size = int(case.split("_")[1])
    target = (size // 16) ** 2
    text = TEXT_TOKENS["edit" if "edit" in case else "t2i"]
    reference = target if "edit" in case else 0
    prefix = text + reference
    prefill = [(text, text // 2), (reference, prefix), (target, prefix + target)]
    return step_tflop(prefill), step_tflop([(target, prefix + target)])


def main() -> None:
    rows = []
    for variant in ("turbo", "base"):
        data = json.loads((ROOT / "out/bench" / f"{variant}.json").read_text())
        for engine in ("diffusers", "magpie"):
            for case, row in data[engine].items():
                steps = row["dit_steps_p50"]
                cached = "nocache" not in case
                prefill, later = work(case.replace("_nocache", ""))
                rest = sum(steps[1:]) / (len(steps) - 1)
                stage = lambda key: round(row[key]["p50"] * 1e3, 1) if key in row else 0.0
                rows.append({
                    "variant": variant, "engine": engine, "case": case, "steps": len(steps),
                    "total_ms": stage("total"), "text_ms": stage("text"), "vae_encode_ms": stage("vae_encode"),
                    "dit_ms": stage("dit"), "vae_decode_ms": stage("vae_decode"), "other_ms": stage("other"),
                    "step0_ms": round(steps[0] * 1e3, 1), "later_step_ms": round(rest * 1e3, 1),
                    "step0_tflop": round(prefill, 1), "later_step_tflop": round(later if cached else prefill, 1),
                    "later_step_tflops": round((later if cached else prefill) / rest, 1),
                    "peak_gib": row["peak_allocated_gib"],
                    "psnr_vs_diffusers_db": [round(row["psnr_db"][k], 1) for k in ("min", "p50", "max")] if "psnr_db" in row else None,
                })
    turbo = json.loads((ROOT / "out/bench/turbo.json").read_text())
    summary = {"device": turbo["device"], "resident_gib": turbo["resident_gib"], "load_seconds": round(turbo["load_seconds"], 1),
               "roofline": {k: round(v, 1) for k, v in turbo["roofline"].items()}, "rows": rows,
               "ops": {case: ops[:14] for case, ops in turbo["ops"].items()},
               "op_groups_ms": {case: op_groups(ops) for case, ops in turbo["ops"].items()}}
    (ROOT / "out/bench/summary.json").write_text(json.dumps(summary, indent=1))
    for r in rows:
        print(f"{r['variant']:5s} {r['engine']:9s} {r['case']:18s} {r['total_ms']:8.0f} ms  step {r['later_step_ms']:7.1f} ms  {r['later_step_tflop']:6.1f} tflop  {r['later_step_tflops']:5.1f} tflop/s  peak {r['peak_gib']}")


if __name__ == "__main__":
    main()
