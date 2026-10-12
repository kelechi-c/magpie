"""the gate, with no provider in it: runs wherever a blackwell card, the weights and the baseline files are.

    python tools/gate.py --tag <name> [--cases edit_1024] [--compare] [--sdpa]

expects the modal layout on disk: weights in the hugging face cache, edit inputs in INPUTS, and under
OUT_ROOT/baseline/<variant>/ the frozen `reference`, `conditioning`, `magpie` and `steps` folders.
`--compare` first times the baseline arithmetic (kernels off) on this machine and judges against that,
which is what a card other than the one the baseline was frozen on needs.
"""

import argparse
import json
import pathlib as pl
import statistics
import time
from typing import Any, Callable

INPUTS = "/root/inputs"
OUT_ROOT = "/out"
REPOS = {"turbo": "Qwen/Qwen-Image-2.1-Turbo", "base": "Qwen/Qwen-Image-2.1"}
Timings = dict[str, list[float]]


# (prompt, flicker image used as the edit input, edit instruction): ink / pencil subjects throughout.
SUITE = [
    ("Ink drawing of a huge radio telescope array on a salt flat, hundreds of dishes shrinking to hatched ellipses toward the horizon, sepia pen lines on warm ivory paper with small turquoise and rust accents, wide empty sky, sparse and precise, no text",
     "03.png", "Change the season to deep winter: lay snow on the dishes and the salt flat with sparse pencil hatching, keep the sepia pen linework and the ivory paper"),
    ("A mountain range built entirely from stacked wavering contour lines like a topographic map, drawn in dark teal ink on cream paper, a loose orange and pink watercolour wash for the sunrise behind the highest peak, minimal, hand-drawn, no text",
     "07.png", "Replace the sunrise wash with a night sky, a deep indigo ink wash scattered with tiny white stars, keep the contour-line mountains unchanged"),
    ("A bloom of jellyfish rising through open water, their tentacles drawn as long beaded chains of tiny scalloped loops that cross and tangle, violet and aquamarine ballpoint ink on warm ivory paper, fine hatching in the bells, airy composition, no text",
     "08.png", "Redraw this as a soft graphite pencil sketch on grey paper, remove all colour, keep the composition"),
    ("Hand-drawn ink diagram of a half-open folding fan, hair-thin dark ribs radiating from a single pivot point with a zigzag outer edge, soft indigo and coral watercolour blooms bleeding across a few of its panels, faint smudged pencil construction lines behind it, on flat pale peach paper, minimal and quiet, no text",
     "01.png", "Add a small sleeping cat beneath the fan, drawn in the same hair-thin ink line style, keep the fan and the paper unchanged"),
    ("A magpie perched on a bare winter branch, drawn in graphite pencil with dense cross-hatching for the black feathers and bare paper for the white, one small accent of cobalt blue ink on the wing, warm cream paper, generous empty space, hand-drawn, no text",
     "06.png", "Add a second, smaller moon behind the planet, drawn in the same red and blue coloured pencil hatching, keep everything else unchanged"),
]


CASES: dict[str, dict[str, Any]] = {
    "t2i_512": dict(edit=False, size=512),
    "edit_512": dict(edit=True, size=512),
    "t2i_1024": dict(edit=False, size=1024),
    "edit_1024": dict(edit=True, size=1024),
    "edit_1024_nocache": dict(edit=True, size=1024, use_kv_cache=False),
    "t2i_2048": dict(edit=False, size=2048),
    "edit_2048": dict(edit=True, size=2048),
}


GATE_CASES = ["t2i_1024", "edit_1024", "t2i_2048", "edit_2048"]
TIMED_CASES = "edit_1024"  # what a candidate is timed on: five warm requests; 2048 is estimated from it
STEP_ERROR_MAX = 0.02  # worst relative l2 error of one dit step on the baseline's own inputs
DINO_MIN = 0.98  # lowest dinov2 cls cosine between a candidate image and the frozen baseline image
DINO = ("facebook/dinov2-base", "f9e44c814b77203eaa57a6bdbbd535f21ede1415")  # as in flicker


def summary(values: list[float]) -> dict[str, float]:
    return {"p50": statistics.median(values), "min": min(values), "max": max(values), "n": len(values)}


# one request's stage totals plus the per-step dit times, then clears the collector.
def collect(timings: Timings, total: float) -> dict[str, Any]:
    stages = {name: sum(values) for name, values in timings.items()}
    row = {"total": total, **stages, "other": total - sum(stages.values()), "dit_steps": list(timings.get("dit", []))}
    timings.clear()
    return row


# median of every scalar stage across requests, and the per-step medians of the dit.
def reduce(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scalars = {key: summary([row.get(key, 0.0) for row in rows]) for key in rows[0] if key != "dit_steps"}
    steps = [statistics.median(step) for step in zip(*[row["dit_steps"] for row in rows])]
    return {**scalars, "dit_steps_p50": steps}


def device_info() -> dict[str, Any]:
    import diffusers
    import torch
    import transformers

    return {"gpu": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0)),
            "vram_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1),
            "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "diffusers": diffusers.__version__, "transformers": transformers.__version__,
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "cudnn_benchmark": torch.backends.cudnn.benchmark}


def psnr(a: Any, b: Any) -> float:
    import numpy as np

    error = np.mean((np.asarray(a.convert("RGB"), dtype=np.float64) - np.asarray(b.convert("RGB"), dtype=np.float64)) ** 2)
    return float(10 * np.log10(255.0**2 / max(error, 1e-12)))


# gate items per case: every suite prompt, except base at 2048 where three keep the session short.
def gate_items(variant: str, case: str) -> int:
    return 3 if variant == "base" and "2048" in case else len(SUITE)


# like `timed`, and also keeps the stage's peak allocation in gib (resident weights included).
def staged(function: Callable, name: str, timings: Timings, peaks: dict[str, float]) -> Callable:
    import torch

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        result = function(*args, **kwargs)
        torch.cuda.synchronize()
        timings.setdefault(name, []).append(time.perf_counter() - start)
        peaks[name] = max(peaks.get(name, 0.0), torch.cuda.max_memory_allocated() / 2**30)
        return result

    return wrapper


# the prompt and condition images of one gate item.
def gate_input(case: str, index: int) -> tuple[str, tuple[Any, ...]]:
    from PIL import Image

    prompt, source, instruction = SUITE[index % len(SUITE)]
    return (instruction, (Image.open(f"{INPUTS}/{source}"),)) if CASES[case]["edit"] else (prompt, ())


# the port's dit and vae on the card, and nothing else: no pipeline, no text encoder. the checkpoint's
# safetensors are read directly and cast to bf16, so only torch and safetensors are needed. `slim` keeps the mlp
# weights in fp8 only; pass False for anything that runs the baseline arithmetic (`unfused`).
def load_port(variant: str, slim: bool = True, attention: tuple[str, ...] = ("k", "v")) -> tuple[Any, Any]:
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    import magpie.pipeline as lean

    torch.set_grad_enabled(False)
    # the cached snapshot folder; a repo id would query the hub, which is offline on modal
    snapshot = pl.Path(hf_hub_download(REPOS[variant], "model_index.json", local_files_only=True)).parent

    def read(part: str) -> dict[str, Any]:
        return {name: weight.to("cuda", torch.bfloat16)
                for file in sorted((snapshot / part).glob("*.safetensors")) for name, weight in load_file(file).items()}

    config = json.loads((snapshot / "vae" / "config.json").read_text())
    return lean.adopt(read("transformer"), read("vae"), config["latents_mean"], config["latents_std"], slim, attention)


# where one port request leaves the reference, with accumulation removed: every dit step is rerun on
# the reference's own inputs, and the vae decodes the reference's own latents.
def diagnose(dit: Any, vae: Any, record: dict, text: Any, slots: Any, shapes: list, sampler: Any, seed: int) -> dict[str, Any]:
    import torch

    from magpie.dit import layout
    from magpie.pipeline import pack
    from magpie.sampling import schedule

    def gap(ours: Any, theirs: Any) -> float:
        return float((ours.double().cpu() - theirs.double().cpu()).abs().max() / theirs.double().abs().max())

    target = shapes[-1][0] * shapes[-1][1]
    joint = record["joint"].to("cuda")
    noise = torch.randn((1, 1, 64, *shapes[-1]), generator=torch.Generator("cpu").manual_seed(seed), dtype=joint.dtype)
    lay = layout(slots.to("cuda"), shapes, dit.cfg)
    levels = (schedule(target, sampler).to("cuda") * 1000).to(joint.dtype) / 1000
    first, cache = dit.prefill(joint, text.to("cuda", joint.dtype), levels[:1], lay)
    steps = [gap(first, record["velocity"][0])] + [
        gap(dit.decode(record["latents"][i].to("cuda"), levels[i : i + 1], lay, cache), record["velocity"][i])
        for i in range(1, len(record["velocity"]))]
    z, pixels = record["decode"]
    raw = vae.decoder(vae.post_quant_conv(z[:, :, 0].to("cuda"))).clamp(-1.0, 1.0)
    return {"text": gap(text, record["text"]), "noise": gap(pack(noise[:, 0]), joint[:, -target:]), "dit_steps_same_input": steps,
            "vae_decode_same_input_mean_abs": float((raw.float().cpu() - pixels[:, :, 0].float()).abs().mean())}


# the baseline port's behaviour for a build that has since gained kernels: qk-norm + rope unfused, torch's sdpa, bf16 mlp.
def unfused(dit_module: Any) -> None:
    def prep(x: Any, weight: Any, rope: Any, eps: float, out: Any = None) -> Any:
        plain = dit_module.rotate(dit_module.rms(x, weight, eps), rope)
        return plain if out is None else out.copy_(plain)

    dit_module.prep = prep
    dit_module.attend = lambda q, k, v, bound, mask=None: dit_module.sdpa(q, k, v, mask)
    dit_module.fp8_ready = lambda packed, x: False
    dit_module.normed = lambda x, mod, eps: dit_module.F.layer_norm(x, x.shape[-1:], eps=eps) * mod


# relative l2 error of every dit step when the candidate is fed the baseline's own step inputs
# (`record` from the `steps` freeze). no trajectory is followed, so rounding noise is not amplified between steps.
def step_errors(dit: Any, record: dict, text: Any, slots: Any, shapes: list, sampler: Any) -> list[float]:
    from magpie.dit import layout
    from magpie.sampling import schedule

    def gap(ours: Any, theirs: Any) -> float:
        theirs = theirs.to("cuda").float()
        return float((ours.float() - theirs).norm() / theirs.norm())

    joint = record["joint"].to("cuda")
    lay = layout(slots.to("cuda"), shapes, dit.cfg)
    levels = (schedule(shapes[-1][0] * shapes[-1][1], sampler).to("cuda") * 1000).to(joint.dtype) / 1000
    first, cache = dit.prefill(joint, text.to("cuda", joint.dtype), levels[:1], lay)
    later = [gap(dit.decode(latents.to("cuda"), levels[i : i + 1], lay, cache), record["velocity"][i])
             for i, latents in enumerate(record["latents"], start=1)]
    return [gap(first, record["velocity"][0]), *later]


# dinov2 cls-token cosine similarity of image pairs: 1.0 means the same picture to the encoder.
def dino_similarity(pairs: list[tuple[Any, Any]]) -> list[float]:
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoImageProcessor, AutoModel

    folder = pl.Path(hf_hub_download(DINO[0], "config.json", revision=DINO[1], local_files_only=True)).parent
    processor, model = AutoImageProcessor.from_pretrained(folder), AutoModel.from_pretrained(folder).to("cuda").eval()
    images = [image.convert("RGB") for pair in pairs for image in pair]
    features = model(**processor(images=images, return_tensors="pt").to("cuda")).last_hidden_state[:, 0]
    features = torch.nn.functional.normalize(features.float(), dim=-1).unflatten(0, (-1, 2))
    return (features[:, 0] * features[:, 1]).sum(-1).tolist()


# phase b, and the gate itself: the lean port's dit + vae on saved conditioning, no text encoder.
# per case: stage times, per-stage peaks, each item's psnr against the frozen reference image, and, when
# this run is not the baseline, its dinov2 similarity to the frozen baseline image and its step errors.
# the timers it hangs on the models are taken off again at the end.
def gate_pass(dit: Any, vae: Any, variant: str, cases: list[str], folder: pl.Path, out: pl.Path, captured: dict) -> dict[str, Any]:
    import torch
    from PIL import Image

    import magpie.pipeline as lean
    from magpie.sampling import BASE, TURBO

    sampler = TURBO if variant == "turbo" else BASE
    timings: Timings = {}
    peaks: dict[str, float] = {}
    plain = dit.prefill, dit.decode, vae.encode, vae.decode
    dit.prefill, dit.decode = staged(dit.prefill, "dit", timings, peaks), staged(dit.decode, "dit", timings, peaks)
    vae.encode, vae.decode = staged(vae.encode, "vae_encode", timings, peaks), staged(vae.decode, "vae_decode", timings, peaks)
    results: dict[str, Any] = {}
    for case in cases:
        spec = CASES[case]

        def request(index: int) -> tuple[Any, Any, Any, list]:
            saved = torch.load(folder / "conditioning" / case / f"{index:02d}.pt")
            images = lean.resized(gate_input(case, index)[1], spec["size"])
            output = lean.render(dit, vae, saved["text"], saved["slots"], sampler, images, None if images else (spec["size"],) * 2, spec["size"], index)
            return output, saved["text"], saved["slots"], images

        request(0)
        timings.clear()
        peaks.clear()
        rows, scores, drift, checks, pairs, errors = [], [], [], [], [], []
        for index in range(gate_items(variant, case)):
            torch.cuda.synchronize()
            start = time.perf_counter()
            output, text, slots, images = request(index)
            torch.cuda.synchronize()
            rows.append(collect(timings, time.perf_counter() - start))
            scores.append(psnr(output, Image.open(folder / "reference" / case / f"{index:02d}.png")))
            frozen = folder / "magpie" / case / f"{index:02d}.png"
            shapes = [(image.height // 16, image.width // 16) for image in images] + [(output.height // 16, output.width // 16)]
            if out != folder / "magpie" and frozen.exists():
                drift.append(psnr(output, Image.open(frozen)))
                pairs.append((output, Image.open(frozen)))
                errors.append(step_errors(dit, torch.load(folder / "steps" / case / f"{index:02d}.pt"), text, slots, shapes, sampler))
                timings.clear()
            (out / case).mkdir(parents=True, exist_ok=True)
            output.convert("RGB").save(out / case / f"{index:02d}.png")
            if (case, index) in captured:
                checks.append(diagnose(dit, vae, captured[(case, index)], text, slots, shapes, sampler, index))
                timings.clear()
        results[case] = {**reduce(rows), "stage_peak_gib": {k: round(v, 2) for k, v in peaks.items()}, "psnr_db": scores,
                         "psnr_vs_baseline_db": drift, "diagnosis": checks, "step_error": errors,
                         "dino_vs_baseline": dino_similarity(pairs) if pairs else []}
        print(f"gate {case}: dit+vae p50 {results[case]['total']['p50']:.3f} s, psnr min {min(scores):.1f} dB", flush=True)
    dit.prefill, dit.decode, vae.encode, vae.decode = plain
    return results


# pass / fail of a gate run: no dit step may leave the baseline's output by more than STEP_ERROR_MAX on the
# baseline's own inputs, and no image may fall below DINO_MIN similarity to the frozen baseline image.
# psnr against the official reference is reported, not judged: it cannot tell rounding from damage.
def verdict(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    cases = {}
    for case, row in candidate["magpie"].items():
        frozen = baseline["magpie"][case]
        worst_step, worst_dino = max(max(item) for item in row["step_error"]), min(row["dino_vs_baseline"])
        cases[case] = {"pass": worst_step <= STEP_ERROR_MAX and worst_dino >= DINO_MIN,
                       "worst_step_error": round(worst_step, 5), "worst_dino": round(worst_dino, 4),
                       "worst_psnr_change_db": round(min(ours - theirs for ours, theirs in zip(row["psnr_db"], frozen["psnr_db"])), 2),
                       "dit_vae_ms": round(row["total"]["p50"] * 1e3, 1), "baseline_ms": round(frozen["total"]["p50"] * 1e3, 1),
                       "speedup": round(frozen["total"]["p50"] / row["total"]["p50"], 3)}
    return {"pass": all(c["pass"] for c in cases.values()), "cases": cases}


# one gate run on this machine. with `compare`, the baseline arithmetic is timed here first and judged against.
def run(variant: str, tag: str, cases: list[str], compare: bool, sdpa: bool) -> dict[str, Any]:
    import torch

    import magpie.dit as dit_module

    dit, vae = load_port(variant, slim=not compare)
    folder = pl.Path(OUT_ROOT) / "baseline" / variant
    result: dict[str, Any] = {"variant": variant, "tag": tag, "device": device_info(),
                              "resident_dit_vae_gib": round(torch.cuda.memory_allocated() / 2**30, 2)}
    baseline = json.loads((pl.Path(__file__).resolve().parent.parent / "baseline" / f"{variant}.json").read_text())
    if compare:
        kernels = dit_module.prep, dit_module.attend, dit_module.fp8_ready, dit_module.normed
        unfused(dit_module)
        baseline = {"magpie": gate_pass(dit, vae, variant, cases, folder, pl.Path(OUT_ROOT) / "gate" / f"{tag}-unfused" / variant, {})}
        dit_module.prep, dit_module.attend, dit_module.fp8_ready, dit_module.normed = kernels
        result["unfused_here"] = baseline["magpie"]
    if sdpa:  # torch's attention under the other kernels, to price the attention kernel alone
        dit_module.attend = lambda q, k, v, bound, mask=None: dit_module.sdpa(q, k, v, mask)
    result["magpie"] = gate_pass(dit, vae, variant, cases, folder, pl.Path(OUT_ROOT) / "gate" / tag / variant, {})
    return {**result, "verdict": verdict(result, baseline)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="candidate")
    parser.add_argument("--variant", default="turbo")
    parser.add_argument("--cases", default=TIMED_CASES)
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--sdpa", action="store_true")
    args = parser.parse_args()
    result = run(args.variant, args.tag, args.cases.split(","), args.compare, args.sdpa)
    out = pl.Path(OUT_ROOT) / "gate" / f"{args.tag}-{args.variant}.json"
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result["verdict"], indent=1))


if __name__ == "__main__":
    main()
