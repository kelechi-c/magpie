"""the gate, with no provider in it: runs wherever a blackwell card, the weights and the baseline files are.

    python tools/gate.py --tag <name> [--cases edit_1024] [--compare] [--sdpa]

expects the modal layout on disk: weights in the hugging face cache, edit inputs in INPUTS, and under
OUT_ROOT/baseline/<variant>/ the frozen `reference`, `conditioning`, `magpie` and `steps` folders.
`--compare` first times the baseline arithmetic (kernels off) on this machine and judges against that,
which is what a card other than the one the baseline was frozen on needs.
"""

import argparse
import json
import os
import pathlib as pl
import statistics
import time
from typing import Any, Callable

INPUTS = "/root/inputs"
OUT_ROOT = "/out"
WIDE_INPUTS = "/out/baseline/turbo/inputs16"
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


# the wide set, sixteen items for the final gate: (prompt for the text-to-image input, edit instruction). the inputs
# are generated once by the official pipeline (`freeze-wide`) and live on the volume under WIDE_INPUTS.
WIDE = [
    ("Pencil drawing of a lighthouse on a rocky headland, dense cross-hatched cliffs, a pale wash of grey-blue sea, soft graphite clouds, cream paper, generous empty sky, no text",
     "Turn the scene to night: darken the sky with a deep indigo ink wash and add a bright beam of light from the lighthouse, keep the pencil hatching of the rocks"),
    ("Ink drawing of an old bicycle leaning against a brick wall, thin black pen lines, brick texture in loose hatching, a single red watercolour accent on the bicycle frame, ivory paper, no text",
     "Add a wicker basket full of yellow flowers on the handlebars, drawn in the same thin pen lines with a light watercolour wash, keep everything else unchanged"),
    ("Graphite sketch of a fox curled asleep among ferns, soft shading on the fur, fine pencil strokes for the fern fronds, a faint ochre wash, warm grey paper, no text",
     "Cover the ferns and the ground with snow, leaving the fox uncovered, using sparse pencil strokes and bare paper for the snow"),
    ("Pen and ink drawing of a wooden sailing ship on calm water, rigging in hair-thin lines, stippled reflections, pale turquoise wash over the sea, off-white paper, no text",
     "Add a large full moon behind the masts, drawn as a plain circle with a light yellow wash, keep the ship and the water unchanged"),
    ("Charcoal drawing of a stone bridge with three arches over a slow river, smudged shadows under the arches, reeds drawn with quick strokes, textured beige paper, no text",
     "Redraw this as a clean technical ink line drawing with no shading, thin uniform black lines on white paper, keep the composition"),
    ("Coloured pencil drawing of a teapot and two cups on a checked tablecloth, soft hatching in blue and terracotta, visible pencil grain, cream paper, no text",
     "Change the colour of the teapot to deep green, keeping the coloured pencil hatching style and everything else unchanged"),
    ("Ink sketch of a narrow street of timber-framed houses, overhanging upper floors, cobblestones in loose circles, sepia pen lines with a light brown wash, ivory paper, no text",
     "Add strings of small paper lanterns hanging between the houses, in red and orange watercolour, keep the sepia linework unchanged"),
    ("Pencil study of an owl perched on a fence post, detailed feather pattern in graphite, large dark eyes, loose grass sketched at the base, off-white paper, no text",
     "Make it rain: add fine diagonal pencil strokes for rain across the whole drawing and small puddles at the base of the post"),
    ("Ink and watercolour drawing of a greenhouse full of potted plants, glass panes as thin ruled lines, leaves in three shades of green wash, pale paper, no text",
     "Remove all colour and render the drawing in black ink only, with hatching where the green washes were"),
    ("Fine pen drawing of a pocket watch with its back open, visible gears and springs in precise line work, light stippling for shadow, a touch of gold ochre wash, white paper, no text",
     "Replace the gold ochre wash with a cool blue wash and keep every line of the mechanism unchanged"),
    ("Graphite drawing of a mountain hut beside a lake, pine trees as vertical hatched shapes, the lake left as bare paper with a few horizontal strokes, grey-toned paper, no text",
     "Add a small rowing boat on the lake with one figure in it, drawn in the same graphite style, keep the hut and the trees unchanged"),
    ("Ink drawing of a market stall with crates of fruit, awning in broad striped wash of red and white, fruit as simple circles with dots of colour, cream paper, no text",
     "Change the awning stripes to blue and white, keep the pen lines, the fruit and the crates unchanged"),
    ("Pencil drawing of a grand piano seen from above, the lid open, strings drawn as fine parallel lines, soft shadow on a wooden floor, warm ivory paper, no text",
     "Add an open book of sheet music on the music stand and a single red rose lying on the closed part of the lid, in coloured pencil"),
    ("Ink sketch of a windmill on a flat plain, sails as crossed lattice lines, long grass in quick upward strokes, a wide sky with three small birds, pale buff paper, no text",
     "Turn the season to autumn: add orange and brown watercolour washes to the grass and a few falling leaves, keep the ink lines unchanged"),
    ("Detailed pen drawing of a snail on a mossy branch, spiral shell with fine concentric lines, moss as dense stippling, a pale green wash, white paper, no text",
     "Redraw this as a soft watercolour painting with no ink lines, loose washes and blurred edges, keep the composition"),
    # the lettering item (2026-10-13; it replaced a row of bottles whose fill level was the model's free choice)
    ('Black and white letterpress poster for a village fair, printed in black ink on white paper, no colour anywhere. Across the top in large bold condensed '
     'capitals: "MIDSUMMER FAIR". Directly beneath, in smaller italic capitals: "SATURDAY 21 JUNE - GATES OPEN 10 AM". In the centre, an engraved illustration of '
     'a carousel horse on a striped pole, drawn with fine cross-hatching, with one five-pointed star on each side. Below the illustration, three stacked lines in '
     'clean sans-serif capitals: "LIVE BRASS BAND", "CIDER AND PIES", "FIREWORKS AT DUSK". Along the bottom, inside a ruled box, in small capitals: '
     '"ADMISSION 2 SHILLINGS - CHILDREN FREE". A double-line frame runs around the whole sheet, with a small ornament in each corner.',
     'Recolour the poster as a two-colour print: make the title "MIDSUMMER FAIR" and the two stars deep red, and give the carousel horse a mustard-yellow wash, '
     'keeping all other ink black on white paper. Change the line "FIREWORKS AT DUSK" to "FIREWORKS AT NINE" and change "2 SHILLINGS" to "3 SHILLINGS". '
     'Keep every other word, the layout and the engraving unchanged.'),
]
# what the lettering item's input and its edit must say, line by line.
LETTERING = (["MIDSUMMER FAIR", "SATURDAY 21 JUNE - GATES OPEN 10 AM", "LIVE BRASS BAND", "CIDER AND PIES", "FIREWORKS AT DUSK", "ADMISSION 2 SHILLINGS - CHILDREN FREE"],
             ["MIDSUMMER FAIR", "SATURDAY 21 JUNE - GATES OPEN 10 AM", "LIVE BRASS BAND", "CIDER AND PIES", "FIREWORKS AT NINE", "ADMISSION 3 SHILLINGS - CHILDREN FREE"])


CASES: dict[str, dict[str, Any]] = {
    "t2i_512": dict(edit=False, size=512),
    "edit_512": dict(edit=True, size=512),
    "t2i_1024": dict(edit=False, size=1024),
    "edit_1024": dict(edit=True, size=1024),
    "edit_1024_nocache": dict(edit=True, size=1024, use_kv_cache=False),
    "t2i_2048": dict(edit=False, size=2048),
    "edit_2048": dict(edit=True, size=2048),
    "edit16_1024": dict(edit=True, size=1024, wide=True),
    "edit16_2048": dict(edit=True, size=2048, wide=True),
}


GATE_CASES = ["t2i_1024", "edit_1024", "t2i_2048", "edit_2048"]
TIMED_CASES = "edit_1024"  # what a candidate is timed on: five warm requests; 2048 is estimated from it
STEP_ERROR_MAX = 0.02  # relative l2 error of one dit step on the baseline's own inputs; reported, not judged
DREAMSIM_MAX = 0.05  # highest dreamsim distance between a candidate image and the frozen baseline image (provisional)
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


# gate items per case: every suite prompt (sixteen for the wide cases), except base at 2048 where three keep the session short.
def gate_items(variant: str, case: str) -> int:
    return len(WIDE) if CASES[case].get("wide") else 3 if variant == "base" and "2048" in case else len(SUITE)


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

    if CASES[case].get("wide"):
        return WIDE[index][1], (Image.open(f"{WIDE_INPUTS}/{index:02d}.png"),)
    prompt, source, instruction = SUITE[index % len(SUITE)]
    return (instruction, (Image.open(f"{INPUTS}/{source}"),)) if CASES[case]["edit"] else (prompt, ())


# a `packed_state` dict <-> what safetensors can hold: fp8 tensors go to disk as bytes under "<name>@f8".
def to_disk(state: dict[str, Any]) -> dict[str, Any]:
    import torch

    return {(f"{name}@f8" if tensor.dtype == torch.float8_e4m3fn else name): (tensor.view(torch.uint8) if tensor.dtype == torch.float8_e4m3fn else tensor).cpu()
            for name, tensor in state.items()}


def from_disk(state: dict[str, Any]) -> dict[str, Any]:
    import torch

    return {name.removesuffix("@f8"): (tensor.view(torch.float8_e4m3fn) if name.endswith("@f8") else tensor) for name, tensor in state.items()}


# where the prepacked dit lives on the hub: a private repo (the weights' licence forbids passing them on).
PACKED_REPO = "tensorkelechi/magpie-qwen-image-2.1-turbo-nvfp4-mxfp8-bf16"


# the prepacked dit for this build if there is one: the local file, else the private hub copy, downloaded once
# next to it (skipped offline). None means it has to be packed here.
def prepacked(local: pl.Path) -> pl.Path | None:
    from huggingface_hub import HfApi, constants, hf_hub_download

    if local.exists():
        return local
    if constants.HF_HUB_OFFLINE or not HfApi().file_exists(PACKED_REPO, local.name):
        return None
    print(f"downloading {local.name} from {PACKED_REPO}", flush=True)
    return pl.Path(hf_hub_download(PACKED_REPO, local.name, local_dir=local.parent))


# the port's dit and vae on the card, and nothing else: no pipeline, no text encoder. the checkpoint's
# safetensors are read directly and cast to bf16, so only torch and safetensors are needed. `slim` keeps the mlp
# weights in fp8 only; pass False for anything that runs the baseline arithmetic (`unfused`). a slim dit on the
# native matmul is read packed from HF_HOME/magpie/ (fetched from the private PACKED_REPO if missing, else packed
# here and saved), so later loads never hold the bf16 dit and need no transformer/ download.
def load_port(variant: str, slim: bool = True, attention: tuple[str, ...] = ("q", "k", "v", "out")) -> tuple[Any, Any]:
    import torch
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file, save_file

    import magpie.pipeline as lean
    from magpie.kernels import MX, NV

    torch.set_grad_enabled(False)
    # the cached snapshot folder; a repo id would query the hub, which is offline on modal
    snapshot = pl.Path(hf_hub_download(REPOS[variant], "model_index.json", local_files_only=True)).parent

    def read(part: str) -> dict[str, Any]:
        return {name: weight.to("cuda", torch.bfloat16)
                for file in sorted((snapshot / part).glob("*.safetensors")) for name, weight in load_file(file).items()}

    config = json.loads((snapshot / "vae" / "config.json").read_text())
    packed = pl.Path(os.environ.get("HF_HOME", pl.Path.home() / ".cache/huggingface")) / "magpie" / f"{variant}-nv_{'_'.join(sorted(NV))}-fp8_{'_'.join(attention)}.safetensors"
    ready = prepacked(packed) if slim and MX and not os.environ.get("MAGPIE_REPACK") else None
    if ready:
        return lean.adopt_packed(from_disk(load_file(ready, device="cuda")), read("vae"), config["latents_mean"], config["latents_std"])
    dit, vae = lean.adopt(read("transformer"), read("vae"), config["latents_mean"], config["latents_std"], slim, attention)
    if slim and MX:
        packed.parent.mkdir(parents=True, exist_ok=True)
        save_file(to_disk(lean.packed_state(dit)), packed)
    return dit, vae


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


# the baseline port's behaviour for a build that has since gained kernels: qk-norm + rope unfused, torch's sdpa,
# bf16 linears; with `vae_module`, the vae's norms and shortcuts as plain torch ops too.
def unfused(dit_module: Any, vae_module: Any = None) -> None:
    if vae_module is not None:
        vae_module.fusable = lambda x: False
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


# dreamsim (ensemble) distance of image pairs: 0.0 means the same picture to a model tuned on human similarity
# judgements. its weights are fetched once into HF_HOME/dreamsim.
def dreamsim_distance(pairs: list[tuple[Any, Any]]) -> list[float]:
    import torch
    from dreamsim import dreamsim

    model, preprocess = dreamsim(pretrained=True, device="cuda", cache_dir=str(pl.Path(os.environ.get("HF_HOME", ".")) / "dreamsim"))
    with torch.no_grad():
        left, right = (torch.cat([preprocess(pair[side].convert("RGB")) for pair in pairs]).to("cuda") for side in (0, 1))
        return model(left, right).tolist()


# both image scores for (candidate, frozen baseline) pairs, as the fields a case's result carries.
def image_scores(pairs: list[tuple[Any, Any]]) -> dict[str, list[float]]:
    return {"dino_vs_baseline": dino_similarity(pairs) if pairs else [], "dreamsim_vs_baseline": dreamsim_distance(pairs) if pairs else []}


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
                         "psnr_vs_baseline_db": drift, "diagnosis": checks, "step_error": errors, **image_scores(pairs)}
        print(f"gate {case}: dit+vae p50 {results[case]['total']['p50']:.3f} s, psnr min {min(scores):.1f} dB", flush=True)
    dit.prefill, dit.decode, vae.encode, vae.decode = plain
    return results


# pass / fail of a gate run: no image may fall below DINO_MIN similarity to the frozen baseline image or sit further
# than DREAMSIM_MAX from it. `flagged` lists the items that do (the ones worth a look, or the vlm judge); step
# errors on the baseline's own inputs are reported next to them but not judged.
# psnr against the official reference is reported, not judged: it cannot tell rounding from damage.
def verdict(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    cases = {}
    for case, row in candidate["magpie"].items():
        frozen = baseline["magpie"][case]
        worst_step, worst_dino, worst_dream = max(max(item) for item in row["step_error"]), min(row["dino_vs_baseline"]), max(row["dreamsim_vs_baseline"])
        steps = [error for item in row["step_error"] for error in item]
        flagged = [index for index, (dino, dream) in enumerate(zip(row["dino_vs_baseline"], row["dreamsim_vs_baseline"])) if dino < DINO_MIN or dream > DREAMSIM_MAX]
        cases[case] = {"pass": not flagged, "flagged": flagged, "worst_dino": round(worst_dino, 4), "worst_dreamsim": round(worst_dream, 4),
                       "worst_step_error": round(worst_step, 5), "mean_step_error": round(sum(steps) / len(steps), 5),
                       "steps_over_line": f"{sum(error > STEP_ERROR_MAX for error in steps)} of {len(steps)}",
                       "worst_psnr_change_db": round(min(ours - theirs for ours, theirs in zip(row["psnr_db"], frozen["psnr_db"])), 2),
                       "dit_vae_ms": round(row["total"]["p50"] * 1e3, 1), "baseline_ms": round(frozen["total"]["p50"] * 1e3, 1),
                       "speedup": round(frozen["total"]["p50"] / row["total"]["p50"], 3)}
    return {"pass": all(c["pass"] for c in cases.values()), "cases": cases}


# one gate run on this machine. with `compare`, the baseline arithmetic is timed here first and judged against.
def run(variant: str, tag: str, cases: list[str], compare: bool, sdpa: bool, attention: str = "q,k,v,out") -> dict[str, Any]:
    import torch

    import magpie.dit as dit_module

    dit, vae = load_port(variant, not compare, tuple(filter(None, attention.split(","))))
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
    parser.add_argument("--attention", default="q,k,v,out")  # the attention linears run in fp8
    args = parser.parse_args()
    result = run(args.variant, args.tag, args.cases.split(","), args.compare, args.sdpa, args.attention)
    out = pl.Path(OUT_ROOT) / "gate" / f"{args.tag}-{args.variant}.json"
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps(result["verdict"], indent=1))


if __name__ == "__main__":
    main()
