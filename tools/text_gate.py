# the text encoder in one format against the gate, with the current dit + vae: `python tools/text_gate.py <tag>
# [--mlp bf16|mxfp8|nvfp4] [--attention bf16|mxfp8|nvfp4] [--case edit16_1024|edit16_2048] [--copy 1024]` on the gpu
# machine -> /out/textgate/<case>/<tag>.json. `--copy N` shows the encoder an N^2-pixel copy of each condition image
# while the vae still reads the full one.
# reports load time and peak for the dit (packed file or not) and the encoder, the encoder's resident size, its
# latency and peak at a 1024 and a 2048 edit (median of 5 warm), and over the case's sixteen edits: the error of its
# states against the frozen conditioning, and dinov2 / dreamsim of the rendered image against the frozen baseline
# image and, when a `bf16` run exists, against that run's image (same dit, so the encoder's effect alone).
import argparse
import json
import pathlib as pl
import statistics
import time
from typing import Any, Callable

import torch
from PIL import Image

import magpie.pipeline as lean
from gate import CASES, DINO_MIN, DREAMSIM_MAX, OUT_ROOT, REPOS, gate_input, image_scores, load_port
from magpie.sampling import TURBO
from magpie.text import encode, load_encoder, reslot

GIB = 2**30


# fn() -> (result, seconds, peak allocated gib during it, allocated gib it left behind).
def measured(fn: Callable[[], Any]) -> tuple[Any, float, float, float]:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before, start = torch.cuda.memory_allocated(), time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    return result, time.perf_counter() - start, torch.cuda.max_memory_allocated() / GIB, (torch.cuda.memory_allocated() - before) / GIB


def main() -> None:
    from huggingface_hub import hf_hub_download

    parser = argparse.ArgumentParser()
    parser.add_argument("tag")
    parser.add_argument("--mlp", default="bf16")
    parser.add_argument("--attention", default="bf16")
    parser.add_argument("--case", default="edit16_1024")
    parser.add_argument("--copy", type=int, default=0)
    parser.add_argument("--calibrated", default="")  # repo id of a calibrated nvfp4 checkpoint already in the hf cache
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    out = pl.Path(OUT_ROOT) / "textgate" / args.case

    # text states and slots for a request at `size`, the encoder reading `--copy`-sized images when that is set.
    def conditioning(prompt: str, sized: list[Any]) -> tuple[Any, Any]:
        if not args.copy:
            return encode(encoder, processor, prompt, sized)
        text, slots = encode(encoder, processor, prompt, lean.resized(tuple(sized), args.copy))
        return reslot(text, slots, [image.width * image.height // 1024 for image in sized])

    (out / args.tag).mkdir(parents=True, exist_ok=True)
    snapshot = pl.Path(hf_hub_download(REPOS["turbo"], "model_index.json", local_files_only=True)).parent
    (dit, vae), dit_s, dit_peak, dit_gib = measured(lambda: load_port("turbo"))
    theirs = pl.Path(hf_hub_download(args.calibrated, "config.json", local_files_only=True)).parent if args.calibrated else None
    (encoder, processor), encoder_s, encoder_peak, encoder_gib = measured(lambda: load_encoder(snapshot, args.mlp, args.attention, theirs))
    report: dict[str, Any] = {"mlp": args.mlp, "attention": args.attention, "calibrated": args.calibrated, "case": args.case, "copy": args.copy, "device": torch.cuda.get_device_name(0),
                              "dit_vae": {"load_s": round(dit_s, 1), "load_peak_gib": round(dit_peak, 2), "resident_gib": round(dit_gib, 2)},
                              "encoder": {"load_s": round(encoder_s, 1), "load_peak_gib": round(encoder_peak, 2), "resident_gib": round(encoder_gib, 2)}}
    for case, size in (("edit_1024", 1024), ("edit16_2048", 2048)):
        prompt, images = gate_input(case, 0)
        sized = lean.resized(images, size)
        runs = [measured(lambda: conditioning(prompt, sized)) for _ in range(7)][2:]
        report["encoder"][f"ms_{size}"] = round(statistics.median(run[1] for run in runs) * 1e3, 1)
        report["encoder"][f"peak_gib_{size}"] = round(max(run[2] for run in runs), 2)
        report["encoder"][f"tokens_{size}"] = int(runs[0][0][0].shape[1])
    print(json.dumps(report), flush=True)
    frozen = pl.Path(OUT_ROOT) / "baseline" / "turbo"
    errors, pairs, twins = [], [], []
    size = CASES[args.case]["size"]
    for index in range(16):
        prompt, images = gate_input(args.case, index)
        sized = lean.resized(images, size)
        text, slots = conditioning(prompt, sized)
        saved = torch.load(frozen / "conditioning" / args.case / f"{index:02d}.pt")
        kept = ~saved["slots"].bool().to(text.device)  # the states the dit reads: everything outside the image slots
        ours, theirs = text[0, ~slots.bool()].float(), saved["text"].to(text.device)[0, kept].float()
        errors.append(float((ours - theirs).norm() / theirs.norm()) if ours.shape == theirs.shape else -1.0)
        image = lean.render(dit, vae, text, slots, TURBO, sized, None, size, index).convert("RGB")
        image.save(out / args.tag / f"{index:02d}.png")
        pairs.append((image, Image.open(frozen / "magpie" / args.case / f"{index:02d}.png")))
        twin = out / "bf16" / f"{index:02d}.png"
        twins += [(image, Image.open(twin))] if twin.exists() and args.tag != "bf16" else []
    del encoder, dit, vae
    torch.cuda.empty_cache()
    scores, same_dit = image_scores(pairs), image_scores(twins)
    report["quality"] = {"state_error_mean": round(statistics.mean(errors), 4), "state_error_worst": round(max(errors), 4),
                         "dino_min_vs_frozen": round(min(scores["dino_vs_baseline"]), 4), "dreamsim_max_vs_frozen": round(max(scores["dreamsim_vs_baseline"]), 4),
                         "flagged_vs_frozen": [i for i, (d, s) in enumerate(zip(scores["dino_vs_baseline"], scores["dreamsim_vs_baseline"])) if d < DINO_MIN or s > DREAMSIM_MAX],
                         **({"dino_min_vs_bf16_encoder": round(min(same_dit["dino_vs_baseline"]), 4), "dreamsim_max_vs_bf16_encoder": round(max(same_dit["dreamsim_vs_baseline"]), 4),
                             "flagged_vs_bf16_encoder": [i for i, (d, s) in enumerate(zip(same_dit["dino_vs_baseline"], same_dit["dreamsim_vs_baseline"])) if d < DINO_MIN or s > DREAMSIM_MAX]}
                            if twins else {})}
    print(json.dumps(report["quality"]), flush=True)
    (out / f"{args.tag}.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
