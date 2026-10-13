# re-freezes one item of the sixteen-image set after its prompts in gate.WIDE changed, with the kernels-off port and
# the bf16 encoder on this machine: its input image (text-to-image at 1024, seed 1000 + index) and, for edit16_1024
# and edit16_2048, its conditioning, baseline image, reference image (the baseline again: the official pipeline is
# not here) and step recording. the changed files then go up to the private gate dataset.
#   python tools/refreeze_item.py 15
import pathlib as pl
import sys
from typing import Any

import torch
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image

import magpie.dit as dit_module
import magpie.pipeline as lean
import magpie.vae as vae_module
from gate import CASES, OUT_ROOT, REPOS, WIDE, WIDE_INPUTS, load_port, unfused
from magpie.sampling import TURBO
from magpie.text import encode, load_encoder

DATASET = "tensorkelechi/magpie-gate"


def main(index: int) -> None:
    torch.set_grad_enabled(False)
    folder = pl.Path(OUT_ROOT) / "baseline" / "turbo"
    dit, vae = load_port("turbo", slim=False)
    unfused(dit_module, vae_module)
    encoder, processor = load_encoder(pl.Path(hf_hub_download(REPOS["turbo"], "model_index.json", local_files_only=True)).parent)
    prompt, instruction = WIDE[index]
    name = f"{index:02d}"
    lean.render(dit, vae, *encode(encoder, processor, prompt, []), TURBO, [], (1024, 1024), 1024, 1000 + index).convert("RGB").save(f"{WIDE_INPUTS}/{name}.png")
    written = [pl.Path(f"{WIDE_INPUTS}/{name}.png")]
    prefill, decode = dit.prefill, dit.decode
    record: dict[str, Any] = {}

    def spy_prefill(latents: Any, *args: Any) -> Any:
        velocity, cache = prefill(latents, *args)
        record.update(joint=latents.cpu(), latents=[], velocity=[velocity.cpu()])
        return velocity, cache

    def spy_decode(latents: Any, *args: Any) -> Any:
        velocity = decode(latents, *args)
        record["latents"].append(latents.cpu())
        record["velocity"].append(velocity.cpu())
        return velocity

    dit.prefill, dit.decode = spy_prefill, spy_decode
    for case in ("edit16_1024", "edit16_2048"):
        size = CASES[case]["size"]
        sized = lean.resized((Image.open(f"{WIDE_INPUTS}/{name}.png"),), size)
        text, slots = encode(encoder, processor, instruction, sized)
        output = lean.render(dit, vae, text, slots, TURBO, sized, None, size, index).convert("RGB")
        files = {"conditioning": ".pt", "steps": ".pt", "magpie": ".png", "reference": ".png"}
        torch.save({"text": text.cpu(), "slots": slots.cpu()}, folder / "conditioning" / case / f"{name}.pt")
        torch.save(dict(record), folder / "steps" / case / f"{name}.pt")
        output.save(folder / "magpie" / case / f"{name}.png")
        output.save(folder / "reference" / case / f"{name}.png")
        written += [folder / part / case / f"{name}{suffix}" for part, suffix in files.items()]
        print(f"froze {case} item {index}: {text.shape[1]} text tokens", flush=True)
    api = HfApi()
    for path in written:
        api.upload_file(path_or_fileobj=str(path), path_in_repo=str(path.relative_to(OUT_ROOT)), repo_id=DATASET, repo_type="dataset")
    print("uploaded", len(written), "files", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]))
