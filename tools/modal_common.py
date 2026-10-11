import pathlib as pl

import modal

ROOT = pl.Path(__file__).resolve().parent.parent
REMOTE = "/root/magpie"
HF_ROOT = "/hf"
OUT_ROOT = "/out"
INPUTS = "/root/inputs"
FLICKER_IMAGES = "/home/tensor/Shared/code/ml/flicker/out/article/images"
DIFFUSERS = "git+https://github.com/huggingface/diffusers@85c9fa4caee33d1c5b473395784471d2d7957c1b"
REPOS = {"turbo": "Qwen/Qwen-Image-2.1-Turbo", "base": "Qwen/Qwen-Image-2.1"}

image = (
    modal.Image.debian_slim("3.12")
    .apt_install("git")
    .uv_pip_install("torch", "torchvision", "transformers>=5.17", DIFFUSERS, "accelerate", "pillow", "safetensors",
                    "huggingface_hub", "numpy")
    .env({"HF_HOME": HF_ROOT, "HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1", "PYTHONPATH": f"{REMOTE}/src"})
    .add_local_dir(FLICKER_IMAGES, remote_path=INPUTS, copy=True, ignore=["grid*", "*.txt"])
    .add_local_dir(str(ROOT / "src"), remote_path=f"{REMOTE}/src", ignore=["__pycache__"])
    .add_local_python_source("modal_common")
)
hf_volume = modal.Volume.from_name("magpie-hf-cache", create_if_missing=True)
out_volume = modal.Volume.from_name("magpie-out", create_if_missing=True)
VOLUMES = {HF_ROOT: hf_volume, OUT_ROOT: out_volume}
