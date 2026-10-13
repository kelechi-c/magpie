# uploads this machine's prepacked dit (HF_HOME/magpie/turbo-*.safetensors, written by gate.load_port) to the private
# repo gate.PACKED_REPO, with the licence notice. `python tools/publish_packed.py`
import os
import pathlib as pl

from huggingface_hub import HfApi

from gate import PACKED_REPO

NOTICE = """---
license: other
license_name: qwen-research
base_model: Qwen/Qwen-Image-2.1-Turbo
---
# magpie: Qwen-Image 2.1 Turbo DiT, prepacked (NVFP4 + MXFP8 + bf16)

Private. Derived from `Qwen/Qwen-Image-2.1-Turbo`, which is under the **Qwen Research License (non-commercial)**:
do not make this repo public or pass the file on.

One safetensors file per build, read by `magpie` (`tools/gate.py: load_port`) on sm_120 cards:

- MLP hidden and output matmuls: NVFP4 (e2m1 values, one fp8 scale per 16)
- attention q, k, v, out: MXFP8 (e4m3 values, one power-of-two scale per 32), stored as bytes under `<name>@f8`
- everything else: bf16

Scales are in CUTLASS's slot order for 128-wide tiles. The VAE and text encoder still come from the base repo.
"""

if __name__ == "__main__":
    folder = pl.Path(os.environ.get("HF_HOME", pl.Path.home() / ".cache/huggingface")) / "magpie"
    api = HfApi()
    api.create_repo(PACKED_REPO, private=True, exist_ok=True)
    api.upload_file(path_or_fileobj=NOTICE.encode(), path_in_repo="README.md", repo_id=PACKED_REPO)
    for file in sorted(folder.glob("turbo-*.safetensors")):
        api.upload_file(path_or_fileobj=str(file), path_in_repo=file.name, repo_id=PACKED_REPO)
        print(f"uploaded {file.name}: {file.stat().st_size / 2**30:.2f} gib", flush=True)
