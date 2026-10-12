import json
import pathlib as pl
import sys

import modal

sys.path.insert(0, str(pl.Path(__file__).resolve().parent))
from kernel_bench import bench_kernel
from modal_common import ROOT, VOLUMES, image

app = modal.App("magpie-kernels", image=image.add_local_python_source("kernel_bench"))


@app.function(gpu="L4", cpu=4, memory=16384, timeout=900, retries=0)
def check(kernel: str) -> str:
    return bench_kernel(kernel, "check")


@app.function(gpu="RTX-PRO-6000", cpu=8, memory=65536, timeout=1800, retries=0, volumes=VOLUMES)
def sweep(kernel: str) -> str:
    return bench_kernel(kernel, "sweep")


@app.local_entrypoint()
def main(mode: str = "check", kernel: str = "attention") -> None:
    out = ROOT / "out" / "kernels"
    out.mkdir(parents=True, exist_ok=True)
    payload = (sweep if mode == "sweep" else check).remote(kernel)
    (out / f"{kernel}_{mode}.json").write_text(json.dumps(json.loads(payload), indent=1))
