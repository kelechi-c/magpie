import json
import pathlib as pl
import sys

import modal

sys.path.insert(0, str(pl.Path(__file__).resolve().parent))
from gate import INPUTS, OUT_ROOT
from modal_common import DIFFUSERS, FLICKER_IMAGES, HF_ROOT, REMOTE, ROOT, VOLUMES, out_volume

SAGE = "d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5"  # thu-ml/SageAttention, apache-2.0
# both packages are built from source for sm_120 with no gpu present: sage 2 reads the target from
# TORCH_CUDA_ARCH_LIST, sage 3 asks the device, so its setup.py is told (12, 0); torch 2.14 headers need c++20.
BUILD = ("git clone https://github.com/thu-ml/SageAttention /sage && cd /sage && git checkout " + SAGE,
         "cd /sage && sed -i 's/-std=c++17/-std=c++20/g' setup.py sageattention3_blackwell/setup.py"
         " && sed -i 's/torch.cuda.get_device_capability()/(12, 0)/' sageattention3_blackwell/setup.py",
         "cd /sage && TORCH_CUDA_ARCH_LIST=12.0 pip install . --no-build-isolation",
         "cd /sage/sageattention3_blackwell && pip install . --no-build-isolation")
image = (
    modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.12")
    .apt_install("git", "build-essential")
    .uv_pip_install("torch", "torchvision", "numpy", "setuptools<75", "wheel<0.44", "packaging<24", "ninja", "triton", "transformers>=5.17", "pillow",
                    "safetensors", "huggingface_hub", "dreamsim", DIFFUSERS, "accelerate")
    .env({"CUDA_HOME": "/usr/local/cuda", "CC": "gcc", "CXX": "g++", "HF_HOME": HF_ROOT, "HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1", "PYTHONPATH": f"{REMOTE}/src"})
    .run_commands(*BUILD)
    .add_local_dir(FLICKER_IMAGES, remote_path=INPUTS, copy=True, ignore=["grid*", "*.txt"])
    .add_local_dir(str(ROOT / "src"), remote_path=f"{REMOTE}/src", ignore=["__pycache__"])
    .add_local_python_source("modal_common", "gate", "sage_bench")
)
app = modal.App("magpie-sage", image=image)


@app.function(gpu="RTX-PRO-6000", cpu=8, memory=65536, timeout=1800, retries=0)
def bench() -> str:
    from sage_bench import run

    return run()


# the gate with every unmasked attention call of the current build handed to a sage kernel: "sage2" (int8 q x k,
# fp8 weights x values, fp32 accumulators) or "sage3" (fp4). masked text runs keep torch's sdpa.
@app.function(gpu="RTX-PRO-6000", cpu=8, memory=65536, timeout=3600, retries=0, volumes=VOLUMES)
def gate(kernel: str, cases: list[str]) -> str:
    import torch
    from sageattention import sageattn_qk_int8_pv_fp8_cuda
    from sageattn3 import sageattn3_blackwell

    import magpie.dit as dit_module
    from gate import device_info, gate_pass, load_port

    by_head = lambda x: x.transpose(1, 2)
    kernels = {"sage2": lambda q, k, v: sageattn_qk_int8_pv_fp8_cuda(q, k, v, tensor_layout="NHD", qk_quant_gran="per_warp", pv_accum_dtype="fp32+fp32"),
               "sage3": lambda q, k, v: by_head(sageattn3_blackwell(by_head(q), by_head(k).clone(), by_head(v)))}
    dit_module.attend = lambda q, k, v, bound, mask=None: kernels[kernel](q, k, v) if mask is None else dit_module.sdpa(q, k, v, mask)
    dit, vae = load_port("turbo")
    result = {"variant": "turbo", "tag": kernel, "device": device_info(), "resident_dit_vae_gib": round(torch.cuda.memory_allocated() / 2**30, 2),
              "magpie": gate_pass(dit, vae, "turbo", cases, pl.Path(OUT_ROOT) / "baseline" / "turbo", pl.Path(OUT_ROOT) / "gate" / kernel / "turbo", {})}
    (pl.Path(OUT_ROOT) / "gate" / f"{kernel}-turbo.json").write_text(json.dumps(result))  # kept if the client drops
    out_volume.commit()
    return json.dumps(result)


# modal run tools/modal_sage.py -> out/kernels/sage.json
# modal run tools/modal_sage.py --mode gate [--kernels sage2,sage3] [--cases ...] -> out/gate/<kernel>-turbo.json
@app.local_entrypoint()
def main(mode: str = "bench", kernels: str = "sage2,sage3", cases: str = "edit_1024,edit16_1024") -> None:
    from gate import verdict

    if mode == "bench":
        out = ROOT / "out" / "kernels"
        out.mkdir(parents=True, exist_ok=True)
        (out / "sage.json").write_text(json.dumps(json.loads(bench.remote()), indent=1))
        return
    baseline = json.loads((ROOT / "baseline" / "turbo.json").read_text())
    for payload in gate.starmap([(kernel, cases.split(",")) for kernel in kernels.split(",")]):
        result = json.loads(payload)
        result["verdict"] = verdict(result, baseline)
        (ROOT / "out" / "gate" / f"{result['tag']}-turbo.json").write_text(json.dumps(result, indent=1))
        print(result["tag"], json.dumps(result["verdict"], indent=1))
