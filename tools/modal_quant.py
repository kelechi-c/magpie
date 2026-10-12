import json
import pathlib as pl
import sys

import modal

sys.path.insert(0, str(pl.Path(__file__).resolve().parent))
from modal_common import ROOT, VOLUMES, image

app = modal.App("magpie-quant", image=image.add_local_python_source("quant_probe"))


@app.function(gpu="RTX-PRO-6000", cpu=4, memory=16384, timeout=1800, retries=0)
def run_speed() -> str:
    from quant_probe import speed

    return speed()


@app.function(gpu="RTX-PRO-6000", cpu=8, memory=65536, timeout=1800, retries=0, volumes=VOLUMES)
def run_quality() -> str:
    from quant_probe import quality

    return quality()


@app.function(gpu="RTX-PRO-6000", cpu=8, memory=65536, timeout=1800, retries=0, volumes=VOLUMES)
def run_ranges() -> str:
    from quant_probe import ranges

    return ranges()


@app.function(gpu="RTX-PRO-6000", cpu=8, memory=65536, timeout=1800, retries=0, volumes=VOLUMES)
def run_ablate() -> str:
    from quant_probe import ablate

    return ablate()


@app.function(gpu="RTX-PRO-6000", cpu=8, memory=65536, timeout=3600, retries=0, volumes=VOLUMES)
def run_inside(pick: str = "", case: str = "edit_1024") -> str:
    from quant_probe import inside

    return inside(pick, case)


# modal run tools/modal_quant.py [--only speed,quality,ranges,ablate,inside]: the chosen probes at once -> out/quant/<name>.json
@app.local_entrypoint()
def main(only: str = "speed,quality", pick: str = "", case: str = "edit_1024") -> None:
    out = ROOT / "out" / "quant"
    out.mkdir(parents=True, exist_ok=True)
    names = only.split(",")
    calls = [{"speed": run_speed, "quality": run_quality, "ranges": run_ranges, "ablate": run_ablate, "inside": run_inside}[name].spawn(*((pick, case) if name == "inside" else ())) for name in names]
    for name, payload in zip(names, modal.FunctionCall.gather(*calls)):
        (out / f"{name}.json").write_text(json.dumps(json.loads(payload), indent=1))
