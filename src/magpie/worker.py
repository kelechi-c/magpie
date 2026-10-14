"""runpod serverless worker for magpie: the http server's request, run as a queue job.

    python -u -m magpie.worker

a job's `input` is what `POST /generate` takes (see magpie.server); the output is its reply, JPEG unless the
input asks otherwise (a PNG at 2048² is close to the size a job's output may have). the model loads once per
worker, before the first job is taken.
"""

from typing import Any

import runpod
import torch

from magpie import server

RUNTIME = server.load()
PARKED = [False]  # whether the last job left the encoder in host memory


# one job -> the reply, or {"error": ...}, which runpod reports as a failed job.
def handle(job: dict[str, Any]) -> dict[str, Any]:
    given = job["input"]
    if not isinstance(given.get("prompt"), str) or not given["prompt"].strip():
        return {"error": "prompt is required"}
    try:
        body, PARKED[0] = server.generate(RUNTIME, {"format": "jpeg", **given}, PARKED[0])
    except torch.OutOfMemoryError as error:  # the one failure a caller can act on: ask for less
        message = str(error)[:200]
        del error
        RUNTIME.park()
        PARKED[0] = RUNTIME.parks
        return {"error": f"out of gpu memory: {message}"}
    print(f"{body['width']}x{body['height']} {len(given.get('images', []))} images {body['ms']} peak {body['peak_gib']} gib", flush=True)
    return body


if __name__ == "__main__":
    runpod.serverless.start({"handler": handle})
