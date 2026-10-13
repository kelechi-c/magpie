import modal

HF_ROOT = "/hf"
REPOS = ("Qwen/Qwen-Image-2.1-Turbo", "Qwen/Qwen-Image-2.1")
DINO = "facebook/dinov2-base@f9e44c814b77203eaa57a6bdbbd535f21ede1415"  # the gate's image-similarity encoder
SKIP = ["assets/*", "*.md", ".gitattributes"]

image = (
    modal.Image.debian_slim("3.12")
    .uv_pip_install("huggingface_hub", "hf_transfer")
    .env({"HF_HOME": HF_ROOT, "HF_HUB_ENABLE_HF_TRANSFER": "1", "PYTHONUNBUFFERED": "1"})
)
app = modal.App("magpie-prepare", image=image)
hf_volume = modal.Volume.from_name("magpie-hf-cache", create_if_missing=True)


# downloads one repo (`name` or `name@revision`) into the shared volume; cpu only, so no gpu time is billed.
@app.function(cpu=4, memory=8192, timeout=3600, volumes={HF_ROOT: hf_volume})
def prepare(repo: str) -> str:
    from huggingface_hub import snapshot_download

    name, _, revision = repo.partition("@")
    path = snapshot_download(name, revision=revision or None, ignore_patterns=SKIP)
    hf_volume.commit()
    print(f"prepared {repo} at {path}", flush=True)
    return path


GATE_DATA = "tensorkelechi/magpie-gate"  # private dataset: the frozen gate files, so another machine can pull them
out_volume = modal.Volume.from_name("magpie-out")


# copies the turbo baseline (inputs, references, conditioning, port images, step recordings) from the out volume
# to the private dataset. generated images and activations only: no weights leave the volume.
@app.function(cpu=4, memory=8192, timeout=3600, volumes={"/out": out_volume}, secrets=[modal.Secret.from_name("huggingface")])
def publish() -> str:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(GATE_DATA, repo_type="dataset", private=True, exist_ok=True)
    api.upload_folder(folder_path="/out/baseline/turbo", path_in_repo="baseline/turbo", repo_id=GATE_DATA, repo_type="dataset",
                      allow_patterns=["*/edit_1024/*", "*/edit16_1024/*", "*/edit16_2048/*", "inputs16/*"])
    return GATE_DATA


# modal run tools/modal_prepare.py [--repos a,b]; --publish copies the gate files to the private dataset instead.
@app.local_entrypoint()
def main(repos: str = ",".join(REPOS), publish_gate: bool = False) -> None:
    if publish_gate:
        print(publish.remote())
        return
    for path in prepare.map(repos.split(",")):
        print(path)
