import modal

HF_ROOT = "/hf"
REPOS = ("Qwen/Qwen-Image-2.1-Turbo", "Qwen/Qwen-Image-2.1")
SKIP = ["assets/*", "*.md", ".gitattributes"]

image = (
    modal.Image.debian_slim("3.12")
    .uv_pip_install("huggingface_hub", "hf_transfer")
    .env({"HF_HOME": HF_ROOT, "HF_HUB_ENABLE_HF_TRANSFER": "1", "PYTHONUNBUFFERED": "1"})
)
app = modal.App("magpie-prepare", image=image)
hf_volume = modal.Volume.from_name("magpie-hf-cache", create_if_missing=True)


# downloads one checkpoint into the shared volume; cpu only, so no gpu time is billed for the fetch.
@app.function(cpu=4, memory=8192, timeout=3600, volumes={HF_ROOT: hf_volume})
def prepare(repo: str) -> str:
    from huggingface_hub import snapshot_download

    path = snapshot_download(repo, ignore_patterns=SKIP)
    hf_volume.commit()
    print(f"prepared {repo} at {path}", flush=True)
    return path


@app.local_entrypoint()
def main() -> None:
    for path in prepare.map(REPOS):
        print(path)
