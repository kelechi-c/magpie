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


@app.local_entrypoint()
def main(repos: str = ",".join(REPOS)) -> None:
    for path in prepare.map(repos.split(",")):
        print(path)
