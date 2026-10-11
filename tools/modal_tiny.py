import json
import sys
import pathlib as pl

import modal

sys.path.insert(0, str(pl.Path(__file__).resolve().parent))
from modal_common import image

app = modal.App("magpie-tiny", image=image)


# largest absolute difference relative to the reference's largest magnitude.
def gap(ours, theirs) -> float:
    return float((ours.double() - theirs.double()).abs().max() / theirs.double().abs().max())


# cpu-only check of the lean port against the diffusers classes, with tiny random fp32 weights.
# covers text-to-image and two-reference editing layouts, with and without the kv cache, and the vae.
@app.function(cpu=4, memory=8192, timeout=900)
def tiny() -> str:
    import torch
    from diffusers import AutoencoderKLQwenImage21, QwenImage21Transformer2DModel
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache

    from magpie.dit import Dit, DitConfig, layout
    from magpie.vae import Vae, VaeConfig, image_weights

    torch.manual_seed(0)
    report: dict[str, float] = {}
    reference = QwenImage21Transformer2DModel(num_layers=3, num_attention_heads=2, context_in_dim=48).eval()
    for parameter in reference.parameters():
        torch.nn.init.normal_(parameter, std=0.05)
    cfg = DitConfig(width=256, heads=2, layers=3, context_dim=48)
    ours = Dit(cfg).eval()
    ours.load_state_dict(reference.state_dict())

    cases = {"t2i": ([False] * 9, [(4, 6)]),
             "edit": ([False] * 3 + [True] * 6 + [False] * 2 + [True] * 4 + [False] * 5, [(6, 4), (4, 4), (8, 6)])}
    with torch.no_grad():
        for name, (slot_list, shapes) in cases.items():
            slots = torch.tensor(slot_list)
            target = shapes[-1][0] * shapes[-1][1]
            tokens = sum(h * w for h, w in shapes)
            text = torch.randn(1, len(slot_list), 48)
            latents = torch.randn(1, tokens, 64)
            sigma = torch.tensor([0.7])
            img_mask = torch.cat([slots, torch.ones(target // 4, dtype=torch.bool)])[None]
            img_shapes = [[(1, h, w) for h, w in shapes]]
            common = dict(encoder_hidden_states=text, img_shapes=img_shapes, img_mask=img_mask, return_dict=False)
            plain = reference(hidden_states=latents, timestep=sigma, **common)[0][:, -target:]
            cache = QwenImage21KVCache(3)
            first = reference(hidden_states=latents, timestep=sigma, kv_cache=cache, kv_cache_mode="extract", **common)[0][:, -target:]
            later = reference(hidden_states=latents, timestep=sigma * 0.5, kv_cache=cache, kv_cache_mode="cached", **common)[0]

            lay = layout(slots, shapes, cfg)
            mine_first, mine_cache = ours.prefill(latents, text, sigma, lay)
            mine_later = ours.decode(latents[:, -target:], sigma * 0.5, lay, mine_cache)
            report[f"dit_{name}_prefill_vs_plain"] = gap(mine_first, plain)
            report[f"dit_{name}_prefill_vs_extract"] = gap(mine_first, first)
            report[f"dit_{name}_decode_vs_cached"] = gap(mine_later, later)

        vae_ref = AutoencoderKLQwenImage21(base_dim=8, decoder_base_dim=12, z_dim=64).eval()
        for parameter in vae_ref.parameters():
            torch.nn.init.normal_(parameter, std=0.05)
        mean, std = vae_ref.config.latents_mean, vae_ref.config.latents_std
        vae = Vae(VaeConfig(encoder_dim=8, decoder_dim=12), mean, std).eval()
        vae.load_state_dict(image_weights(vae_ref.state_dict()))
        pixels = torch.rand(1, 4, 96, 64) * 2 - 1
        scale = lambda t: t.view(1, -1, 1, 1)
        ref_latents = (vae_ref.encode(pixels[:, :, None]).latent_dist.mode()[:, :, 0] - scale(torch.tensor(mean))) / scale(torch.tensor(std))
        report["vae_encode"] = gap(vae.encode(pixels), ref_latents)
        z = torch.randn(1, 64, 6, 4)
        ref_pixels = vae_ref.decode((z * scale(torch.tensor(std)) + scale(torch.tensor(mean)))[:, :, None], return_dict=False)[0][:, :, 0]
        report["vae_decode"] = gap(vae.decode(z), ref_pixels)
        report["vae_decode_range"] = float(ref_pixels.abs().max())
    return json.dumps(report)


@app.local_entrypoint()
def main() -> None:
    for name, value in json.loads(tiny.remote()).items():
        print(f"{name}: {value:.3e}")
