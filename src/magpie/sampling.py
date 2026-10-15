import math
from dataclasses import dataclass, replace

import torch
from torch import Tensor


# flow-matching sampler settings. `sigmas` set means a fixed grid shipped with the checkpoint (turbo);
# otherwise a linear grid of `steps` values is shifted towards high noise by a resolution-dependent
# amount and stretched to end at `terminal` (base). `skip` names steps (1 .. last) that reuse the previous step's
# velocity instead of running the dit: fewer transformer passes for a small change in the image.
@dataclass(frozen=True)
class SamplerConfig:
    steps: int = 40
    sigmas: tuple[float, ...] | None = None
    terminal: float | None = 0.02
    base_tokens: int = 256
    max_tokens: int = 8192
    base_shift: float = 0.5
    max_shift: float = 0.9
    skip: tuple[int, ...] = ()


BASE = SamplerConfig()
TURBO = SamplerConfig(sigmas=(1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568))
TURBO_FAST = replace(TURBO, skip=(3, 6))  # 6 of 8 dit passes (opt-in): on an l4, 16 edits stayed inside the gate lines against the 8-step images (dino min 0.9938)


# shift exponent mu, linear in the number of target tokens: more tokens -> more time at high noise.
def shift(tokens: int, cfg: SamplerConfig) -> float:
    slope = (cfg.max_shift - cfg.base_shift) / (cfg.max_tokens - cfg.base_tokens)
    return cfg.base_shift + slope * (tokens - cfg.base_tokens)


# the noise levels to visit, from 1 (pure noise) down to a final 0 (clean image): steps + 1 values.
def schedule(tokens: int, cfg: SamplerConfig) -> Tensor:
    if cfg.sigmas is not None:
        return torch.tensor([*cfg.sigmas, 0.0], dtype=torch.float32)
    grid = torch.linspace(1.0, 1.0 / cfg.steps, cfg.steps, dtype=torch.float64).float()
    mu = math.exp(shift(tokens, cfg))
    rest = 1.0 - mu / (mu + (1.0 / grid - 1.0))
    stretched = 1.0 - rest / (rest[-1] / (1.0 - cfg.terminal))
    return torch.cat([stretched, stretched.new_zeros(1)])


# one euler step along the predicted velocity. as in the reference scheduler, the step `dt * velocity` is
# formed in the model's dtype (a 0-dim fp32 `dt` does not promote a bf16 tensor) and only the sum is fp32.
# multiplying in fp32 is more exact but drifts from the reference: 53.6 -> 39.8 dB on the l4 parity image.
def euler(latents: Tensor, velocity: Tensor, sigma: Tensor, sigma_next: Tensor) -> Tensor:
    return (latents.float() + (sigma_next - sigma) * velocity).to(velocity.dtype)
