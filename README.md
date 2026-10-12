# magpie

Qwen-Image 2.1 Turbo image editing (8 steps, 7.1 B DiT + VAE) made fast on Blackwell GPUs: a lean PyTorch
port with hand-written Triton kernels.

| Turbo edit, DiT + VAE, RTX PRO 6000, 5 warm requests | 1024² | 2048² |
|---|---|---|
| port, PyTorch ops only | 2,484 ms | 15,298 ms |
| + fused QK-norm / RoPE, max-free flash attention | 2,245 ms | |
| + FP8 MLP matmuls | 2,023 ms | |
| + fused VAE, K/V buffer, fused block glue | 1,832 ms | 12,167 ms |

Every row passes the gate: per-step DiT error ≤ 2 % on the baseline's own inputs, DINOv2 similarity ≥ 0.98
to the baseline image. Resident memory 13.9 → 9.4 GiB.

## layout

- `src/magpie/` — the port (`dit.py`, `vae.py`, `pipeline.py`, `sampling.py`, `text.py`) and `kernels.py`
- `tools/` — the gate (`gate.py`), kernel benchmarks, quantisation probes, Modal and Colab launchers
- `docs/` — `project.md` (code map), `session_handoff.md` (state and commands), `notes.md`, `lessons.md`,
  `predictions.md`, and the write-up `qwen-image-2-1-blackwell.md`

## run

```
modal run tools/modal_prepare.py                                   # weights -> a modal volume
modal run tools/modal_bench.py --mode gate --tag <name> --variants turbo
```

## license

No weights are in this repository. Qwen-Image 2.1 weights are under the Qwen Research License
(non-commercial); do not redistribute them, and keep that notice with any derived checkpoint.
