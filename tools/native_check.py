# the native block-scaled matmul against torch on the card: `python tools/native_check.py`.
# x gets a different magnitude in every block of 32 and every row, so a scale read from the wrong slot shows as a
# large error; a correct layout leaves only fp8 rounding (around 1-2 %). then ms and tflop/s at the dit's shapes,
# and the whole mlp branch against its bf16 reference.
import json
import time

import torch
import torch.nn.functional as F

from magpie.kernels import mlp_mx, mx_columns, mx_norm, mx_project, mx_rows, pack_mlp_mx


def bench(fn, iters: int = 20) -> float:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1e3


def main() -> None:
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    report = {}
    for m, n, k in ((4096, 4096, 4096), (4096, 24576, 4096), (4096, 4096, 12288), (8239, 4096, 4096), (300, 4096, 4096)):
        spread = torch.exp2(torch.randint(-6, 7, (m, k // 32, 1), device="cuda").float())
        x = (torch.randn(m, k // 32, 32, device="cuda") * spread).flatten(1).bfloat16()
        w = (torch.randn(n, k, device="cuda") * 0.02 * torch.exp2(torch.randint(-3, 4, (n, 1), device="cuda").float())).bfloat16()
        want = x.float() @ w.float().t()
        rows, linear = mx_rows(x), mx_columns(w)
        got = mx_project(*rows, linear).float()
        ms = bench(lambda: mx_project(*rows, linear))
        report[f"{m}x{n}x{k}"] = {"gap": float((got - want).norm() / want.norm()), "ms": ms, "tflops": 2 * m * n * k / ms / 1e9,
                                  "bf16_ms": bench(lambda: F.linear(x, w)), "quantise_ms": bench(lambda: mx_rows(x))}
        print(f"{m}x{n}x{k}", json.dumps(report[f"{m}x{n}x{k}"]), flush=True)
    tokens, width, hidden = 4096, 4096, 12288
    x = torch.randn(1, tokens, width, device="cuda").bfloat16()
    mod, gate = (1 + 0.1 * torch.randn(1, 1, width, device="cuda")).bfloat16(), torch.randn(1, 1, width, device="cuda").mul(0.3).bfloat16()
    g, p, o = (torch.randn(shape, device="cuda").mul(0.02).bfloat16() for shape in ((hidden, width), (hidden, width), (width, hidden)))
    n = F.layer_norm(x.float(), (width,), eps=1e-6) * mod.float()
    want = x.float() + gate.float() * F.linear(F.silu(F.linear(n, g.float())) * F.linear(n, p.float()), o.float())
    packed = pack_mlp_mx(g, p, o)
    got = mlp_mx(x, mod, gate, packed, 1e-6).float()
    flat, flat_mod = x.flatten(0, 1), mod.flatten(0, 1).contiguous()
    rows = mx_norm(flat, flat_mod, 1e-6)
    raw = mx_project(*rows, packed.hidden)
    hidden_rows = mx_rows(raw, swiglu=True)
    report["mlp"] = {"branch_gap": float(((got - x.float()) - (want - x.float())).norm() / (want - x.float()).norm()), "ms": bench(lambda: mlp_mx(x, mod, gate, packed, 1e-6)),
                     "norm_ms": bench(lambda: mx_norm(flat, flat_mod, 1e-6)), "hidden_matmul_ms": bench(lambda: mx_project(*rows, packed.hidden)),
                     "swiglu_ms": bench(lambda: mx_rows(raw, swiglu=True)), "out_matmul_ms": bench(lambda: mx_project(*hidden_rows, packed.out))}
    print("mlp", json.dumps(report["mlp"]), flush=True)


if __name__ == "__main__":
    main()
