import math
from typing import NamedTuple

import torch
import triton
import triton.language as tl
from torch import Tensor

ROWS = 32  # (token, head) rows per program; must divide the head count, so a program never runs off the end
WARPS = 4  # 32-thread groups per program
BLOCK_M, BLOCK_N, ATTENTION_WARPS, STAGES = 256, 32, 8, 3  # queries and keys per tile, warps, prefetch depth (swept on sm_120)
EXP_LIMIT = 88.0  # exp overflows fp32 just above this
FP8_TOP = 448.0  # largest fp8 (e4m3) value: 4 exponent bits, 3 mantissa bits
TOP = tl.constexpr(FP8_TOP)
SWIGLU_TILE, LINEAR_TILE, GEMM_WARPS = (64, 128, 64), (64, 128, 128), 4  # (rows, columns, inner width) per program of each fp8 matmul (swept on sm_120)


# bf16 bits (int16) -> the same number in fp32: bf16 is the top half of an fp32, so shift it into place.
@triton.jit
def widen(bits):
    return (bits.to(tl.int32) << 16).to(tl.float32, bitcast=True)


# fp32 -> the 16 bits of the nearest bf16, ties to even, exactly as torch's `.to(bfloat16)` rounds.
@triton.jit
def nearest(value):
    bits = value.to(tl.uint32, bitcast=True)
    return (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16


# fp32 -> fp32 holding a bf16 value: what a bf16 tensor would contain, kept wide for the next step.
@triton.jit
def rounded(value):
    return (nearest(value) << 16).to(tl.float32, bitcast=True)


# qk-norm and rope for ROWS rows of one projection output, in a single pass over memory.
# x and out are (rows, 2 * HALF) bf16 seen as int16, row = token * heads + head; weight is (2 * HALF,)
# bf16 as int16; rope is (tokens, HALF, 2) fp32 holding cos and sin. each row is read once as HALF
# (even, odd) pairs, reduced to its rms, scaled, multiplied by the weight, rotated pair by pair and
# written once. the two `rounded` calls reproduce the bf16 roundings of the unfused code, and the rotation
# uses the same fused multiply-add torch's complex product compiles to, which makes that stage bit-identical.
# what still differs is the order the squares are summed in (about one output in a million).
@triton.jit
def qk_prep_kernel(x_ptr, out_ptr, weight_ptr, rope_ptr, heads, tokens, eps, ROWS: tl.constexpr, HALF: tl.constexpr):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    pairs = 2 * tl.arange(0, HALF)
    at = rows[:, None] * (2 * HALF) + pairs[None, :]
    a, b = widen(tl.load(x_ptr + at)), widen(tl.load(x_ptr + at + 1))
    scale = tl.rsqrt(tl.sum(a * a + b * b, axis=1) / (2 * HALF) + eps)[:, None]
    a = rounded(rounded(a * scale) * widen(tl.load(weight_ptr + pairs))[None, :])
    b = rounded(rounded(b * scale) * widen(tl.load(weight_ptr + pairs + 1))[None, :])
    phase = ((rows // heads) % tokens)[:, None] * (2 * HALF) + pairs[None, :]
    cos, sin = tl.load(rope_ptr + phase), tl.load(rope_ptr + phase + 1)
    tl.store(out_ptr + at, nearest(tl.fma(a, cos, -(b * sin))).to(tl.int16))
    tl.store(out_ptr + at + 1, nearest(tl.fma(a, sin, b * cos)).to(tl.int16))


# fused `rotate(rms(x, weight, eps), rope)` for a contiguous bf16 x of shape (B, S, heads, head_dim) and
# complex rope of shape (S, head_dim / 2). one kernel launch instead of about ten, one read and one write.
# `out` may be a contiguous slice of a larger buffer.
def qk_prep(x: Tensor, weight: Tensor, rope: Tensor, eps: float, out: Tensor | None = None, rows: int = ROWS, warps: int = WARPS) -> Tensor:
    out = torch.empty_like(x) if out is None else out
    tokens, heads, dim = x.shape[1:]
    grid = (x.numel() // dim // rows,)
    qk_prep_kernel[grid](x.view(torch.int16), out.view(torch.int16), weight.view(torch.int16), torch.view_as_real(rope),
                         heads, tokens, eps, ROWS=rows, HALF=dim // 2, num_warps=warps)
    return out


# largest |q.k| / sqrt(d) the qk-norm weights allow: after the norm a row has length sqrt(d) times at most
# max|weight|, rope keeps lengths, and a dot product is at most the product of the two lengths.
def logit_bound(weight_q: Tensor, weight_k: Tensor) -> float:
    return math.sqrt(weight_q.numel()) * float(weight_q.float().abs().max()) * float(weight_k.float().abs().max())


# true when exp(logit) can be summed over `keys` keys without overflowing fp32, so no running max is needed.
def max_free(bound: float, keys: int) -> bool:
    return bound + math.log(keys) < EXP_LIMIT


# flash attention for one head and BLOCK_M queries. q, k, v are (batch, tokens, heads, DIM) bf16; out is the
# same shape as q, seen as int16. the keys are walked BLOCK_N at a time: scores = q k^T on the tensor cores,
# exponentiated, then p v on the tensor cores again, accumulated in fp32. the score matrix never reaches memory.
# `scale` is log2(e) / sqrt(DIM): multiplied into q once, it both scales the logits and turns exp into the cheaper exp2.
# STABLE keeps a running maximum per query and rescales the accumulators whenever it grows (online softmax).
# without it the scores are exponentiated as they are, which is only legal when they cannot overflow.
@triton.jit
def attend_kernel(q_ptr, k_ptr, v_ptr, out_ptr, queries, keys, heads, scale,
                  DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, STABLE: tl.constexpr):
    head, batch = tl.program_id(1) % heads, tl.program_id(1) // heads
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    dims = tl.arange(0, DIM)
    stride = heads * DIM
    mine = (batch * queries + rows[:, None]) * stride + head * DIM + dims[None, :]
    q = tl.load(q_ptr + mine, mask=rows[:, None] < queries, other=0.0)
    q = rounded(q.to(tl.float32) * scale).to(k_ptr.dtype.element_ty)
    acc = tl.zeros((BLOCK_M, DIM), dtype=tl.float32)
    total = tl.zeros((BLOCK_M,), dtype=tl.float32)
    top = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    for start in range(0, keys, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        live = cols < keys
        theirs = (batch * keys + cols[:, None]) * stride + head * DIM + dims[None, :]
        k = tl.load(k_ptr + theirs, mask=live[:, None], other=0.0)
        v = tl.load(v_ptr + theirs, mask=live[:, None], other=0.0)
        scores = tl.dot(q, tl.trans(k))
        if STABLE:
            scores = tl.where(live[None, :], scores, float("-inf"))
            higher = tl.maximum(top, tl.max(scores, axis=1))
            shrink = tl.exp2(top - higher)
            p = tl.exp2(scores - higher[:, None]).to(v.dtype)
            total = total * shrink + tl.sum(p.to(tl.float32), axis=1)
            acc = acc * shrink[:, None]
            top = higher
        else:
            p = tl.where(live[None, :], tl.exp2(scores), 0.0).to(v.dtype)
            total += tl.sum(p.to(tl.float32), axis=1)
        acc = tl.dot(p, v, acc)
    tl.store(out_ptr + mine, nearest(acc / total[:, None]).to(tl.int16), mask=rows[:, None] < queries)


# softmax(q k^T / sqrt(d)) v for contiguous bf16 (B, S, heads, head_dim) tensors, no mask. `stable` picks the
# online-softmax kernel; pass False only where `max_free` holds.
def flash(q: Tensor, k: Tensor, v: Tensor, stable: bool, block_m: int = BLOCK_M, block_n: int = BLOCK_N,
          warps: int = ATTENTION_WARPS, stages: int = STAGES) -> Tensor:
    batch, queries, heads, dim = q.shape
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    out = torch.empty_like(q)
    grid = (triton.cdiv(queries, block_m), batch * heads)
    attend_kernel[grid](q, k, v, out.view(torch.int16), queries, k.shape[1], heads, math.log2(math.e) / math.sqrt(dim),
                        DIM=dim, BLOCK_M=block_m, BLOCK_N=block_n, STABLE=stable, num_warps=warps, num_stages=stages)
    return out


# one swiglu mlp ready for fp8 matmuls: each weight transposed to (in, out) and divided by a per-output step
# so the largest entry of every output column is FP8_TOP (weight = fp8 * step).
class Fp8Mlp(NamedTuple):
    gate: Tensor
    proj: Tensor
    out: Tensor
    gate_step: Tensor
    proj_step: Tensor
    out_step: Tensor


# (out, in) weight -> ((in, out) fp8 weight, (out,) fp32 step).
def fp8_columns(weight: Tensor) -> tuple[Tensor, Tensor]:
    step = weight.float().abs().amax(1) / FP8_TOP
    return (weight.float() / step[:, None]).t().contiguous().to(torch.float8_e4m3fn), step


def pack_mlp(gate: Tensor, proj: Tensor, out: Tensor) -> Fp8Mlp:
    (g, g_step), (p, p_step), (o, o_step) = fp8_columns(gate), fp8_columns(proj), fp8_columns(out)
    return Fp8Mlp(g, p, o, g_step, p_step, o_step)


# layer norm and modulation of one row. x is (rows, WIDTH) bf16; mod is (mod_rows, WIDTH) bf16 with mod_rows 1 or
# the token count. FP8 writes the row as fp8 with its own step (row = fp8 * step); otherwise it is written as bf16.
@triton.jit
def norm_kernel(x_ptr, mod_ptr, out_ptr, step_ptr, mod_rows, eps, WIDTH: tl.constexpr, FP8: tl.constexpr):
    row = tl.program_id(0)
    at = tl.arange(0, WIDTH)
    x = tl.load(x_ptr + row * WIDTH + at).to(tl.float32)
    x = x - tl.sum(x, axis=0) / WIDTH
    y = x * tl.rsqrt(tl.sum(x * x, axis=0) / WIDTH + eps) * tl.load(mod_ptr + (row % mod_rows) * WIDTH + at).to(tl.float32)
    if FP8:
        step = tl.maximum(tl.max(tl.abs(y), axis=0), 1e-30) / TOP
        tl.store(out_ptr + row * WIDTH + at, tl.clamp(y / step, -TOP, TOP).to(tl.float8e4nv))
        tl.store(step_ptr + row, step)
    else:
        tl.store(out_ptr + row * WIDTH + at, y.to(tl.bfloat16))


# both input matmuls of the swiglu for one (BM, BN) tile, then silu(gate) * proj in registers, written as fp8.
# x is (M, K) fp8 with one step per row; gate and proj are (K, N) fp8 with one step per column. each row of the
# tile gets its own step (its largest value / TOP), stored at out_step[row, tile column]: hidden values span four
# orders of magnitude between tokens, more than one fp8 scale can hold. the hidden tensor is never held in 16 bits.
@triton.jit
def swiglu_fp8_kernel(x_ptr, step_ptr, gate_ptr, proj_ptr, gate_step_ptr, proj_step_ptr, out_ptr, out_step_ptr, M, N, K,
                      BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    live = rows < M
    gate = tl.zeros((BM, BN), dtype=tl.float32)
    proj = tl.zeros((BM, BN), dtype=tl.float32)
    for start in range(0, K, BK):
        ks = start + tl.arange(0, BK)
        a = tl.load(x_ptr + rows[:, None] * K + ks[None, :], mask=live[:, None], other=0.0)
        gate = tl.dot(a, tl.load(gate_ptr + ks[:, None] * N + cols[None, :]), gate)
        proj = tl.dot(a, tl.load(proj_ptr + ks[:, None] * N + cols[None, :]), proj)
    step = tl.load(step_ptr + rows, mask=live, other=0.0)[:, None]
    gate = gate * step * tl.load(gate_step_ptr + cols)[None, :]
    hidden = gate * tl.sigmoid(gate) * proj * step * tl.load(proj_step_ptr + cols)[None, :]
    step = tl.maximum(tl.max(tl.abs(hidden), axis=1), 1e-30) / TOP
    tl.store(out_ptr + rows[:, None] * N + cols[None, :], tl.clamp(hidden / step[:, None], -TOP, TOP).to(tl.float8e4nv), mask=live[:, None])
    tl.store(out_step_ptr + rows * (N // BN) + tl.program_id(1), step, mask=live)


# out = skip + gate * (x @ w) for one (BM, BN) tile, in bf16. x is (M, K) fp8 whose every SPAN columns of a row
# share a step (x_step is (M, K / SPAN)); w is (K, N) fp8 with one step per column. BK divides SPAN, so each partial
# product has a single step per row and is scaled by it before it joins the sum. skip is (M, N) bf16 and gate
# (gate_rows, N) bf16 with gate_rows 1 or the token count: the block's residual, added while the tile is in registers.
@triton.jit
def linear_fp8_kernel(x_ptr, x_step_ptr, w_ptr, w_step_ptr, skip_ptr, gate_ptr, out_ptr, M, N, K, gate_rows,
                      BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SPAN: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    live = rows < M
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for start in range(0, K, BK):
        ks = start + tl.arange(0, BK)
        a = tl.load(x_ptr + rows[:, None] * K + ks[None, :], mask=live[:, None], other=0.0)
        step = tl.load(x_step_ptr + rows * (K // SPAN) + start // SPAN, mask=live, other=0.0)
        acc += tl.dot(a, tl.load(w_ptr + ks[:, None] * N + cols[None, :])) * step[:, None]
    at = rows[:, None] * N + cols[None, :]
    gate = tl.load(gate_ptr + (rows % gate_rows)[:, None] * N + cols[None, :]).to(tl.float32)
    out = tl.load(skip_ptr + at, mask=live[:, None], other=0.0).to(tl.float32) + gate * acc * tl.load(w_step_ptr + cols)[None, :]
    tl.store(out_ptr + at, out.to(tl.bfloat16), mask=live[:, None])


# layer_norm(x) * mod as fp8 rows and their steps. x is (rows, width) bf16, mod is (1 or tokens, width) bf16.
def fp8_norm(x: Tensor, mod: Tensor, eps: float) -> tuple[Tensor, Tensor]:
    out, step = torch.empty_like(x, dtype=torch.float8_e4m3fn), x.new_empty(x.shape[0], dtype=torch.float32)
    norm_kernel[(x.shape[0],)](x, mod, out, step, mod.shape[0], eps, WIDTH=x.shape[1], FP8=True, num_warps=8)
    return out, step


# layer_norm(x) * mod in bf16, one read and one write. x is (B, S, width) bf16, mod is (B, 1 or S, width) bf16.
def norm_mod(x: Tensor, mod: Tensor, eps: float) -> Tensor:
    out, mod = torch.empty_like(x), mod.flatten(0, 1).contiguous()
    norm_kernel[(x.shape[0] * x.shape[1],)](x, mod, out, out, mod.shape[0], eps, WIDTH=x.shape[2], FP8=False, num_warps=8)
    return out


# silu(x @ gate) * (x @ proj) as (rows, hidden) fp8 and its (rows, hidden / tile columns) steps.
def fp8_swiglu(x: Tensor, step: Tensor, packed: Fp8Mlp, tile: tuple[int, int, int] = SWIGLU_TILE, warps: int = GEMM_WARPS) -> tuple[Tensor, Tensor]:
    (m, k), n = x.shape, packed.gate.shape[1]
    out, out_step = x.new_empty(m, n), step.new_empty(m, n // tile[1])
    swiglu_fp8_kernel[(triton.cdiv(m, tile[0]), n // tile[1])](x, step, packed.gate, packed.proj, packed.gate_step, packed.proj_step, out, out_step,
                                                               m, n, k, BM=tile[0], BN=tile[1], BK=tile[2], num_warps=warps, num_stages=3)
    return out, out_step


# skip + gate * (x @ w) in bf16 for fp8 x of shape (rows, in) with block steps `x_step`, fp8 w of shape (in, out),
# bf16 skip of shape (rows, out) and bf16 gate of shape (1 or tokens, out).
def fp8_linear(x: Tensor, x_step: Tensor, w: Tensor, w_step: Tensor, skip: Tensor, gate: Tensor,
               tile: tuple[int, int, int] = LINEAR_TILE, warps: int = GEMM_WARPS) -> Tensor:
    (m, k), n = x.shape, w.shape[1]
    out = torch.empty_like(skip)
    linear_fp8_kernel[(triton.cdiv(m, tile[0]), n // tile[1])](x, x_step, w, w_step, skip, gate, out, m, n, k, gate.shape[0], BM=tile[0], BN=tile[1],
                                                               BK=tile[2], SPAN=k // x_step.shape[1], num_warps=warps, num_stages=3)
    return out


# a block's whole mlp branch and its residual, `x + gate * out(silu(gate_layer(n)) * proj(n))` with
# n = layer_norm(x) * mod, in three launches: norm to fp8, both input matmuls with the swiglu in their tail,
# the output matmul with the residual in its tail. x is (B, S, width) bf16; mod and gate are (B, 1 or S, width).
def mlp_fp8(x: Tensor, mod: Tensor, gate: Tensor, packed: Fp8Mlp, eps: float) -> Tensor:
    flat = x.flatten(0, 1)
    rows, step = fp8_norm(flat, mod.flatten(0, 1).contiguous(), eps)
    hidden, hidden_step = fp8_swiglu(rows, step, packed)
    return fp8_linear(hidden, hidden_step, packed.out, packed.out_step, flat, gate.flatten(0, 1).contiguous()).view_as(x)


# the vae's norm for P pixels of a channels-last image: each pixel's channel vector is scaled to length
# `scale` (sqrt of the channel count), multiplied by gamma and, with SILU, passed through silu. one read, one
# write. x and out are (pixels, C) bf16; the channel axis is padded to CPAD, a power of two, and masked.
@triton.jit
def pixel_norm_kernel(x_ptr, gamma_ptr, out_ptr, pixels, scale, C, CPAD: tl.constexpr, P: tl.constexpr, SILU: tl.constexpr):
    pix = tl.program_id(0) * P + tl.arange(0, P)
    ch = tl.arange(0, CPAD)
    live = (pix[:, None] < pixels) & (ch[None, :] < C)
    at = pix[:, None] * C + ch[None, :]
    x = tl.load(x_ptr + at, mask=live, other=0.0).to(tl.float32)
    length = tl.maximum(tl.sqrt(tl.sum(x * x, axis=1)), 1e-12)
    y = x * (scale / length)[:, None] * tl.load(gamma_ptr + ch, mask=ch < C, other=0.0).to(tl.float32)[None, :]
    if SILU:
        y = y * tl.sigmoid(y)
    tl.store(out_ptr + at, y.to(tl.bfloat16), mask=live)


# (pixels per program, padded channel count) that keeps a tile near 4096 values.
def pixel_tile(channels: int) -> tuple[int, int]:
    padded = triton.next_power_of_2(channels)
    return max(4096 // padded, 1), padded


# normalize(x over channels) * scale * gamma, then silu if asked, for a channels-last bf16 x of shape (B, C, H, W).
def pixel_norm(x: Tensor, gamma: Tensor, scale: float, silu: bool) -> Tensor:
    out, channels = torch.empty_like(x), x.shape[1]
    pixels, (per, padded) = x.numel() // channels, pixel_tile(channels)
    pixel_norm_kernel[(triton.cdiv(pixels, per),)](x.permute(0, 2, 3, 1), gamma, out.permute(0, 2, 3, 1), pixels, scale, channels,
                                                   CPAD=padded, P=per, SILU=silu, num_warps=8)
    return out


# y + dup_up(x) for P output pixels, channels-last: y is the upsampled branch (H, W, C), x the block input
# (H/2, W/2, CIN). output channel c at sub-pixel (dy, dx) reads x channel ((c * FRAMES + FRAMES - 1) * 4 + 2 dy + dx) // REPEAT
# of its parent pixel, which is what repeat_interleave -> keep the last frame -> pixel_shuffle selects.
@triton.jit
def grow_add_kernel(y_ptr, x_ptr, out_ptr, H, W, C, CIN, FRAMES, REPEAT, pixels, CPAD: tl.constexpr, P: tl.constexpr):
    pix = tl.program_id(0) * P + tl.arange(0, P)
    ch = tl.arange(0, CPAD)
    live = (pix[:, None] < pixels) & (ch[None, :] < C)
    image, row, col = pix // (H * W), (pix % (H * W)) // W, pix % W
    parent = image * (H // 2) * (W // 2) + (row // 2) * (W // 2) + col // 2
    source = ((ch[None, :] * FRAMES + FRAMES - 1) * 4 + ((row % 2) * 2 + col % 2)[:, None]) // REPEAT
    x = tl.load(x_ptr + parent[:, None] * CIN + source, mask=live, other=0.0).to(tl.float32)
    at = pix[:, None] * C + ch[None, :]
    tl.store(out_ptr + at, (tl.load(y_ptr + at, mask=live, other=0.0).to(tl.float32) + x).to(tl.bfloat16), mask=live)


# y + dup_up(x, channels of y, temporal), in place in y. both channels-last bf16: y (B, C, 2h, 2w), x (B, CIN, h, w).
def grow_add(y: Tensor, x: Tensor, temporal: bool) -> Tensor:
    channels, frames = y.shape[1], 2 if temporal else 1
    pixels, (per, padded) = y.numel() // channels, pixel_tile(channels)
    rows = y.permute(0, 2, 3, 1)
    grow_add_kernel[(triton.cdiv(pixels, per),)](rows, x.permute(0, 2, 3, 1), rows, y.shape[2], y.shape[3], channels, x.shape[1], frames,
                                                 channels * frames * 4 // x.shape[1], pixels, CPAD=padded, P=per, num_warps=8)
    return y
