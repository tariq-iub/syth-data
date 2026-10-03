"""
CopperSpec-Synth :: tiny_model/model.py
=======================================
Tiny conditional DDPM U-Net for 64x64 copper patches.

Design targets (2 GB VRAM laptop):
    * input 3 x 64 x 64, channels [64, 128, 256], only 2 downsamples
    * condition = flattened palette vector (24 colours x 3 LAB = 72 dim)
      -> MLP -> 128-d, injected into every residual block via FiLM
      (scale & shift) modulation
    * ~3 M parameters total (< 5 M budget, < 10 M hard cap)

All code is device agnostic: tensors live wherever the caller puts them;
nothing here ever calls .cuda(). CPU is the default device.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------
def sinusoidal_timestep_emb(t: torch.Tensor, dim: int = 64) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0)
                      * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class ResBlock(nn.Module):
    """Conv-GroupNorm-SiLU with FiLM modulation from the joint (time+palette)
    embedding; optional self-attention on the output."""

    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, attn: bool = False):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(8, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.film1 = nn.Linear(emb_dim, out_ch * 2)          # scale + shift
        self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.film2 = nn.Linear(emb_dim, out_ch * 2)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.attn = nn.MultiheadAttention(out_ch, num_heads=4, batch_first=True) \
            if attn else None
        self.act = nn.SiLU()

    @staticmethod
    def _modulate(x: torch.Tensor, film: torch.Tensor) -> torch.Tensor:
        scale, shift = film.chunk(2, dim=-1)
        return x * (1 + scale[:, :, None, None]) + shift[:, :, None, None]

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = self._modulate(h, self.film1(emb))
        h = self.conv2(self.act(self.norm2(h)))
        h = self._modulate(h, self.film2(emb))
        x = self.skip(x) + h
        if self.attn is not None:
            b, c, hh, ww = x.shape
            tok = x.flatten(2).transpose(1, 2)               # (B, HW, C)
            att, _ = self.attn(tok, tok, tok, need_weights=False)
            x = x + att.transpose(1, 2).reshape(b, c, hh, ww)
        return x


class Down(nn.Module):
    """Channel-changing strided convolution (ch_in -> ch_out), halves spatial."""

    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self.op = nn.Conv2d(ch_in, ch_out, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Up(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.op = nn.Conv2d(ch, ch * 2, 3, padding=1)

    def forward(self, x):
        return self.op(F.interpolate(x, scale_factor=2, mode="nearest"))


class PaletteEncoder(nn.Module):
    """(24*3 = 72)-dim flattened LAB palette -> 128-d FiLM context vector."""

    def __init__(self, in_dim: int = 72, hidden: int = 128, out_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, pal: torch.Tensor) -> torch.Tensor:
        return self.mlp(pal)


# --------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------
class TinyConditionalUNet(nn.Module):
    """
    DDPM noise-prediction network.

    x   : (B,3,64,64) noisy patch in [-1,1]
    t   : (B,) integer diffusion timesteps
    pal : (B,72) flattened palette (24 LAB colours), DB-derived condition
    ->  (B,3,64,64) predicted noise epsilon
    """

    def __init__(self, base_ch: int = 64, ch_mults: Tuple[int, ...] = (1, 2, 4),
                 cond_dim: int = 128, palette_len: int = 72):
        super().__init__()
        emb_dim = cond_dim * 2                                 # time + palette
        self.time_mlp = nn.Sequential(
            nn.Linear(64, emb_dim), nn.SiLU(), nn.Linear(emb_dim, cond_dim))
        # joint embedding is projected down to cond_dim so every FiLM Linear
        # sees a fixed width regardless of conditioning depth (VRAM-friendly)
        self.emb_proj = nn.Linear(emb_dim, cond_dim)
        self.pal_enc = PaletteEncoder(palette_len, 128, cond_dim)

        chans = [base_ch * m for m in ch_mults]                # [64,128,256]
        self.stem = nn.Conv2d(3, chans[0], 3, padding=1)

        # ---- encoder: stem + block at current res are both saved as skips,
        #      then downsample. skip_chans tracks the tensor widths pushed. ---
        # Down blocks change channel width (ch -> ch*2); ResBlocks keep width.
        self.pre_blocks = nn.ModuleList([ResBlock(ch, ch, cond_dim)
                                         for ch in chans[:-1]])
        self.downs = nn.ModuleList([Down(chans[i], chans[i + 1])
                                    for i in range(len(chans) - 1)])
        skip_chans = []                                        # pushed per forward
        c_in = chans[-1]
        self.mid_film = nn.Linear(cond_dim, c_in * 2)  # extra FiLM at bottleneck
        self.mid1 = ResBlock(c_in, c_in, cond_dim)
        self.mid2 = ResBlock(c_in, c_in, cond_dim, attn=True)

        # ---- decoder: upsample, concat matching skip, residual block -----
        # NOTE: encoder block outputs (skip_chans[1:]) are consumed by the
        # downsampler, so the matching skips are [ch0, ch1]; the stem output
        # pairs with the final 64->64 block.
        self.up_blocks = nn.ModuleList()
        self.ups = nn.ModuleList()
        # static push order -> deterministic pop order in forward():
        #   pushes: [64,64, 128,128, 256]   pops: 256,128,128,64,64
        up_specs = [(256 + 256, 128), (128 + 128, 64), (64 + 64, 64)]
        for cat_ch, ch in up_specs:
            self.ups.append(Up(c_in))
            self.up_blocks.append(ResBlock(c_in + cat_ch, ch, cond_dim))
            c_in = ch
        self.out_norm = nn.GroupNorm(8, chans[0])
        self.out_conv = nn.Conv2d(chans[0], 3, 3, padding=1)
        self.act = nn.SiLU()

        self._checkpoint = False                               # opt-in

    def enable_grad_checkpointing(self, flag: bool = True):
        self._checkpoint = flag

    def _run_block(self, blk, x, emb):
        if self._checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(blk, x, emb, use_reentrant=False)
        return blk(x, emb)

    def forward(self, x: torch.Tensor, t: torch.Tensor,
                pal: torch.Tensor) -> torch.Tensor:
        temb = self.time_mlp(sinusoidal_timestep_emb(t, 64))
        pemb = self.pal_enc(pal)
        emb = self.emb_proj(torch.cat([temb, pemb], dim=-1))

        h = self.stem(x)
        skips = [h]                                            # 64 @ 64x64
        for blk, dn in zip(self.pre_blocks, self.downs):
            h = self._run_block(blk, h, emb)
            skips.append(h)                                    # 64 / 128
            h = dn(h)                                          # -> 128 / 256
        skips.append(h)                                        # 256 @ 16x16
        m_emb = emb
        scale, shift = self.mid_film(emb).chunk(2, dim=-1)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self._run_block(self.mid1, h, m_emb)
        h = self._run_block(self.mid2, h, m_emb)
        for blk, up in zip(self.up_blocks, self.ups):
            h = up(h)
            s = skips.pop()
            if h.shape[-2:] != s.shape[-2:]:
                h = F.interpolate(h, size=s.shape[-2:], mode="nearest")
            h = torch.cat([h, s], dim=1)
            h = self._run_block(blk, h, emb)
        h = self.act(self.out_norm(h))
        return self.out_conv(h)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# --------------------------------------------------------------------------
# Gaussian diffusion utilities (DDPM training + DDIM sampling)
# --------------------------------------------------------------------------
def make_beta_schedule(T: int = 250, device=None) -> torch.Tensor:
    """Linear beta ramp 4e-4 -> 0.02 (stable for small T on toy data)."""
    return torch.linspace(0.0004, 0.02, T, device=device)


class Diffusion:
    def __init__(self, T: int = 250, device=None):
        self.T = T
        betas = make_beta_schedule(T, device)
        self.betas = betas
        self.alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)

    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        return self

    def q_sample(self, x0, t, eps=None):
        acp = self.alphas_cumprod[t][:, None, None, None]
        if eps is None:
            eps = torch.randn_like(x0)
        return acp.sqrt() * x0 + (1 - acp).sqrt() * eps, eps

    @torch.no_grad()
    def ddim_sample(self, model, shape, pal, steps: int = 50, eta: float = 0.0,
                    generator=None) -> torch.Tensor:
        """DDIM sampler over a uniform subsequence of timesteps."""
        device = pal.device
        x = torch.randn(shape, device=device, generator=generator)
        times = torch.linspace(self.T - 1, 0, steps, device=device).long()
        acp = self.alphas_cumprod
        for i in range(steps):
            t = times[i]
            tb = torch.full((shape[0],), int(t), device=device, dtype=torch.long)
            eps = model(x, tb, pal)
            a_now = acp[t]
            a_prev = acp[times[i + 1]] if i + 1 < steps \
                else torch.tensor(1.0, device=device)
            x0_hat = ((x - (1 - a_now).sqrt() * eps) / a_now.sqrt()).clamp(-1, 1)
            sigma = eta * (((1 - a_prev) / (1 - a_now)
                            * (1 - a_now / a_prev)).clamp(min=0)).sqrt()
            dir_xt = (1 - a_prev - sigma ** 2).clamp(min=0).sqrt() * eps
            x = a_prev.sqrt() * x0_hat + dir_xt
            if eta > 0 and i + 1 < steps:
                x = x + sigma * torch.randn(x.shape, device=device,
                                            generator=generator)
        return x


if __name__ == "__main__":
    torch.manual_seed(0)
    m = TinyConditionalUNet()
    n = count_params(m)
    x = torch.randn(2, 3, 64, 64)
    t = torch.randint(0, 250, (2,))
    p = torch.randn(2, 72)
    out = m(x, t, p)
    print(f"params: {n:,} ({n / 1e6:.2f}M)  output {tuple(out.shape)}")
    assert out.shape == x.shape and n < 5_000_000
    print("model OK")
