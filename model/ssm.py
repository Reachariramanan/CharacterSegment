"""Pure-PyTorch S4D (diagonal state-space model) block -- NOT the official
mamba-ssm package. mamba-ssm's selective-scan/causal-conv1d kernels are
custom CUDA (nvcc) builds with no Windows wheels; this implementation uses
only standard torch ops (complex tensors + FFT), so it runs on CPU or CUDA
on Windows with no extra build step, at the cost of not having a fused
kernel (irrelevant at this problem's tiny sequence lengths / channel count).

Math (continuous-time diagonal SSM, per-channel independent):
    x'(t) = A x(t) + B u(t)
    y(t)  = C x(t) + D u(t)
with A diagonal (state dim d_state, one bank of poles per channel), and
u a length-L 1D sequence per channel (the "time" axis is the image's x/column
axis here, not real time).

Discretization (zero-order hold, per S4D):
    Abar_n = exp(dt * A_n)
    Bbar_n = (Abar_n - 1) / A_n          (assumes B_n == 1; only C carries
                                           scale, standard S4D simplification)
Convolutional form (avoids a sequential scan): the SSM's impulse response is
    K[l] = 2 * Re( sum_n C_n * Bbar_n * Abar_n^l ),   l = 0..L-1
and y = K * u (causal 1D convolution), computed via FFT for O(L log L).

A's real part is parameterized as -exp(log_A_real) to guarantee Re(A) < 0
(stability: Abar_n stays inside the unit disk, so K decays rather than
diverging over the sliding-window lengths used here).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.config import SSMConfig


class S4DKernel(nn.Module):
    def __init__(self, d_model: int, d_state: int, dt_min: float = 1e-3, dt_max: float = 1e-1):
        super().__init__()
        log_dt = torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)

        # HiPPO-inspired init: imaginary parts spread across [0, pi*(d_state-1)],
        # real part initialized to -0.5 for all (n, channel) pairs.
        A_imag_init = math.pi * torch.arange(d_state).float().unsqueeze(0).repeat(d_model, 1)
        self.A_imag = nn.Parameter(A_imag_init)
        self.A_real_log = nn.Parameter(torch.log(0.5 * torch.ones(d_model, d_state)))

        self.C_real = nn.Parameter(torch.randn(d_model, d_state) * 0.5)
        self.C_imag = nn.Parameter(torch.randn(d_model, d_state) * 0.5)

    def forward(self, length: int) -> torch.Tensor:
        """Returns the convolution kernel K, shape (d_model, length)."""
        dt = torch.exp(self.log_dt)  # (d_model,)
        A = -torch.exp(self.A_real_log) + 1j * self.A_imag  # (d_model, d_state)
        C = self.C_real + 1j * self.C_imag  # (d_model, d_state)

        dtA = A * dt.unsqueeze(-1)  # (d_model, d_state)
        Abar = torch.exp(dtA)
        Bbar = (Abar - 1.0) / A  # B assumed == 1; A has strictly negative real part so never 0

        l = torch.arange(length, device=A.device, dtype=torch.float32)
        # Abar^l computed stably as exp(l * dtA), since Abar = exp(dtA).
        power_terms = torch.exp(dtA.unsqueeze(-1) * l.view(1, 1, -1))  # (d_model, d_state, length)

        coeff = (C * Bbar).unsqueeze(-1)  # (d_model, d_state, 1)
        kernel = 2.0 * torch.real((coeff * power_terms).sum(dim=1))  # (d_model, length)
        return kernel


class S4DLayer(nn.Module):
    def __init__(self, d_model: int, d_state: int):
        super().__init__()
        self.kernel = S4DKernel(d_model, d_state)
        self.D = nn.Parameter(torch.randn(d_model))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """u: (B, d_model, L) -> (B, d_model, L), causal depthwise SSM conv."""
        length = u.shape[-1]
        kernel = self.kernel(length)  # (d_model, L)

        n_fft = 2 * length  # zero-padded to avoid circular-convolution wraparound
        u_f = torch.fft.rfft(u, n=n_fft)
        k_f = torch.fft.rfft(kernel, n=n_fft)
        y = torch.fft.irfft(u_f * k_f.unsqueeze(0), n=n_fft)[..., :length]
        return y + u * self.D.view(1, -1, 1)


class GLUFeedForward(nn.Module):
    def __init__(self, d_model: int, expand: int = 2):
        super().__init__()
        hidden = d_model * expand
        self.in_proj = nn.Linear(d_model, 2 * hidden)
        self.out_proj = nn.Linear(hidden, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model)"""
        a, b = self.in_proj(x).chunk(2, dim=-1)
        return self.out_proj(a * torch.sigmoid(b))


class S4DBlock(nn.Module):
    """Pre-norm residual S4D layer + pre-norm residual GLU FFN, operating on
    (B, L, d_model) -- the usual transformer-style block shape, with the SSM
    doing the sequence mixing that self-attention would normally do."""

    def __init__(self, d_model: int, d_state: int, ffn_expand: int = 2):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.ssm = S4DLayer(d_model, d_state)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = GLUFeedForward(d_model, ffn_expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model)"""
        h = self.norm1(x).transpose(1, 2)  # (B, d_model, L)
        h = self.ssm(h).transpose(1, 2)  # (B, L, d_model)
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x


class S4DStack(nn.Module):
    def __init__(self, cfg: SSMConfig | None = None):
        super().__init__()
        cfg = cfg or SSMConfig()
        self.blocks = nn.ModuleList(
            [S4DBlock(cfg.d_model, cfg.d_state, cfg.ffn_expand) for _ in range(cfg.n_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model) -> (B, L, d_model)"""
        for block in self.blocks:
            x = block(x)
        return x
