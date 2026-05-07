"""
Components for the wavelet policy network.

Reusable submodules (causal convs, lifting blocks, splitters, fusers)
that get composed into WaveletForDiffusion.
"""


from typing import Union, Optional, Tuple
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin

logger = logging.getLogger(__name__)

class CausalDilatedConv1d(nn.Module):
    """
    Causal dilated 1D convolution with channels-last input/output.
    
    Input:  (B, T, d_model)
    Output: (B, T, d_model)
    
    Output at position n depends only on input positions <= n.
    Sequence length is preserved (no decimation).
    """
    def __init__(self,
            d_model: int,
            kernel_size: int = 3,
            dilation: int = 1):
        super().__init__()

        self.d_model = d_model 
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.left_pad = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, T, d_model)

            x = x.transpose(1, 2)              # (B, d_model, T) needed for pytorch conv
            x = F.pad(x, (self.left_pad, 0))   # (B, d_model, T + left_pad)
            x = self.conv(x)                   # (B, d_model, T)
            x = x.transpose(1, 2)              # (B, T, d_model)
            return x
    
class LiftingAnalysisBlock(nn.Module):
    """
    Single-level lifting analysis block.
    
    Given input signal S_hat, produces:
        S_d = S_hat - P(S_hat)        # detail/high-frequency stream
        S_s = S_hat + U(S_d)          # approximation/low-frequency stream
    
    Input:  (B, T, d_model)
    Output: tuple (S_d, S_s), both shape (B, T, d_model)
    """
    def __init__(self,
                 d_model: int,
                 kernel_size: int = 3,
                 dilation: int = 1,
                 n_inner_layers: int = 3,
                 dropout: float = 0.1):
        super().__init__()

        self.predictor = nn.Sequential(*[
            CausalConvBlock(d_model, kernel_size, dilation, dropout)
            for _ in range(n_inner_layers)
        ]) # P

        self.updater = nn.Sequential(*[
            CausalConvBlock(d_model, kernel_size, dilation, dropout)
            for _ in range(n_inner_layers)
        ]) # U
    

    def forward(self, s_hat: torch.Tensor):
         
        # get s detail and s coarse
        s_d = s_hat - self.predictor(s_hat)
        s_s = s_hat + self.updater(s_d)

        return s_d, s_s

class CausalConvBlock(nn.Module):
    """LayerNorm + CausalDilatedConv1d + Dropout, with residual connection."""
    def __init__(self, d_model, kernel_size=3, dilation=1, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.conv = CausalDilatedConv1d(d_model, kernel_size, dilation)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: (B, T, d_model), channels-last
        residual = x
        y = self.ln(x)
        y = self.conv(y)
        y = self.dropout(y)
        return residual + y
    

class Fuser(nn.Module):
    """
    Cross-attention fuser that merges approximation and detail streams.
    
    Q = A_s (approximation)
    K = V = A_d (detail — what to incorporate)
    
    Input:  a_s, a_d both (B, T, d_model)
    Output: (B, T, d_model)
    """
    def __init__(self,
                 d_model: int,
                 n_heads: int = 4,
                 ffn_mult: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)

        self.ffn =  nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult),
            nn.GELU(),  # or nn.Mish() to match the existing codebase's style
            nn.Linear(d_model * ffn_mult, d_model),
        )

        # layer norms
        self.ln_attn = nn.LayerNorm(d_model)
        self.ln_ffn = nn.LayerNorm(d_model)

    def forward(self, a_s, a_d):
        residual = a_s
        a_s_norm = self.ln_attn(a_s)
        a_d_norm = self.ln_attn(a_d)
        atten_out, _ = self.cross_attn(query=a_s_norm, key=a_d_norm, value=a_d_norm)
        a_s = residual + atten_out

        residual = a_s
        a_s_normed = self.ln_ffn(a_s)
        ffn_out = self.ffn(a_s_normed)
        a_s = residual + ffn_out

        return a_s

class SynthesisBlock(nn.Module):
    def __init__(self, d_model, n_heads=4, ffn_mult=4, dropout=0.1):
        super().__init__()
        self.fuser = Fuser(d_model, n_heads, ffn_mult, dropout)
    
    def forward(self, a_s, a_d):
        return self.fuser(a_s, a_d)

def causal_moving_average(x: torch.Tensor, window: int = 3) -> torch.Tensor:
    """
    Causal moving average over the time dimension.
    
    For each output position n, computes the mean of input positions
    [n-window+1, ..., n], left-padding with zeros at the start.
    
    Input:  (B, T, d_model)
    Output: (B, T, d_model)
    """
    # x: (B, T, d_model) -> transpose to (B, d_model, T) for Conv1d
    x = x.transpose(1, 2)
    x = F.pad(x, (window - 1, 0))

    d_model = x.shape[1]
    kernel = torch.full(
        (d_model, 1, window),
        fill_value=1.0 / window,
        device=x.device,
        dtype=x.dtype,
    )
    
    # groups=d_model means each channel uses its own (identical) kernel
    smoothed = F.conv1d(x, kernel, groups=d_model)
    
    # back to (B, T, d_model)
    return smoothed.transpose(1, 2)


# ------------------------ tests ------------------------
def test_causal_moving_average():
    # constant input -> output equals input (after padding effects)
    x = torch.ones(2, 16, 8) * 5.0
    smoothed = causal_moving_average(x, window=3)
    # interior should be exactly 5.0 since all neighbors are 5
    assert torch.allclose(smoothed[:, 2:, :], x[:, 2:, :], atol=1e-5), "constant input failed"
    print("causal_moving_average constant input ok")
    
    # causality: changing future shouldn't affect past
    x1 = torch.randn(1, 16, 4)
    x2 = x1.clone()
    x2[0, 10:, :] = torch.randn(6, 4)
    
    y1 = causal_moving_average(x1, window=3)
    y2 = causal_moving_average(x2, window=3)
    
    # outputs at positions 0..9 should be identical
    assert torch.allclose(y1[0, :10], y2[0, :10]), "causality violated"
    print("causal_moving_average causality ok")
    
    # smoothing actually smooths
    x_noisy = torch.randn(1, 16, 4)
    y_smooth = causal_moving_average(x_noisy, window=3)
    
    # variance of smoothed should be less than variance of input
    var_in = x_noisy.var()
    var_out = y_smooth[:, 2:, :].var()  # skip padding-affected positions
    assert var_out < var_in, "smoothing didn't reduce variance"
    print("causal_moving_average smoothing ok")


def test_synthesis_block():
    block = SynthesisBlock(d_model=8, n_heads=2)
    a_s = torch.randn(2, 16, 8)
    a_d = torch.randn(2, 16, 8)
    
    out = block(a_s, a_d)
    
    assert out.shape == (2, 16, 8), f"shape mismatch: {out.shape}"
    print("synthesis_block shape ok")
    
    loss = out.sum()
    loss.backward()
    for name, p in block.named_parameters():
        assert p.grad is not None, f"NO GRADIENT: {name}"
    print("synthesis_block gradient flow ok")


def test_causal_conv():
    conv = CausalDilatedConv1d(d_model=8, kernel_size=3, dilation=2)
    
    # shape test
    x = torch.randn(2, 16, 8)
    y = conv(x)
    assert y.shape == x.shape, f"shape mismatch: got {y.shape}, expected {x.shape}"
    print("causal_conv shape ok:", y.shape)
    
    # causality test
    x1 = torch.randn(1, 16, 8)
    x2 = x1.clone()
    x2[0, 10:, :] = torch.randn(6, 8)  # change positions 10..15
    
    y1 = conv(x1)
    y2 = conv(x2)
    
    # outputs at positions 0..9 should be identical
    assert torch.allclose(y1[0, :10], y2[0, :10]), "causality violated!"
    print("causal_conv causality ok")


def test_lifting_analysis():
    block = LiftingAnalysisBlock(d_model=8, kernel_size=3, dilation=1)
    x = torch.randn(2, 16, 8)
    
    s_d, s_s = block(x)
    
    # shape tests
    assert s_d.shape == x.shape, f"s_d shape mismatch: {s_d.shape}"
    assert s_s.shape == x.shape, f"s_s shape mismatch: {s_s.shape}"
    print("lifting_analysis shapes ok")
    
    # output should differ from input (proves P and U are doing something)
    assert not torch.allclose(s_d, x), "S_d should differ from input"
    assert not torch.allclose(s_s, x), "S_s should differ from input"
    print("lifting_analysis output differs from input ok")
    
    # gradient flow test
    loss = s_d.sum() + s_s.sum()
    loss.backward()
    for name, p in block.named_parameters():
        assert p.grad is not None, f"NO GRADIENT: {name}"
    print("lifting_analysis gradient flow ok")


def test_fuser():
    fuser = Fuser(d_model=8, n_heads=2)
    a_s = torch.randn(2, 16, 8)
    a_d = torch.randn(2, 16, 8)
    
    out = fuser(a_s, a_d)
    
    # shape test
    assert out.shape == a_s.shape, f"shape mismatch: {out.shape}"
    print("fuser shape ok:", out.shape)
    
    # sanity: output should differ from inputs
    assert not torch.allclose(out, a_s), "output should differ from a_s"
    assert not torch.allclose(out, a_d), "output should differ from a_d"
    print("fuser output differs from inputs ok")
    
    # gradient test
    loss = out.sum()
    loss.backward()
    for name, p in fuser.named_parameters():
        assert p.grad is not None, f"NO GRADIENT: {name}"
    print("fuser gradient flow ok")


if __name__ == "__main__":
    test_causal_conv()
    test_lifting_analysis()
    test_fuser()
    test_synthesis_block()
    test_causal_moving_average()




