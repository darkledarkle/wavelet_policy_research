from typing import Union, Optional, Tuple
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusion_policy.model.diffusion.wavelet_components import (
    CausalDilatedConv1d, LiftingAnalysisBlock, Fuser, SynthesisBlock, causal_moving_average
)
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin


logger = logging.getLogger(__name__)

class WaveletForDiffusion(ModuleAttrMixin):
    def __init__(self,
            input_dim: int,
            output_dim: int,
            horizon: int,
            n_obs_steps: int,
            cond_dim: int = 0,
            d_model=128,
            n_levels=6,
            kernel_size=3,
            n_heads=4,
            ffn_mult=4,
            dropout=0.1
        ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.horizon = horizon
        self.n_levels = n_levels

        # embedding stems -> adapt input
        self.action_emb = nn.Linear(input_dim, d_model)
        self.time_emb = SinusoidalPosEmb(d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.Mish(),
            nn.Linear(4 * d_model, d_model),
        )
        
        if cond_dim > 0:
            self.obs_emb = nn.Linear(cond_dim, d_model)
        else:
            self.obs_emb = None

        # analysis blocks - L lifting
        self.analysis_blocks = nn.ModuleList([
            LiftingAnalysisBlock(d_model, kernel_size, dilation=2**i)
            for i in range(n_levels)
        ])

        # converters - one per action stream
        self.converter_approx = CausalDilatedConv1d(d_model, kernel_size)
        self.converters_detail = nn.ModuleList([
            CausalDilatedConv1d(d_model, kernel_size)
            for _ in range(n_levels)
        ])

        # synthesis
        self.synthesis_blocks = nn.ModuleList([
            SynthesisBlock(d_model, n_heads, ffn_mult, dropout)
            for _ in range(n_levels)
        ])

        # output head
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, output_dim)


        self.apply(self._init_weights)
        logger.info("number of parameters: %e", 
                    sum(p.numel() for p in self.parameters()))



    def forward(self, 
        sample: torch.Tensor, 
        timestep: Union[torch.Tensor, float, int], 
        cond: Optional[torch.Tensor]=None,
        return_aux: bool = False, **kwargs):

        """
        sample: (B, T, input_dim) — noisy action chunk
        timestep: (B,) or scalar — diffusion timestep
        cond: (B, T_obs, cond_dim) — observations
        returns: (B, T, output_dim) — predicted noise
        """
        
        # timestamps
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        # embeddings
        a_emb = self.action_emb(sample)
        t_emb = self.time_mlp(self.time_emb(timesteps))
        t_emb = t_emb.unsqueeze(1)

        x = a_emb + t_emb # (B, T, d_model)

        if self.obs_emb is not None and cond is not None:
            o_emb = self.obs_emb(cond)                  # (B, T_obs, d_model)
            o_emb = o_emb.mean(dim=1, keepdim=True)     # (B, 1, d_model) pooled
            x = x + o_emb                               # broadcast add

        detail_streams = []
        s_s = x

        for analysis_block in self.analysis_blocks:
            s_d, s_s = analysis_block(s_s)
            detail_streams.append(s_d)
        # s_s is deepest aprox

        a_s = self.converter_approx(s_s)
        a_ds = [conv(s) for conv, s in zip(self.converters_detail, detail_streams)] # maps detail tensor with converter

        # collect a_s tensors at each synthesis level for aux loss
        # a_s_per_level[0] is the deepest (A^L_s, before any synthesis)
        # a_s_per_level[i+1] is the result after synthesis_blocks[L-1-i]

        smoothes = [a_s]

        # synthesis cascade
        for i in reversed(range(self.n_levels)):
            # inverse U update to disentangle a_s from detailed
            u_net = self.analysis_blocks[i].updater
            even = a_s - u_net(a_ds[i])

            # fuser
            a_s = self.synthesis_blocks[i](even, a_ds[i])
            smoothes.append(a_s)
        
        x = self.ln_f(a_s)
        x = self.head(x) # (B, T, output_dim)

        d_loss = torch.mean(torch.stack([
            F.smooth_l1_loss(a_d, torch.zeros_like(a_d)) for a_d in a_ds
        ]))

        s_loss = torch.mean(torch.stack([
            F.smooth_l1_loss(s_prev, causal_moving_average(s_next, window=3))
            for s_prev, s_next in zip(smoothes[:-1], smoothes[1:])
        ]))

        if return_aux:
            return x, d_loss, s_loss
        return x

    def _init_weights(self, module):
        ignore_types = (nn.Dropout, 
            SinusoidalPosEmb, 
            nn.TransformerEncoderLayer, 
            nn.TransformerDecoderLayer,
            nn.TransformerEncoder,
            nn.TransformerDecoder,
            nn.ModuleList,
            nn.Mish,
            nn.GELU,
            nn.Sequential,
            CausalDilatedConv1d,
            LiftingAnalysisBlock,
            Fuser,
            SynthesisBlock,
            WaveletForDiffusion)
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = [
                'in_proj_weight', 'q_proj_weight', 'k_proj_weight', 'v_proj_weight']
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)
            
            bias_names = ['in_proj_bias', 'bias_k', 'bias_v']
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, nn.Conv1d):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, ignore_types):
            # no param
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))


    def get_optim_groups(self, weight_decay: float=1e-3):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention, torch.nn.Conv1d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.startswith("bias"):
                    # MultiheadAttention bias starts with "bias"
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # # special case the position embedding parameter in the root GPT module as not decayed
        # no_decay.add("pos_emb")
        no_decay.add("_dummy_variable")
        # if self.cond_pos_emb is not None:
        #     no_decay.add("cond_pos_emb")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups

    def configure_optimizers(self, 
            learning_rate: float=1e-4, 
            weight_decay: float=1e-3,
            betas: Tuple[float, float]=(0.9,0.95)):
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer


def test():
    # match push-T low-dim dimensions exactly
    model = WaveletForDiffusion(
        input_dim=2,        # push-T action dim
        output_dim=2,       # push-T action dim
        horizon=16,         # push-T horizon
        n_obs_steps=2,      # push-T obs steps
        cond_dim=20,        # push-T obs dim
        d_model=128,
        n_levels=2,
    )
    
    # dummy inputs at push-T shapes
    B = 4
    sample = torch.randn(B, 16, 2)
    timestep = torch.tensor(50)
    cond = torch.randn(B, 2, 20)
    
    # forward pass
    out = model(sample, timestep, cond)
    
    # shape check
    assert out.shape == (B, 16, 2), f"shape mismatch: got {out.shape}"
    print(f"shape ok: {out.shape}")
    
    # check grad flow
    loss = out.sum()
    loss.backward()
    missing = []
    for name, p in model.named_parameters():
        if name == "_dummy_variable":
            continue
        if p.grad is None:
            missing.append(name)
    if missing:
        print(f"PARAMETERS WITHOUT GRADIENTS:")
        for name in missing:
            print(f"  {name}")
        raise RuntimeError(f"{len(missing)} parameters did not receive gradients")
    print("gradient flow ok — all parameters connected")
    
    # parameter count
    total = sum(p.numel() for p in model.parameters())
    print(f"total parameters: {total:,}")
    
    # optimizer check
    opt = model.configure_optimizers()
    print(f"optimizer configured: {type(opt).__name__}")


if __name__ == "__main__":
    test()
    