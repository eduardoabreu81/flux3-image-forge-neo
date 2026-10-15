# FLUX 3 Image Klein support for Forge Neo
# The FLUX 3 image DiT, ported from black-forest-labs/flux-3-image-klein (src/flux3/dit.py, see FLUX3_COMMIT in
# native/__init__.py), with ComfyUI's batched layout (comfy/ldm/flux3/model.py, COMFYUI_COMMIT).
#
# Five per-mode blocks run the image, the reference images (image_cond) and the text each on its own, then 28
# single-stream blocks run the joint sequence [image, image_cond, text] under per-mode modulation. Every block is
# parallel attention + SwiGLU under one adaLN and one gate. Text and references sit at t = 0; the image at the
# step's t. The distilled checkpoint (a flow map) also embeds the time each step jumps to (ref_time_in).
#
# Layers are plain torch modules: inside Forge's using_forge_operations they become Forge operations (dtype,
# INT8 ConvRot / NVFP4 quantization, offload, LoRA), outside it they stay torch modules for CPU tests.
# Differences from upstream: samples are a batch, not a packed sequence; Forge's attention runs through
# kernels.attention; reference latents come from Forge's dynamic_args.ref_latents.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import flowmap, kernels

AXES_DIM = (32, 32, 32, 32)
THETA = 10000
IMAGE_ROPE_SCALE = 0.5
TIME_DIM = 256
EPS = 1e-6


def timestep_embedding(t, dim=TIME_DIM, max_period=10000):
    # upstream feeds the time in bfloat16 (its timesteps are bf16 tensors, the reference time is cast to them), so
    # the embedding sees the bf16-rounded time whatever the compute dtype
    t = 1000.0 * t.to(torch.bfloat16).float()
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def rope(pos, dim, theta):
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    out = torch.einsum("...n,d->...nd", pos.double(), 1.0 / (theta ** scale))
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    return out.reshape(*out.shape[:-1], 2, 2).float()


def rope_embed(ids):
    """ids (L, 4) -> rotation matrices (1, 1, L, head_dim // 2, 2, 2)."""
    emb = torch.cat([rope(ids[..., i], AXES_DIM[i], THETA) for i in range(len(AXES_DIM))], dim=-3)
    return emb[None, None]


def rmsnorm_rope(x, scale, freqs):
    # x [B, H, L, D] -> RMS norm and rotation in float32, back in x's dtype
    b, h, length, d = x.shape
    x_f = x.float()
    x_norm = x_f * x_f.pow(2).mean(-1, keepdim=True).add(EPS).rsqrt() * kernels.cast_to(scale, torch.float32, x.device)
    x_r = x_norm.reshape(b, h, length, d // 2, 1, 2)
    rotated = freqs[..., 0] * x_r[..., 0] + freqs[..., 1] * x_r[..., 1]
    return rotated.reshape(b, h, length, d).to(x.dtype)


def image_ids(h, w, index, device):
    # (t, row, col, l); references are told apart on the l axis
    ids = torch.zeros(h, w, 4, device=device, dtype=torch.float32)
    ids[..., 1] = torch.arange(h, device=device, dtype=torch.float32)[:, None] * IMAGE_ROPE_SCALE
    ids[..., 2] = torch.arange(w, device=device, dtype=torch.float32)[None, :] * IMAGE_ROPE_SCALE
    ids[..., 3] = index
    return ids.flatten(0, 1)


def text_ids(length, device):
    ids = torch.zeros(length, 4, device=device, dtype=torch.float32)
    ids[:, 3] = torch.arange(length, device=device, dtype=torch.float32)
    return ids


def modulated_norm(x, shift, scale):
    # LayerNorm without affine, then the adaLN shift/scale, in float32 (upstream runs it under bf16 autocast, where
    # layer_norm computes in float32)
    return (1.0 + scale.float()) * F.layer_norm(x.float(), (x.shape[-1],), eps=EPS) + shift.float()


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # the checkpoints name it scale (upstream); torch.ones, not empty: the checkpoint overwrites it
        self.scale = nn.Parameter(torch.ones(dim))


class QKNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q, k, freqs):
        return rmsnorm_rope(q, self.query_norm.scale, freqs), rmsnorm_rope(k, self.key_norm.scale, freqs)


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=False)
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.out_layer(F.silu(self.in_layer(x)))


class ModeModulation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.lin = nn.Linear(dim, 3 * dim, bias=False)

    def forward(self, vec):
        # (B, D) -> shift, scale, gate, each (B, 1, D)
        return tuple(t[:, None] for t in self.lin(F.silu(vec)).chunk(3, dim=-1))


class LastLayer(nn.Module):
    def __init__(self, hidden, out_channels):
        super().__init__()
        self.linear = nn.Linear(hidden, out_channels, bias=False)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 2 * hidden, bias=False))

    def forward(self, x, vec):
        shift, scale = self.adaLN_modulation(vec)[:, None].chunk(2, dim=-1)
        return self.linear(modulated_norm(x, shift, scale).to(x.dtype))


class Block(nn.Module):
    """Parallel attention + SwiGLU MLP under one adaLN and one gate; per-mode and single-stream blocks alike."""

    def __init__(self, hidden, heads, mlp_hidden, fused_qkv=False):
        super().__init__()
        self.heads = heads
        self.fused_qkv = fused_qkv
        if fused_qkv:
            # the INT8 ConvRot build fuses q, k, v into one GEMM
            self.qkv = nn.Linear(hidden, 3 * hidden, bias=False)
        else:
            self.q_proj = nn.Linear(hidden, hidden, bias=False)
            self.k_proj = nn.Linear(hidden, hidden, bias=False)
            self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.mlp_in = nn.Linear(hidden, 2 * mlp_hidden, bias=False)
        self.attn_out = nn.Linear(hidden, hidden, bias=False)
        self.mlp_out = nn.Linear(mlp_hidden, hidden, bias=False)
        self.norm = QKNorm(hidden // heads)

    def forward(self, x, spans, freqs, dtype, transformer_options=None):
        """x (B, N, D); spans: (start, end, (shift, scale, gate)) row ranges covering x, one per mode."""
        b, n, _ = x.shape
        x_mod = torch.cat([modulated_norm(x[:, a:e], shift, scale) for a, e, (shift, scale, _) in spans],
                          dim=1).to(dtype)
        if self.fused_qkv:
            q, k, v = self.qkv(x_mod).chunk(3, dim=-1)
        else:
            q, k, v = self.q_proj(x_mod), self.k_proj(x_mod), self.v_proj(x_mod)
        split = lambda t: t.reshape(b, n, self.heads, -1).transpose(1, 2)  # noqa: E731  B N (H D) -> B H N D
        q, k = self.norm(split(q), split(k), freqs)
        attn = kernels.attention(q, k, split(v).to(q.dtype), transformer_options=transformer_options)
        gate_in, up = self.mlp_in(x_mod).chunk(2, dim=-1)
        del x_mod
        out = self.attn_out(attn.to(dtype)) + self.mlp_out(F.silu(gate_in) * up)
        return torch.cat([x[:, a:e] + gate.to(x.dtype) * out[:, a:e].to(x.dtype) for a, e, (_, _, gate) in spans],
                         dim=1)


class Flux3DiT(nn.Module):
    def __init__(self, hidden_size=3072, num_heads=24, depth=5, depth_single_blocks=28, mlp_hidden=9216,
                 in_channels=128, text_dim=20480, fused_qkv=False, flow_map=False, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.flow_map = flow_map
        content_modes = ("image", "image_cond")
        block = lambda: Block(hidden_size, num_heads, mlp_hidden, fused_qkv)  # noqa: E731

        self.emb_in = nn.ModuleDict({m: nn.Linear(in_channels, hidden_size, bias=False) for m in content_modes})
        self.txt_in = nn.Linear(text_dim, hidden_size, bias=False)
        self.time_in = MLPEmbedder(TIME_DIM, hidden_size)
        if flow_map:
            # step-distilled flow map: embeds the time each step jumps to
            self.ref_time_in = MLPEmbedder(TIME_DIM, hidden_size)
        self.early_stream_modulations = nn.ModuleDict({m: ModeModulation(hidden_size)
                                                       for m in (*content_modes, "txt")})
        self.single_stream_modulations = nn.ModuleDict({m: ModeModulation(hidden_size)
                                                        for m in (*content_modes, "txt")})
        self.content_mode_blocks = nn.ModuleDict({m: nn.ModuleList([block() for _ in range(depth)])
                                                  for m in content_modes})
        self.txt_mode_blocks = nn.ModuleList([block() for _ in range(depth)])
        self.single_blocks = nn.ModuleList([block() for _ in range(depth_single_blocks)])
        self.final_layer = nn.ModuleDict({"image": LastLayer(hidden_size, in_channels)})

    def _embed_time(self, t, dtype):
        return self.time_in(timestep_embedding(t).to(dtype))

    def forward(self, x, timestep, context, transformer_options=None, ref_latents=None, **kwargs):
        """x (B, 128, H, W) latents, timestep (B,) = sigma, context (B, T, 20480) -> velocity (B, 128, H, W)."""
        if ref_latents is None:
            from backend.args import dynamic_args

            ref_latents = dynamic_args.ref_latents
        return self.run(x, timestep, context, ref_latents, transformer_options)

    def run(self, x, timestep, context, ref_latents=(), transformer_options=None):
        b, _, h, w = x.shape
        device = x.device
        # Forge hands the inputs over in the computation dtype (k_model.apply_model); the weights may be stored in
        # another one (INT8, NVFP4, manual cast), so the context is what says which
        dtype = context.dtype
        # fp16 overflows the residual stream of large DiTs (the Iris port saw it); keep it in float32 there
        res_dtype = torch.float32 if dtype == torch.float16 else dtype

        zero = timestep.new_zeros(1)
        vec_img = self._embed_time(timestep, dtype)
        vec_zero = self._embed_time(zero, dtype)
        if self.flow_map:
            # every mode is also modulated by the time this step jumps to: the next sigma of the schedule
            target = timestep.new_tensor([flowmap.target(float(timestep[0]))])
            vec_target = self.ref_time_in(timestep_embedding(target).to(dtype))
            vec_img, vec_zero = vec_img + vec_target, vec_zero + vec_target
        vec_zero = vec_zero.expand(b, -1)

        # the image's own early blocks, then the conditioning's (text and references attend only to themselves)
        img = self.emb_in["image"](x.flatten(2).transpose(1, 2).to(dtype)).to(res_dtype)
        parts = [(img, "image", image_ids(h, w, 0, device), vec_img)]
        refs = [r for r in (ref_latents or []) if r is not None]
        if refs:
            tokens = torch.cat([r.to(device=device, dtype=dtype).flatten(2).transpose(1, 2) for r in refs], dim=1)
            ref_tokens = self.emb_in["image_cond"](tokens).to(res_dtype).expand(b, -1, -1)
            ids = torch.cat([image_ids(r.shape[-2], r.shape[-1], -1 - i, device) for i, r in enumerate(refs)])
            parts.append((ref_tokens, "image_cond", ids, vec_zero))
        txt = self.txt_in(context.to(device=device, dtype=dtype)).to(res_dtype)
        parts.append((txt, "txt", text_ids(txt.shape[1], device), vec_zero))

        blocks = {"image": self.content_mode_blocks["image"], "image_cond": self.content_mode_blocks["image_cond"],
                  "txt": self.txt_mode_blocks}
        streams = []
        for tokens, mode, ids, vec in parts:
            mod = self.early_stream_modulations[mode](vec)
            freqs = rope_embed(ids)
            for block in blocks[mode]:
                tokens = block(tokens, [(0, tokens.shape[1], mod)], freqs, dtype, transformer_options)
            streams.append(tokens)

        spans, start = [], 0
        for tokens, (_, mode, _, vec) in zip(streams, parts):
            spans.append((start, start + tokens.shape[1], self.single_stream_modulations[mode](vec)))
            start += tokens.shape[1]
        freqs = rope_embed(torch.cat([ids for _, _, ids, _ in parts]))
        joint = torch.cat(streams, dim=1)
        del streams, parts
        for block in self.single_blocks:
            joint = block(joint, spans, freqs, dtype, transformer_options)

        out = self.final_layer["image"](joint[:, : h * w].to(dtype), vec_img)
        return out.transpose(1, 2).reshape(b, self.in_channels, h, w)
