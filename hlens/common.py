"""Shared setup for the H-lens pilot.

Provenance / reuse (read-only imports, nothing copied):
- ``flagship/src/wor/modeling.py``: ``LENS_REGISTRY``, ``lens_file`` (neuronpedia lens paths),
  ``norm_gain`` (final-norm effective gain, Qwen3.5 convention 1+w).
- ``flagship/third_party/jlens`` (Apache-2.0, anthropics/jacobian-lens @ 581d398):
  ``JacobianLens.load`` and the fitting conventions (valid positions = [16, T-2],
  one-hot cotangent at every valid target position, mean over valid source positions,
  target layer = last block output, pre-final-norm).

Conventions used throughout:
- "layer l" = OUTPUT of decoder block l (jlens convention). h_final = output of block 23
  (pre-final-norm), the J-lens target (lens source layers are 0..22; jlens requires
  source < target, so target = n_layers-1 = 23).
- u_t = g * W_U[t] (g = final-norm gain 1+w): the linear readout direction in the
  pre-norm final basis, the same direction the J-lens/ablation code uses.
- Model: layers + norms in fp32, eager attention, Gated-DeltaNet pure-torch fallback;
  embeddings stay bf16 (exact: checkpoint is bf16) and are cast to fp32.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # matsempirical2027
sys.path.insert(0, str(ROOT / "flagship" / "src"))
from wor import modeling as wmod  # noqa: E402  (also puts vendored jlens on sys.path)
import jlens  # noqa: E402

OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
SKIP_FIRST = 16


def log(msg: str, path: Path | None = None) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(path or (HERE / "runlog.md"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


@dataclass
class M:
    hf: Any
    tok: Any
    text: Any  # Qwen3_5TextModel
    layers: Any
    gain: torch.Tensor  # [d] fp32
    W_U: torch.Tensor  # [V, d] bf16 (tied)
    name: str
    n_layers: int
    d: int
    final_norm: Any

    def u(self, token_ids) -> torch.Tensor:
        """[k, d] fp32 readout directions g * W_U[t]."""
        ids = torch.as_tensor(token_ids, device=self.W_U.device).reshape(-1)
        return self.W_U[ids].float() * self.gain[None]


def load_model(name: str = "Qwen/Qwen3.5-0.8B") -> M:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    hf = AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.bfloat16, device_map="cuda", low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    hf.eval()
    for p in hf.parameters():
        p.requires_grad_(False)
    text = hf.model
    # layers + final norm -> fp32 (embedding / tied lm_head stay bf16)
    for blk in text.layers:
        blk.float()
    text.norm.float()
    if hasattr(text, "rotary_emb"):
        text.rotary_emb.float()
    torch.cuda.empty_cache()
    gain, conv = wmod.norm_gain(text.norm, hf.config.get_text_config().hidden_size)
    assert conv == "1+w", conv
    cfg = hf.config.get_text_config()
    return M(hf=hf, tok=tok, text=text, layers=text.layers, gain=gain.float(),
             W_U=hf.lm_head.weight, name=name, n_layers=cfg.num_hidden_layers,
             d=cfg.hidden_size, final_norm=text.norm)


def load_lens(name: str = "Qwen/Qwen3.5-0.8B") -> jlens.JacobianLens:
    return jlens.JacobianLens.load(wmod.lens_file(name))


class Ctx:
    """One context: residual at layer l (fp32, no grad) plus the kwargs every decoder
    block receives (position embeddings, masks), captured from a real forward."""

    def __init__(self, m: M, ids: torch.Tensor, l: int):
        self.m, self.l = m, l
        self.ids = ids.reshape(1, -1).to("cuda")
        self.T = self.ids.shape[1]
        self.kw: dict[int, dict] = {}
        store: dict[int, torch.Tensor] = {}
        hooks = []

        def pre(idx):
            def f(mod, args, kwargs):
                self.kw[idx] = {k: v for k, v in kwargs.items() if k not in ("past_key_values", "use_cache")}
            return f

        def post(idx):
            def f(mod, args, out):
                o = out[0] if isinstance(out, tuple) else out
                store[idx] = o.detach().float().clone()
            return f

        for i, blk in enumerate(m.layers):
            hooks.append(blk.register_forward_pre_hook(pre(i), with_kwargs=True))
            hooks.append(blk.register_forward_hook(post(i)))
        try:
            with torch.no_grad():
                emb = m.text.embed_tokens(self.ids).float()
                m.text(inputs_embeds=emb, use_cache=False)
        finally:
            for h in hooks:
                h.remove()
        self.h_all = store  # layer -> [1, T, d]
        self.h = store[l]
        valid = torch.zeros(self.T, dtype=torch.bool)
        valid[SKIP_FIRST: self.T - 1] = True
        self.valid = valid.to("cuda")
        self.n_valid = int(valid.sum())

    def kwargs_for(self, idx: int, B: int = 1) -> dict:
        kw = dict(self.kw[idx])
        if B > 1:
            pe = kw.get("position_embeddings")
            if pe is not None:
                kw["position_embeddings"] = tuple(t.expand(B, *t.shape[1:]) if t.shape[0] == 1 else t for t in pe)
        return kw


def run_from(m: M, ctx: Ctx, h: torch.Tensor, start: int, stop: int | None = None,
             record: list[int] | None = None, record_mid: list[int] | None = None):
    """Run blocks start..stop-1 (default to the end) on h [B,T,d]. Returns (h_out, rec, rec_mid):
    rec[i] = output of block i, rec_mid[i] = residual between token mixer and MLP of block i."""
    stop = m.n_layers if stop is None else stop
    rec, rec_mid = {}, {}
    B = h.shape[0]
    for i in range(start, stop):
        blk = m.layers[i]
        kw = ctx.kwargs_for(i, B)
        if record_mid is not None and i in record_mid:
            h = block_forward_split(blk, h, kw, rec_mid, i)
        else:
            out = blk(h, **kw)
            h = out[0] if isinstance(out, tuple) else out
        if record is not None and i in record:
            rec[i] = h
    return h, rec, rec_mid


def block_forward_split(blk, h, kw, rec_mid, i):
    """Replicates Qwen3_5DecoderLayer.forward, exposing the mid residual."""
    resid = h
    x = blk.input_layernorm(h)
    if blk.layer_type == "linear_attention":
        x = blk.linear_attn(hidden_states=x, cache_params=None, attention_mask=kw.get("attention_mask"))
    else:
        x, _ = blk.self_attn(hidden_states=x, attention_mask=kw.get("attention_mask"),
                             position_ids=kw.get("position_ids"), past_key_values=None,
                             position_embeddings=kw.get("position_embeddings"))
    mid = resid + x
    rec_mid[i] = mid
    return mid + blk.mlp(blk.post_attention_layernorm(mid))


def wikitext_contexts(tok, n: int, T: int = 96, seed: int = 0, offset: int = 0) -> list[torch.Tensor]:
    """n windows of exactly T tokens from distinct wikitext-103 validation articles
    (raw text, no BOS: Qwen3.5 has none, matching how the lens was fit)."""
    import glob

    import pyarrow.parquet as pq

    fs = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--Salesforce--wikitext/snapshots/*/wikitext-103-raw-v1/validation-*.parquet"))
    lines = pq.read_table(fs[0]).column("text").to_pylist()
    # group into articles (headings " = Title = ")
    arts, cur = [], []
    for ln in lines:
        if ln.startswith(" = ") and not ln.startswith(" = = ") and cur:
            arts.append("".join(cur)); cur = []
        cur.append(ln)
    if cur:
        arts.append("".join(cur))
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(arts), generator=g).tolist()
    out = []
    for j in order[offset:]:
        ids = tok(arts[j], add_special_tokens=False)["input_ids"]
        if len(ids) < T + 200:
            continue
        s = 100  # skip the title region
        out.append(torch.tensor(ids[s:s + T]))
        if len(out) == n:
            break
    return out


def gpu_mem() -> str:
    return f"alloc {torch.cuda.memory_allocated()/2**20:.0f} MB, peak {torch.cuda.max_memory_allocated()/2**20:.0f} MB"


# ---------------------------------------------------------------- DeltaNet fast-ish torch path
def chunk_gated_delta_rule_trisolve(query, key, value, g, beta, chunk_size=64, initial_state=None,
                                    output_final_state=False, use_qk_l2norm_in_kernel=False):
    """Mathematically identical to transformers' torch_chunk_gated_delta_rule, but the
    63-step in-place forward-substitution loop that builds (I + A)^{-1} (A = strictly-lower
    k_beta k^T * decay) is replaced by one torch.linalg.solve_triangular call. This makes
    double-backward ~an order of magnitude cheaper. Equality with the original is tested
    in tests/test_deltanet_patch.py."""
    import torch.nn.functional as F
    from transformers.models.qwen3_5.modeling_qwen3_5 import l2norm

    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)]
    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size)); key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size)); beta = F.pad(beta, (0, pad_size)); g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    query = query * (1 / (query.shape[-1] ** 0.5))
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
                                         for x in (query, key, value, k_beta, v_beta)]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    A = ((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    eye = torch.eye(chunk_size, dtype=A.dtype, device=A.device)
    attn = torch.linalg.solve_triangular(A + eye, eye.expand_as(A), upper=False, unitriangular=True)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    S = (torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value) if initial_state is None
         else initial_state.to(value))
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)
    outs = []
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        a = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill(mask, 0)
        v_new = v_i - k_cumdecay[:, :, i] @ S
        outs.append((q_i * g[:, :, i, :, None].exp()) @ S + a @ v_new)
        S = S * g[:, :, i, -1, None, None].exp() + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
    core = torch.stack(outs, 2)
    core = core.reshape(core.shape[0], core.shape[1], -1, core.shape[-1])[:, :, :sequence_length]
    core = core.transpose(1, 2).contiguous().to(initial_dtype)
    return core, (S if output_final_state else None)


class ShiftConv(torch.nn.Module):
    """Causal depthwise conv as a sum of shifted multiplies. Equal to the original
    nn.Conv1d(groups=C, padding=K-1)(x)[..., :T]; returns only the first T outputs (the
    caller's [:, :, :seq_len] slice is then a no-op). Its double-backward is ~70x cheaper
    than cuDNN's depthwise conv double-backward on this GPU."""

    def __init__(self, conv: torch.nn.Conv1d):
        super().__init__()
        assert conv.groups == conv.in_channels and conv.padding[0] == conv.kernel_size[0] - 1
        self.orig = conv
        self.K = conv.kernel_size[0]

    def forward(self, x):
        import torch.nn.functional as F
        T, K, w = x.shape[-1], self.K, self.orig.weight  # w [C,1,K]
        xp = F.pad(x, (K - 1, 0))
        out = sum(w[None, :, 0, k, None] * xp[:, :, k:k + T] for k in range(K))
        if self.orig.bias is not None:
            out = out + self.orig.bias[None, :, None]
        return out


def patch_deltanet(m: "M", on: bool = True) -> None:
    from transformers.models.qwen3_5 import modeling_qwen3_5 as q
    for blk in m.layers:
        if blk.layer_type == "linear_attention":
            la = blk.linear_attn
            la.chunk_gated_delta_rule = chunk_gated_delta_rule_trisolve if on else q.torch_chunk_gated_delta_rule
            if on and not isinstance(la.conv1d, ShiftConv):
                la.conv1d = ShiftConv(la.conv1d)
            elif not on and isinstance(la.conv1d, ShiftConv):
                la.conv1d = la.conv1d.orig
