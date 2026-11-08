#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Export the FlexGlue models for the app's experimental wide-angle loop closure.

Writes, into <out> (App/FlexGlueModels by default, which the app build bundles when present):
- flexglue_descriptor.mlpackage: the DINOv3 ConvNeXt-B descriptor at 384 x 512
  (image [1, 3, 384, 512] in 0..1 -> features [1, 256, 48, 64])
- flexglue_transformer_a.mlpackage / flexglue_transformer_b.mlpackage: the first 6 transformer
  layers, split after layer 4 (f0, f1 -> x0, x1 [1, 512, 1, 3072], covis [1]; y0, y1 -> x0, x1, covis)
- flexglue_head.safetensors: the correspondence head after layer 6, run in MLX by the app

The transformer is rewritten for the Apple Neural Engine with the same weights: tokens channels-first
[1, C, 1, S], linear layers as 1x1 convolutions, attention per head with the queries in chunks of
1024 (Core ML's fused attention op and the token-major layout ran 2-3x slower there). The script
checks the rewrite against FlexGlue's own transformer.

Requires: torch, coremltools, safetensors and FlexGlue (pip install git+https://github.com/cvg/FlexGlue).
FlexGlue downloads its checkpoint (flexglue.pth) on first use. FlexGlue's code and heads are MIT; its
descriptor backbone is DINOv3 ConvNeXt-B under the DINOv3 License, which applies to the exported
descriptor too (App/Splatelier/Licenses/DINOv3.txt ships with the app).

Usage: python scripts/flexglue/export_models.py [--out App/FlexGlueModels]
"""
import argparse
import copy
import os

import coremltools as ct
import numpy as np
import torch
import torch.nn.functional as F
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.frontend.torch.ops import _get_inputs
from coremltools.converters.mil.frontend.torch.torch_op_registry import register_torch_op
from flexglue import FlexGlue
from flexglue.utils import ops
from safetensors.torch import save_file
from torch import nn

H, W = 384, 512          # descriptor input; 48 x 64 = 3072 tokens
LAYERS, SPLIT = 6, 4     # transformer layers used, split after this one (early stop on low covisibility)
CHUNK = 1024             # query chunk of the per-head attention


@register_torch_op
def mt(context, node):
    """Tensor.mT (swap the last two dimensions), which coremltools does not convert."""
    x = _get_inputs(context, node, expected=1)[0]
    perm = list(range(x.rank))
    perm[-1], perm[-2] = perm[-2], perm[-1]
    context.add(mb.transpose(x=x, perm=perm, name=node.name))


class Conv(nn.Module):
    """nn.Linear as a 1x1 convolution on [1, C, 1, S]."""

    def __init__(self, lin):
        super().__init__()
        self.w = nn.Parameter(lin.weight.detach()[:, :, None, None].clone())
        self.b = nn.Parameter(lin.bias.detach().clone()) if lin.bias is not None else None

    def forward(self, x):
        return F.conv2d(x, self.w, self.b)


class ChannelLayerNorm(nn.Module):
    """nn.LayerNorm over dimension `axis` of a channels-first tensor."""

    def __init__(self, ln, axis, shape):
        super().__init__()
        self.axis, self.eps = axis, ln.eps
        self.w = nn.Parameter(ln.weight.detach().reshape(shape).clone())
        self.b = nn.Parameter(ln.bias.detach().reshape(shape).clone())

    def forward(self, x):
        d = x - x.mean(self.axis, keepdim=True)
        return d * torch.rsqrt((d * d).mean(self.axis, keepdim=True) + self.eps) * self.w + self.b


class FFN(nn.Module):
    def __init__(self, seq, message_norm):
        super().__init__()
        self.f1, self.f2 = Conv(seq[0]), Conv(seq[3])
        self.n = ChannelLayerNorm(seq[1], 1, (1, -1, 1, 1))
        self.mn = ChannelLayerNorm(message_norm, 1, (1, -1, 1, 1))

    def forward(self, x, m):
        return x + self.f2(F.gelu(self.n(self.f1(torch.cat([x, self.mn(m)], 1)))))


def attention(q, k, v, heads, dim):
    """q [1, H*D, 1, Sq], k, v [1, H*D, 1, Sk] -> [1, H*D, 1, Sq], one einsum per head and query chunk."""
    scale = dim ** -0.5
    out = []
    for qh, kh, vh in zip(q.split(dim, 1), k.transpose(1, 3).split(dim, 3), v.split(dim, 1)):
        parts = []
        for qc in qh.split(CHUNK, 3):
            w = (torch.einsum("bchq,bkhc->bkhq", qc, kh) * scale).softmax(1)   # [1, Sk, 1, Sq]
            parts.append(torch.einsum("bkhq,bchk->bchq", w, vh))              # [1, D, 1, Sq]
        out.append(torch.cat(parts, 3))
    return torch.cat(out, 1)


class SelfBlock(nn.Module):
    def __init__(self, blk, heads, cos, sin):
        super().__init__()
        blk = copy.deepcopy(blk)
        blk.use_blocked_qkv()
        self.h, self.d = heads, blk.Wqkv.out_features // 3 // heads
        self.qkv, self.out = Conv(blk.Wqkv), Conv(blk.out_proj)
        self.qn = ChannelLayerNorm(blk.q_norm, 2, (1, 1, -1, 1))
        self.kn = ChannelLayerNorm(blk.k_norm, 2, (1, 1, -1, 1))
        self.ffn = FFN(blk.ffn, blk.message_norm)
        rot = torch.zeros(self.d, self.d)
        for i in range(0, self.d, 2):   # rotate_half on interleaved pairs: (x1, x2) -> (-x2, x1)
            rot[i, i + 1], rot[i + 1, i] = -1, 1
        self.register_buffer("rot", rot.repeat(heads, 1)[:, :, None, None].clone())   # grouped 1x1 conv
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def rope(self, t):   # t [1, H, D, S]
        s = t.shape[-1]
        r = F.conv2d(t.reshape(1, self.h * self.d, 1, s), self.rot, groups=self.h).reshape(1, self.h, self.d, s)
        return t * self.cos + r * self.sin

    def forward(self, x):
        s = x.shape[-1]
        q, k, v = self.qkv(x).split(self.h * self.d, 1)
        q = self.rope(self.qn(q.reshape(1, self.h, self.d, s))).reshape(1, -1, 1, s)
        k = self.rope(self.kn(k.reshape(1, self.h, self.d, s))).reshape(1, -1, 1, s)
        return self.ffn(x, self.out(attention(q, k, v, self.h, self.d)))


class CrossBlock(nn.Module):
    def __init__(self, blk, heads):
        super().__init__()
        self.h, self.d = heads, blk.to_qk.out_features // heads
        self.qk, self.v, self.out = Conv(blk.to_qk), Conv(blk.to_v), Conv(blk.to_out)
        self.qkn = ChannelLayerNorm(blk.qk_norm, 2, (1, 1, -1, 1))
        self.ffn = FFN(blk.ffn, blk.message_norm)

    def forward(self, x0, x1):
        def qk(x):
            s = x.shape[-1]
            return self.qkn(self.qk(x).reshape(1, self.h, self.d, s)).reshape(1, -1, 1, s)
        qk0, qk1, v0, v1 = qk(x0), qk(x1), self.v(x0), self.v(x1)
        m0 = self.out(attention(qk0, qk1, v1, self.h, self.d))
        m1 = self.out(attention(qk1, qk0, v0, self.h, self.d))
        return self.ffn(x0, m0), self.ffn(x1, m1)


def layers(m, first, last, cos, sin):
    heads = m.conf.num_heads
    return nn.ModuleList(nn.ModuleList([SelfBlock(t.self_attn, heads, cos, sin), CrossBlock(t.cross_attn, heads)])
                         for t in m.transformers[first:last])


class PartA(nn.Module):
    """Input projection and layers 1..SPLIT, with the covisibility score after layer SPLIT."""

    def __init__(self, m, cos, sin):
        super().__init__()
        self.inp, self.layers, self.cov = Conv(m.input_proj), layers(m, 0, SPLIT, cos, sin), Conv(m.heads[SPLIT].covisibility)

    def forward(self, f0, f1):
        x0, x1 = self.inp(f0.flatten(2)[:, :, None]), self.inp(f1.flatten(2)[:, :, None])
        for sb, cb in self.layers:
            x0, x1 = cb(sb(x0), sb(x1))
        return x0, x1, torch.sigmoid(self.cov(torch.cat([x0, x1], 3))).mean().reshape(1)


class PartB(nn.Module):
    """Layers SPLIT+1..LAYERS, with the covisibility score after layer LAYERS."""

    def __init__(self, m, cos, sin):
        super().__init__()
        self.layers, self.cov = layers(m, SPLIT, LAYERS, cos, sin), Conv(m.heads[LAYERS].covisibility)

    def forward(self, y0, y1):
        x0, x1 = y0, y1
        for sb, cb in self.layers:
            x0, x1 = cb(sb(x0), sb(x1))
        return x0, x1, torch.sigmoid(self.cov(torch.cat([x0, x1], 3))).mean().reshape(1)


class Descriptor(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d

    def forward(self, image):
        return self.d.encode(image)


def convert(module, inputs, outputs, path):
    traced = torch.jit.trace(module, tuple(t for _, t in inputs))
    model = ct.convert(traced, inputs=[ct.TensorType(name=n, shape=t.shape) for n, t in inputs],
                       outputs=[ct.TensorType(name=n) for n in outputs], compute_precision=ct.precision.FLOAT16,
                       minimum_deployment_target=ct.target.iOS18, convert_to="mlprogram")
    model.save(path)
    print("wrote", path)
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "..", "App", "FlexGlueModels"))
    out = os.path.abspath(ap.parse_args().out)
    os.makedirs(out, exist_ok=True)
    m = FlexGlue().eval()
    h, w = H // 8, W // 8
    with torch.no_grad():
        enc = m.posenc(ops.cell_centers(h, w, scale=(max(h, w),) * 2)).detach().reshape(2, h * w, -1)
        cos, sin = (e.T[None, None].contiguous() for e in enc)   # [1, 1, D, S]
        a, b = PartA(m, cos, sin).eval(), PartB(m, cos, sin).eval()

        # The rewrite against FlexGlue's own transformer
        f0, f1 = torch.randn(1, 256, h, w), torch.randn(1, 256, h, w)
        x0, x1, e0, e1 = m.encode_pair(f0, f1)
        for blk in m.transformers[:LAYERS]:
            x0, x1 = blk(x0, x1, e0, e1)
        ya0, ya1, _ = a(f0, f1)
        z0, _, _ = b(ya0, ya1)
        diff = float((z0[0, :, 0].T - x0[0]).abs().max())
        print(f"rewritten transformer vs FlexGlue: max |diff| {diff:.2e} (scale {float(x0.abs().max()):.1f})")
        assert diff < 1e-2, "the rewritten transformer does not match FlexGlue's"

        convert(Descriptor(m.descriptor), [("image", torch.rand(1, 3, H, W))], ["features"],
                os.path.join(out, "flexglue_descriptor.mlpackage"))
        convert(a, [("f0", f0), ("f1", f1)], ["x0", "x1", "covis"], os.path.join(out, "flexglue_transformer_a.mlpackage"))
        convert(b, [("y0", ya0), ("y1", ya1)], ["x0", "x1", "covis"], os.path.join(out, "flexglue_transformer_b.mlpackage"))
        head = {k: v.detach().float().contiguous() for k, v in m.heads[LAYERS].state_dict().items()}
        save_file(head, os.path.join(out, "flexglue_head.safetensors"))
        print("wrote", os.path.join(out, "flexglue_head.safetensors"))


if __name__ == "__main__":
    main()
