# -*- coding: utf-8 -*-
"""
dino_attn.py —— 从冻结的 DINOv3（timm 实现）里读出 CLS→patch 注意力图，作为"主体在哪"的分数。

DINOv3 的 ViT 带 4 个 register token，注意力里的 sink 伪影被它们吸走，所以最后一层 CLS 对各 patch 的注意力
（各头平均）直接就是一张干净的显著性图：鸟身上高、背景低。make_crops.py 用它取框，train_vit_tok.py 用它挑 token。

timm 的 DINOv3 走的是 EVA 风格的 block（q/k 有 RoPE 和 norm），没有现成接口拿注意力矩阵，这里按同样的
公式重算 q、k 并只取 CLS 那一行，其余 token 的注意力不算，开销可以忽略。

    m = DinoV3Attention().cuda()
    attn, feats = m.maps(x)      # x: (B,3,H,W) ImageNet 归一化；attn: (B, H/16, W/16)；feats: (B, N, C) 最终 patch 特征
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DinoV3Attention(nn.Module):
    def __init__(self, model_name="vit_base_patch16_dinov3.lvd1689m", layer=-1):
        super().__init__()
        import timm
        self.m = timm.create_model(model_name, pretrained=True, num_classes=0).eval()
        for p in self.m.parameters():
            p.requires_grad_(False)
        self.layer = layer % len(self.m.blocks)
        self.npt = self.m.num_prefix_tokens                       # CLS + 4 register = 5
        for blk in self.m.blocks:
            blk.attn.fused_attn = False

    def train(self, mode=True):                                   # 永远 eval，不受外层 model.train() 影响
        return super().train(False)

    @torch.no_grad()
    def _cls_attn(self, blk, x, rope):
        """按 timm EvaAttention 的公式算 q、k，只返回 CLS 行的注意力 (B, heads, N)。"""
        a = blk.attn
        B, N, C = x.shape
        xn = blk.norm1(x)
        if a.qkv is not None:
            bias = None if a.q_bias is None else torch.cat((a.q_bias, a.k_bias, a.v_bias))
            qkv = F.linear(xn, a.qkv.weight, bias).reshape(B, N, 3, a.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
        else:
            q = a.q_proj(xn).reshape(B, N, a.num_heads, -1).transpose(1, 2)
            k = a.k_proj(xn).reshape(B, N, a.num_heads, -1).transpose(1, 2)
            v = a.v_proj(xn).reshape(B, N, a.num_heads, -1).transpose(1, 2)
        q, k = a.q_norm(q), a.k_norm(k)
        if rope is not None:
            from timm.layers import apply_rot_embed_cat
            npt, half = a.num_prefix_tokens, getattr(a, "rotate_half", False)
            q = torch.cat([q[:, :, :npt], apply_rot_embed_cat(q[:, :, npt:], rope, half=half)], dim=2).type_as(v)
            k = torch.cat([k[:, :, :npt], apply_rot_embed_cat(k[:, :, npt:], rope, half=half)], dim=2).type_as(v)
        attn = (q[:, :, :1] * a.scale) @ k.transpose(-2, -1)       # (B, heads, 1, N)
        return attn.softmax(dim=-1)[:, :, 0]

    @torch.no_grad()
    def maps(self, x):
        """返回 (CLS 注意力图 (B,gh,gw)，最终 patch 特征 (B,N,C))。强制 fp32：DINOv3 在 fp16 下注意力 logits 会溢出成 NaN。"""
        with torch.cuda.amp.autocast(enabled=False):
            m, x = self.m, x.float()
            B = x.size(0)
            h = m.patch_embed(x)
            h, rope = m._pos_embed(h)
            h = m.norm_pre(h)
            mixed = getattr(m, "rope_mixed", False)
            gh = gw = int(round((h.size(1) - self.npt) ** 0.5))
            attn = None
            for i, blk in enumerate(m.blocks):
                r = rope[i] if (mixed and rope is not None) else rope
                if i == self.layer:
                    attn = self._cls_attn(blk, h, r).mean(dim=1)[:, self.npt:]
                h = blk(h, rope=r)
            feats = m.norm(h)[:, self.npt:]
            return attn.reshape(B, gh, gw), feats
