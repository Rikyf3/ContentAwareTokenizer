from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
import math
from functools import partial
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Mlp

try:
    from xformers.ops import memory_efficient_attention, unbind

    XFORMERS_AVAILABLE = True
except ImportError:
    XFORMERS_AVAILABLE = False


class TokenCreation(nn.Module):
    """
        Reduces tokens to fixed amount and projects up embedding dimension: (B,N,D_in) -> (B,M,D_out)

        Args:
            embed_dim_in: Number of linear projection output channels for patches.
            embed_dim_out: Number of linear projection output channels for tokens.
            num_tokens: Number of output tokens, excluded CLS and registers.
            norm_layer: Normalization layer.
    """

    def __init__(
            self,
            embed_dim: int = 384,
            num_tokens: int = 196,
            patch_size: int = 4,
            img_size: int = 224,
            num_heads: int = 8,
            norm_layer: Optional[Callable] = partial(nn.LayerNorm, eps=1e-6),
            act_layer: Optional[Callable] = nn.GELU,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_tokens = num_tokens

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_heads = num_heads
        num_patches = int(math.pow(self.img_size // self.patch_size, 2))

        self.q = nn.Parameter(torch.zeros(1, num_tokens, num_heads, embed_dim // num_heads))
        self.kv = nn.Linear(embed_dim, 2*embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self.norm1 = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()
        self.act = act_layer() if act_layer is not None else nn.Identity()
        self.mlp = Mlp(in_features=embed_dim, hidden_features=2*embed_dim)

        trunc_normal_(self.q, std=0.02)
        self.apply(self._init_weights)

    def forward(self, x, return_attn=False):
        x = self.norm1(x)

        # Multi Head mechanism
        B, N, C = x.shape
        x = self.act(x)
        kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = self.q.permute(0, 2, 1, 3), kv[0], kv[1]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, self.num_tokens, C)
        x = self.proj(x)

        if return_attn:
            return x, attn

        return x

    @torch.no_grad()
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0.0)
            nn.init.constant_(module.weight, 1.0)


class MemEffTokenCreation(TokenCreation):
    def forward(self, x):
        if not XFORMERS_AVAILABLE:
            return super().forward(x)

        B, N, C = x.shape

        x = self.norm1(x)

        kv = self.act(x)
        kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads)
        kv_sub = kv[:, torch.randperm(kv.shape[1])[:196], :, :, :] #

        k, v = unbind(kv, 2)
        k_sub, v_sub = unbind(kv_sub, 2)

        q = self.q.expand(B, -1, -1, -1)

        q_cond = memory_efficient_attention(q, k_sub, v_sub)
        q_cond = q_cond.reshape([B, self.num_tokens, C])
        q_cond = self.mlp(q_cond)
        q_cond = q_cond.reshape([B, self.num_tokens, self.num_heads, C // self.num_heads])

        idxs = torch.randperm(k.shape[1])[:196].cuda()
        attn = q_cond.permute(0, 2, 1, 3) @ k[:, idxs, :, :].permute(0, 2, 1, 3).transpose(-2, -1)
        #attn = attn.softmax(dim=-1)
        centroids = torch.argmax(attn.mean(dim=1), dim=-1)
        attn_mask = torch.full((B, 196, 3136), -torch.inf, device="cuda")
        offsets = torch.tensor([column - 56*row for row in range(-6, 7) for column in range(-6, 7)]).cuda()
        centroids_indices = idxs[centroids]
        adjusted_indices = centroids_indices.unsqueeze(2) + offsets.unsqueeze(0).unsqueeze(0)
        adjusted_indices = torch.clip(adjusted_indices, 0, 3135)
        rows = torch.arange(0, attn_mask.size(1)).unsqueeze(-1).expand(B, -1, -1)
        expanded_rows = rows.expand(-1, -1, adjusted_indices.size(-1))
        batch_indices = torch.arange(0, B).cuda().unsqueeze(-1).unsqueeze(-1).expand(-1, 196, adjusted_indices.size(-1))
        attn_mask[batch_indices.reshape(-1), expanded_rows.reshape(-1), adjusted_indices.reshape(-1)] = 0.0
        attn_mask = attn_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

        x = memory_efficient_attention(q_cond, k, v, attn_bias=attn_mask)
        x = x.reshape([B, self.num_tokens, C])

        x = self.proj(x)

        return x
