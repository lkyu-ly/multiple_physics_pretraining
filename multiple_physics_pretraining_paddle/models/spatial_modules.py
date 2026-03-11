import math
from functools import partial

import einops
import numpy as np
import paddle

try:
    from .DropPath_util import DropPath
except:
    from DropPath_util import DropPath
try:
    from .shared_modules import (MLP, ContinuousPositionBias1D,
                                 RelativePositionBias)
except:
    from shared_modules import (MLP, ContinuousPositionBias1D,
                                RelativePositionBias)


def build_space_block(params):
    if params.space_type == "axial_attention":
        return partial(
            AxialAttentionBlock,
            params.embed_dim,
            params.num_heads,
            bias_type=params.bias_type,
        )
    else:
        raise NotImplementedError


class RMSInstanceNorm2d(paddle.nn.Module):
    def __init__(self, dim, affine=True, eps=1e-08):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = paddle.nn.Parameter(paddle.ones(dim))
            self.bias = paddle.nn.Parameter(paddle.zeros(dim))

    def forward(self, x):
        """std, mean = torch.std_mean(x, dim=(-2, -1), keepdims=True)
        paddle has no std_mean"""
        std = paddle.std(x=x, axis=(-2, -1), keepdim=True)
        x = x / (std + self.eps)
        if self.affine:
            x = x * self.weight[None, :, None, None]
        return x


class SubsampledLinear(paddle.nn.Module):
    """
    Cross between a linear layer and EmbeddingBag - takes in input
    and list of indices denoting which state variables from the state
    vocab are present and only performs the linear layer on rows/cols relevant
    to those state variables

    Assumes (... C) input
    """

    def __init__(self, dim_in, dim_out, subsample_in=True):
        super().__init__()
        self.subsample_in = subsample_in
        self.dim_in = dim_in
        self.dim_out = dim_out
        temp_linear = paddle.compat.nn.Linear(dim_in, dim_out)
        self.weight = paddle.nn.Parameter(temp_linear.weight)
        self.bias = paddle.nn.Parameter(temp_linear.bias)

    def forward(self, x, labels):
        labels = labels[0]
        label_size = len(labels)
        if self.subsample_in:
            scale = (self.dim_in / label_size) ** 0.5
            x = scale * paddle.compat.nn.functional.linear(
                x, self.weight[:, labels], self.bias
            )
        else:
            x = paddle.compat.nn.functional.linear(
                x, self.weight[labels], self.bias[labels]
            )
        return x


class hMLP_stem(paddle.nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, patch_size=(16, 16), in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.in_proj = paddle.nn.Sequential(
            *[
                paddle.nn.Conv2d(
                    in_chans, embed_dim // 4, kernel_size=4, stride=4, bias=False
                ),
                RMSInstanceNorm2d(embed_dim // 4, affine=True),
                paddle.nn.GELU(),
                paddle.nn.Conv2d(
                    embed_dim // 4, embed_dim // 4, kernel_size=2, stride=2, bias=False
                ),
                RMSInstanceNorm2d(embed_dim // 4, affine=True),
                paddle.nn.GELU(),
                paddle.nn.Conv2d(
                    embed_dim // 4, embed_dim, kernel_size=2, stride=2, bias=False
                ),
                RMSInstanceNorm2d(embed_dim, affine=True),
            ]
        )

    def forward(self, x):
        x = self.in_proj(x)
        return x


class hMLP_output(paddle.nn.Module):
    """Patch to Image De-bedding"""

    def __init__(self, patch_size=(16, 16), out_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.out_chans = out_chans
        self.embed_dim = embed_dim
        self.out_proj = paddle.nn.Sequential(
            *[
                paddle.nn.Conv2DTranspose(
                    in_channels=embed_dim,
                    out_channels=embed_dim // 4,
                    kernel_size=2,
                    stride=2,
                    bias_attr=False,
                ),
                RMSInstanceNorm2d(embed_dim // 4, affine=True),
                paddle.nn.GELU(),
                paddle.nn.Conv2DTranspose(
                    in_channels=embed_dim // 4,
                    out_channels=embed_dim // 4,
                    kernel_size=2,
                    stride=2,
                    bias_attr=False,
                ),
                RMSInstanceNorm2d(embed_dim // 4, affine=True),
                paddle.nn.GELU(),
            ]
        )
        out_head = paddle.nn.Conv2DTranspose(
            in_channels=embed_dim // 4, out_channels=out_chans, kernel_size=4, stride=4
        )
        self.out_kernel = paddle.nn.Parameter(out_head.weight)
        self.out_bias = paddle.nn.Parameter(out_head.bias)

    def forward(self, x, state_labels):
        x = self.out_proj(x)
        x = paddle.nn.functional.conv2d_transpose(
            x=x,
            weight=self.out_kernel[:, state_labels],
            bias=self.out_bias[state_labels],
            stride=4,
        )
        return x


class AxialAttentionBlock(paddle.nn.Module):
    def __init__(
        self,
        hidden_dim=768,
        num_heads=12,
        drop_path=0,
        layer_scale_init_value=1e-06,
        bias_type="rel",
    ):
        super().__init__()
        self.num_heads = num_heads
        self.norm1 = RMSInstanceNorm2d(hidden_dim, affine=True)
        self.norm2 = RMSInstanceNorm2d(hidden_dim, affine=True)
        self.gamma_att = (
            paddle.nn.Parameter(
                layer_scale_init_value * paddle.ones(hidden_dim), requires_grad=True
            )
            if layer_scale_init_value > 0
            else None
        )
        self.gamma_mlp = (
            paddle.nn.Parameter(
                layer_scale_init_value * paddle.ones(hidden_dim), requires_grad=True
            )
            if layer_scale_init_value > 0
            else None
        )
        self.input_head = paddle.nn.Conv2d(hidden_dim, 3 * hidden_dim, 1)
        self.output_head = paddle.nn.Conv2d(hidden_dim, hidden_dim, 1)
        self.qnorm = paddle.nn.LayerNorm(hidden_dim // num_heads)
        self.knorm = paddle.nn.LayerNorm(hidden_dim // num_heads)
        if bias_type == "none":
            self.rel_pos_bias = lambda x, y: None
        elif bias_type == "continuous":
            self.rel_pos_bias = ContinuousPositionBias1D(n_heads=num_heads)
        else:
            self.rel_pos_bias = RelativePositionBias(n_heads=num_heads)
        """"""
        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else paddle.nn.Identity()
        )
        self.mlp = MLP(hidden_dim)
        self.mlp_norm = RMSInstanceNorm2d(hidden_dim, affine=True)

    def forward(self, x, bcs):
        B, C, H, W = x.shape
        input = x.clone()
        x = self.norm1(x)
        x = self.input_head(x)
        x = einops.rearrange(x, "b (he c) h w ->  b he h w c", he=self.num_heads)
        q, k, v = x.tensor_split(num_or_indices=3, axis=-1)
        q, k = self.qnorm(q), self.knorm(k)
        qx, kx, vx = map(
            lambda x: einops.rearrange(x, "b he h w c ->  (b h) he w c"), [q, k, v]
        )
        rel_pos_bias_x = self.rel_pos_bias(W, W, bcs[0, 0])
        if rel_pos_bias_x is not None:
            xx = paddle.compat.nn.functional.scaled_dot_product_attention(
                qx, kx, vx, attn_mask=rel_pos_bias_x
            )
        else:
            xx = paddle.compat.nn.functional.scaled_dot_product_attention(
                qx.contiguous(), kx.contiguous(), vx.contiguous()
            )
        xx = einops.rearrange(xx, "(b h) he w c -> b (he c) h w", h=H)
        qy, ky, vy = map(
            lambda x: einops.rearrange(x, "b he h w c ->  (b w) he h c"), [q, k, v]
        )
        rel_pos_bias_y = self.rel_pos_bias(H, H, bcs[0, 1])
        if rel_pos_bias_y is not None:
            xy = paddle.compat.nn.functional.scaled_dot_product_attention(
                qy, ky, vy, attn_mask=rel_pos_bias_y
            )
        else:
            xy = paddle.compat.nn.functional.scaled_dot_product_attention(
                qy.contiguous(), ky.contiguous(), vy.contiguous()
            )
        xy = einops.rearrange(xy, "(b w) he h c -> b (he c) h w", w=W)
        x = (xx + xy) / 2
        x = self.norm2(x)
        x = self.output_head(x)
        x = self.drop_path(x * self.gamma_att[None, :, None, None]) + input
        input = x.clone()
        x = einops.rearrange(x, "b c h w -> b h w c")
        x = self.mlp(x)
        x = einops.rearrange(x, "b h w c -> b c h w")
        x = self.mlp_norm(x)
        output = input + self.drop_path(self.gamma_mlp[None, :, None, None] * x)
        return output
