import math
from functools import partial

import einops
import numpy as np
import paddle


class ContinuousPositionBias1D(paddle.nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.num_heads = n_heads
        self.cpb_mlp = paddle.nn.Sequential(
            paddle.compat.nn.Linear(1, 512, bias=True),
            paddle.nn.ReLU(),
            paddle.compat.nn.Linear(512, n_heads, bias=False),
        )

    def forward(self, h, h2, bc=0):
        dtype, device = self.cpb_mlp[0].weight.dtype, self.cpb_mlp[0].weight.device
        if bc == 0:
            relative_coords = paddle.arange(-(h - 1), h, dtype=dtype, device=device) / (
                h - 1
            )
        elif bc == 1:
            relative_coords = paddle.cat(
                [
                    paddle.arange(1, h // 2 + 1, dtype=dtype, device=device),
                    paddle.arange(
                        -(h // 2 - 1), h // 2 + 1, dtype=dtype, device=device
                    ),
                    paddle.arange(-(h // 2 - 1), 0, dtype=dtype, device=device),
                ]
            ) / (h - 1)
        coords = paddle.arange(h, dtype=paddle.float32, device=device)
        coords = coords[None, :] - coords[:, None]
        coords = coords + (h - 1)
        rel_pos_model = 16 * paddle.sigmoid(
            self.cpb_mlp(relative_coords[:, None]).squeeze()
        )
        biases = rel_pos_model[coords.long()]
        return biases.permute(2, 0, 1).unsqueeze(0).contiguous()


class RelativePositionBias(paddle.nn.Module):
    """
    From https://gist.github.com/huchenxucs/c65524185e8e35c4bcfae4059f896c16

    Implementation of T5 relative position bias - can probably do better, but starting with something known.
    """

    def __init__(self, bidirectional=True, num_buckets=32, max_distance=128, n_heads=2):
        super(RelativePositionBias, self).__init__()
        self.bidirectional = bidirectional
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.n_heads = n_heads
        self.relative_attention_bias = paddle.nn.Embedding(
            self.num_buckets, self.n_heads
        )

    @staticmethod
    def _relative_position_bucket(
        relative_position, bidirectional=True, num_buckets=32, max_distance=32
    ):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593
        Translate relative position to a bucket number for relative attention.
        The relative position is defined as memory_position - query_position, i.e.
        the distance in tokens from the attending position to the attended-to
        position.  If bidirectional=False, then positive relative positions are
        invalid.
        We use smaller buckets for small absolute relative_position and larger buckets
        for larger absolute relative_positions.  All relative positions >=max_distance
        map to the same bucket.  All relative positions <=-max_distance map to the
        same bucket.  This should allow for more graceful generalization to longer
        sequences than the model has been trained on.
        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer
        Returns:
            a Tensor with the same shape as relative_position, containing int32
            values in the range [0, num_buckets)
        """
        ret = 0
        n = -relative_position
        if bidirectional:
            num_buckets //= 2
            ret += (n < 0).to(paddle.long) * num_buckets
            n = paddle.abs(n)
        else:
            n = paddle.compat.max(n, paddle.zeros_like(n))
        max_exact = num_buckets // 2
        is_small = n < max_exact
        val_if_large = max_exact + (
            paddle.log(n.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(paddle.long)
        val_if_large = paddle.compat.min(
            val_if_large, paddle.full_like(val_if_large, num_buckets - 1)
        )
        ret += paddle.where(is_small, n, val_if_large)
        return ret

    def compute_bias(self, qlen, klen, bc=0):
        """Compute binned relative position bias"""
        context_position = paddle.arange(
            qlen, dtype=paddle.long, device=self.relative_attention_bias.weight.device
        )[:, None]
        memory_position = paddle.arange(
            klen, dtype=paddle.long, device=self.relative_attention_bias.weight.device
        )[None, :]
        relative_position = memory_position - context_position
        """
                   k
             0   1   2   3
        q   -1   0   1   2
            -2  -1   0   1
            -3  -2  -1   0
        """
        if bc == 1:
            thresh = klen // 2
            relative_position[relative_position < -thresh] = (
                relative_position[relative_position < -thresh] % thresh
            )
            relative_position[relative_position > thresh] = (
                relative_position[relative_position > thresh] % -thresh
            )
        rp_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=self.bidirectional,
            num_buckets=self.num_buckets,
        )
        rp_bucket = rp_bucket.to(self.relative_attention_bias.weight.device)
        values = self.relative_attention_bias(rp_bucket)
        values = values.permute([2, 0, 1]).unsqueeze(0)
        return values

    def forward(self, qlen, klen, bc=0):
        return self.compute_bias(qlen, klen, bc)


class MLP(paddle.nn.Module):
    def __init__(self, hidden_dim, exp_factor=4.0):
        super().__init__()
        self.fc1 = paddle.compat.nn.Linear(hidden_dim, int(hidden_dim * exp_factor))
        self.fc2 = paddle.compat.nn.Linear(int(hidden_dim * exp_factor), hidden_dim)
        self.act = paddle.nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class AbsolutePositionBias(paddle.nn.Module):
    """
    From https://gist.github.com/huchenxucs/c65524185e8e35c4bcfae4059f896c16

    Implementation of T5 relative position bias - can probably do better, but starting with something known.
    """

    def __init__(self, hidden_dim, n_tokens):
        super(AbsolutePositionBias, self).__init__()
        self.bias = paddle.nn.Parameter(paddle.randn(1, n_tokens, hidden_dim) * 0.02)

    def forward(self):
        return self.bias


class MLP(paddle.nn.Module):
    def __init__(self, hidden_dim, exp_factor=4.0):
        super().__init__()
        self.fc1 = paddle.compat.nn.Linear(hidden_dim, int(hidden_dim * exp_factor))
        self.fc2 = paddle.compat.nn.Linear(int(hidden_dim * exp_factor), hidden_dim)
        self.act = paddle.nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))
