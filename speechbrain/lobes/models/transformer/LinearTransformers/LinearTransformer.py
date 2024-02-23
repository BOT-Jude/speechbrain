import torch
from torch import nn
import speechbrain as sb

from speechbrain.lobes.models.transformer.LinearTransformers.Transformer \
import MultiHeadedAttention, TransformerEncoderLayer, TransformerEncoder


def causal_linear_attention(qs, ks, vs, key_padding_mask=None):

    if key_padding_mask is not None:
        # is a binary mask of the shape B, T
        broadcast_mask = key_padding_mask.unsqueeze(-1).broadcast_to(vs.shape)
        vs = vs.masked_fill(broadcast_mask, 0.0)

    key_values = ks.unsqueeze(-1) @ vs.unsqueeze(-2)
    data_matrix = torch.cumsum(key_values, dim=-3)
    return {"values": (qs.unsqueeze(-2) @ data_matrix).flatten(-2, -1)}


def linear_attention(qs, ks, vs, key_padding_mask=None):

    if key_padding_mask is not None:
        # is a binary mask of the shape B, T
        broadcast_mask = key_padding_mask.unsqueeze(-1).broadcast_to(vs.shape)
        vs = vs.masked_fill(broadcast_mask, 0.0)

    key_values = ks.unsqueeze(-1) @ vs.unsqueeze(-2)
    data_matrix = torch.sum(key_values, dim=-3, keepdim=True)

    # TODO: consider masking queries, this could be causing NaN loss
    return {"values": (qs.unsqueeze(-2) @ data_matrix).flatten(-2, -1)}


def build_linear_transformer_encoder(
        num_layers,
        nhead,
        d_ffn,
        d_model=None,
        kdim=None,
        vdim=None,
        dropout=0.0,
        activation=nn.ReLU,
        normalize_before=False,
        causal=False,
        layerdrop_prob=0.0
):

    if causal:
        attention_fn = causal_linear_attention
    else:
        attention_fn = linear_attention

    attention_factory = lambda: \
        MultiHeadedAttention(
            d_model=d_model,
            nhead=nhead,
            attention_fn=attention_fn,
            kdim=kdim,
            vdim=vdim
        )

    ffn_factory = lambda: \
        sb.nnet.attention.PositionalwiseFeedForward(
            d_ffn=d_ffn,
            input_size=d_model,
            dropout=dropout,
            activation=activation,
        )

    layer_factory = lambda: \
        TransformerEncoderLayer(
            d_model=d_model,
            dropout=dropout,
            normalize_before=normalize_before,
            attention=attention_factory(),
            ffn=ffn_factory(),
        )

    return TransformerEncoder(
        num_layers=num_layers,
        d_model=d_model,
        layerdrop_prob=layerdrop_prob,
        layer_factory=layer_factory
    )


class LinearTransformerEncoder(nn.Module):

    def __init__(self,
            num_layers=2,
            nhead=4,
            d_ffn=16,
            d_model=8,
            kdim=None,
            vdim=None,
            dropout=0.0,
            activation=nn.ReLU,
            normalize_before=False,
            causal=True,
            layerdrop_prob=0.0
        ):

        super().__init__()

        self.encoder = build_linear_transformer_encoder(
            num_layers=num_layers,
            nhead=nhead,
            d_ffn=d_ffn,
            d_model=d_model,
            kdim=kdim,
            vdim=vdim,
            dropout=dropout,
            activation=activation,
            normalize_before=normalize_before,
            causal=causal,
            layerdrop_prob=layerdrop_prob
        )

        def forward(self, *args, **kwargs):
            return self.encoder(*args, **kwargs)


if __name__ == "__main__":

    encoder = build_linear_transformer_encoder(
        num_layers=2,
        nhead=4,
        d_ffn=16,
        d_model=8,
        kdim=None,
        vdim=None,
        dropout=0.0,
        activation=nn.ReLU,
        normalize_before=False,
        causal=True,
        layerdrop_prob=0.0
    )

    src = torch.randn(2, 16, 8)
    encoder(src)


