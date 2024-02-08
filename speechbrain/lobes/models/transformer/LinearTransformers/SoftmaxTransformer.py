import torch
from torch import nn
import speechbrain as sb
from speechbrain.lobes.models.transformer.LinearTransformers.Transformer import TransformerEncoderLayer, \
    TransformerEncoder, MultiHeadedAttention


def softmax_attention(qs, ks, vs, key_padding_mask=None, causal=True):

    attention_scores = qs @ ks.transpose(-2, -1)

    if key_padding_mask is not None:
        padding_mask = key_padding_mask.unsqueeze(-2).broadcast_to(attention_scores.shape)
        attention_scores = attention_scores.masked_fill(padding_mask, -torch.inf)
        # TODO: also mask queries to prevent divide by 0 in softmax

    if causal:
        # TODO: check this is the right direction for causality
        causal_mask = torch.tril(torch.ones_like(attention_scores)).bool()
        attention_scores = attention_scores.masked_fill(causal_mask, -torch.inf)

    attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)

    return {"values": attention_weights @ vs, "weights": attention_weights}


def build_softmax_transformer_encoder(
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

    attention_fn = lambda *args, **kwargs: \
        softmax_attention(
            *args,
            **kwargs,
            causal=causal
        )

    multihead_factory = lambda: \
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
            attention=multihead_factory(),
            ffn=ffn_factory(),
        )

    return TransformerEncoder(
        num_layers=num_layers,
        d_model=d_model,
        layerdrop_prob=layerdrop_prob,
        layer_factory=layer_factory
    )


if __name__ == "__main__":

    encoder = build_softmax_transformer_encoder(
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

