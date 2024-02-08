import torch
from torch import nn
import speechbrain as sb
from speechbrain.lobes.models.transformer.LinearTransformers.Transformer import TransformerEncoderLayer, \
    TransformerEncoder, MultiHeadedAttention


def static_sparse_attention(qs, ks, vs, key_padding_mask=None, sparse_attention_mask=None):
    # doesn't support masks that have different behaviour for different batches (i.e. must be static)

    B, L, E = qs.shape
    _, S, _ = ks.shape

    # TODO: support key_padding_mask

    # TODO: add batch dim to mask if missing
    # convert sparse attention mask of shape L, S to hybrid L, S | B, E
    weights = torch.sparse_coo_tensor(
        indices=sparse_attention_mask.indices(),
        values=sparse_attention_mask.values().float().unsqueeze(-1).unsqueeze(-1).expand(-1, B, E),
        size=(L, S, B, E)
    )

    # cast queries to shape L, 1, B, E
    qs = qs.transpose(0, 1).unsqueeze(1)

    # cast keys to shape 1, S, B, E
    ks = ks.transpose(0, 1).unsqueeze(0)

    # compute cosine distance 'manually' using pairwise multiplication and sum
    weights = qs * weights
    weights = ks * weights
    weights = torch.sparse_coo_tensor(  # L, S | B, E -> L, S | B
        indices=weights.indices(),
        values=torch.sum(weights.values(), dim=-1),
        size=(L, S, B)
    )

    # use torch.sparse.softmax which treats undefined values as -inf
    weights = torch.sparse.softmax(weights, dim=-2)

    # use sparse-dense matrix multiplication to apply sparse weights
    values =

    # vs: B, S, V @ B, L, S -> B, L, V

    # vs: B, S, V, w: L, S | B
    # vs.t: S, V, B, w: L, S, | B


    return {"values": values, "weights": weights}


def build_sparse_transformer_encoder(
        num_layers,
        nhead,
        d_ffn,
        sparse_attention_mask,
        d_model=None,
        kdim=None,
        vdim=None,
        dropout=0.0,
        activation=nn.ReLU,
        normalize_before=False,
        layerdrop_prob=0.0
):

    attention_fn = lambda *args, **kwargs: \
        static_sparse_attention(
            *args,
            **kwargs,
            sparse_attention_mask=sparse_attention_mask
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

    encoder = build_sparse_transformer_encoder(
        num_layers=2,
        nhead=4,
        d_ffn=16,
        d_model=8,
        kdim=None,
        vdim=None,
        dropout=0.0,
        activation=nn.ReLU,
        normalize_before=False,
        layerdrop_prob=0.0,
        sparse_attention_mask=torch.ones(8, 8).bool().to_sparse()
    )

    src = torch.randn(2, 16, 8)
    encoder(src)
