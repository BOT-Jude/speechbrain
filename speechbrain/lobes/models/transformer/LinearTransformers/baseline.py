import torch
from torch import nn
import speechbrain as sb

from speechbrain.lobes.models.transformer.LinearTransformers.GlobalAttentionTransformer \
    import TransformerEncoder, TransformerEncoderLayer


class TorchMHAWrapper(nn.Module):

    def __init__(
            self,
            d_model=None,
            nhead=None,
            kdim=None,
            vdim=None,
            dropout=0.0,
            bias=False,  # hmmm, we don't use bias in our implementation?
            is_causal=True
    ):
        super().__init__()

        self.causal = is_causal

        self.mha = torch.nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            bias=bias,
            kdim=kdim,
            vdim=vdim,
            batch_first=True,
        )

    def forward(
            self,
            query,
            key,
            value,
            key_padding_mask=None
    ):
        return self.mha(
            query,
            key,
            value,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            is_causal=self.causal
        )


class TestTransformerEncoder(TransformerEncoder):

    def __init__(
            self,
            num_layers,
            nhead,
            d_ffn,
            d_model=None,
            kdim=None,
            vdim=None,
            dropout=0.0,
            causal=True,
            activation=nn.ReLU,
            normalize_before=False,
            layerdrop_prob=0.0,
    ):

        assert causal is True, "GlobalAttention must be causal"

        multihead_factory = lambda: \
            TorchMHAWrapper(
                d_model=d_model,
                nhead=nhead,
                kdim=kdim,
                vdim=vdim,
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

        super().__init__(
            num_layers=num_layers,
            d_model=d_model,
            layerdrop_prob=layerdrop_prob,
            layer_factory=layer_factory
        )

