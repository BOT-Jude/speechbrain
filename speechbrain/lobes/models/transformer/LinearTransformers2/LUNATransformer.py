import torch
from torch import nn
import speechbrain as sb
from speechbrain.lobes.models.transformer.LinearTransformers2.SoftmaxTransformer import SoftmaxMultiHeadedAttention
from speechbrain.lobes.models.transformer.LinearTransformers2.Transformer import TransformerEncoder


class LUNATransformerEncoderLayer(nn.Module):
    def __init__(
            self,
            d_model,
            dropout=0.0,
            normalize_before=False,
            multihead_factory=None,
            ffn=None,
    ):

        super().__init__()

        self.pack_attention = multihead_factory()
        self.unpack_attention = multihead_factory()

        self.ffn = ffn

        self.norm1 = sb.nnet.normalization.LayerNorm(d_model, eps=1e-6)
        self.norm2 = sb.nnet.normalization.LayerNorm(d_model, eps=1e-6)
        self.norm3 = sb.nnet.normalization.LayerNorm(d_model, eps=1e-6)

        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

        self.normalize_before = normalize_before

    def forward(
            self,
            data,
            src_mask=None,
            src_key_padding_mask=None,
            pos_embs=None,
    ):
        """
        Arguments
        ----------
        src : torch.Tensor
            The sequence to the encoder layer.
        src_mask : torch.Tensor
            The areas of the src that are masked
        """

        src, context = data

        if self.normalize_before:
            src1 = self.norm1(src)
        else:
            src1 = src

        # --------
        packed, _ = self.pack_attention(context, src1, src1, key_padding_mask=src_key_padding_mask)
        unpacked, _ = self.unpack_attention(src1, packed, packed)
        # --------

        # add & norm
        src = src + self.dropout1(unpacked)
        if not self.normalize_before:
            src = self.norm1(src)

        if self.normalize_before:
            src1 = self.norm2(src)
        else:
            src1 = src
        output = self.ffn(src1)

        # add & norm
        output = src + self.dropout2(output)
        if not self.normalize_before:
            output = self.norm2(output)

        # add & norm LUNA context
        context = context + packed
        context = self.norm3(context)

        return (output, context)


class AddInitialContext(nn.Module):

    def __init__(self,
                 d_model,
                 luna_context_size
                 ):
        super().__init__()
        self.d_model = d_model
        self.luna_context_size = luna_context_size
        self.initial_context = nn.Parameter(torch.randn(1, luna_context_size, d_model))

    def forward(self, src):
        B, _, _ = src.shape
        return (src, self.initial_context.expand(B, self.luna_context_size, self.d_model))


def drop_context(data):
    return data[0], None


class LUNATransformerEncoder(TransformerEncoder):

    def __init__(self,
                 num_layers,
                 nhead,
                 d_ffn,
                 d_model=None,
                 kdim=None,
                 vdim=None,
                 dropout=0.0,
                 activation=nn.ReLU,
                 normalize_before=False,
                 layerdrop_prob=0.0,
                 luna_context_size=8
                 ):

        multihead_factory = lambda: \
            SoftmaxMultiHeadedAttention(
                d_model=d_model,
                nhead=nhead,
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

        layer_factory = lambda i: \
            LUNATransformerEncoderLayer(
                d_model=d_model,
                dropout=dropout,
                normalize_before=normalize_before,
                multihead_factory=multihead_factory,
                ffn=ffn_factory(),
            )

        add_initial_context = AddInitialContext(
            d_model=d_model,
            luna_context_size=luna_context_size
        )

        super().__init__(
            num_layers=num_layers,
            d_model=d_model,
            layerdrop_prob=layerdrop_prob,
            epilog=add_initial_context,
            layer_factory=layer_factory,
            prolog=drop_context
        )


if __name__ == "__main__":
    encoder = LUNATransformerEncoder(
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
        luna_context_size=8
    )

    src = torch.randn(2, 16, 8)
    encoder(src)
