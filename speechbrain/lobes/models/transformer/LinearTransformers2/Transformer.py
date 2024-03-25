import torch
from torch import nn
import speechbrain as sb
import numpy as np


class TransformerEncoderLayerWrapper(nn.Module):

    def __init__(
            self,
            d_model,
            dropout=0.0,
            normalize_before=False,
            attention=None,
            ffn=None,
    ):
        super().__init__()

        self.self_att = attention
        self.pos_ffn = ffn

        self.norm1 = sb.nnet.normalization.LayerNorm(d_model, eps=1e-6)
        self.norm2 = sb.nnet.normalization.LayerNorm(d_model, eps=1e-6)
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
            The mask for the src query for each example in the batch.
        src_key_padding_mask : torch.Tensor, optional
            The mask for the src keys for each example in the batch.
        """

        src, other = data

        if self.normalize_before:
            src1 = self.norm1(src)
        else:
            src1 = src

        output, other = self.self_att(
            src1,
            src1,
            src1,
            other=other,
            # attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            # pos_embs=pos_embs,
        )

        # add & norm
        src = src + self.dropout1(output)
        if not self.normalize_before:
            src = self.norm1(src)

        if self.normalize_before:
            src1 = self.norm2(src)
        else:
            src1 = src
        output_values = self.pos_ffn(src1)

        # add & norm
        output_values = src + self.dropout2(output_values)
        if not self.normalize_before:
            output_values = self.norm2(output_values)

        return output_values, other


class TransformerEncoder(nn.Module):

    def __init__(
            self,
            num_layers,
            d_model=None,
            layerdrop_prob=0.0,
            epilog=lambda src: (src, None),
            layer_factory=None,
            prolog=lambda data: data
    ):
        super().__init__()

        self.epilog = epilog
        self.layers = torch.nn.ModuleList(
            [
                layer_factory(i)
                for i in range(num_layers)
            ]
        )
        self.prolog = prolog

        self.norm = sb.nnet.normalization.LayerNorm(d_model, eps=1e-6)
        self.layerdrop_prob = layerdrop_prob
        self.rng = np.random.default_rng()

    def forward(
            self,
            src,
            src_mask=None,
            src_key_padding_mask=None,
            pos_embs=None,
    ):
        """
        Arguments
        ----------
        src : tensor
            The sequence to the encoder layer (required).
        src_mask : tensor
            The mask for the src sequence (optional).
        src_key_padding_mask : tensor
            The mask for the src keys per batch (optional).
        """

        data = self.epilog(src)

        if self.layerdrop_prob > 0.0:
            keep_probs = self.rng.random(len(self.layers))
        else:
            keep_probs = None

        for i, enc_layer in enumerate(self.layers):

            if (
                    not self.training
                    or self.layerdrop_prob == 0.0
                    or keep_probs[i] > self.layerdrop_prob
            ):

                data = enc_layer(
                    data,
                    src_mask=src_mask,
                    src_key_padding_mask=src_key_padding_mask,
                    pos_embs=pos_embs,
                )

        output_values, attention_weights = self.prolog(data)

        output_values = self.norm(output_values)

        return output_values, attention_weights
