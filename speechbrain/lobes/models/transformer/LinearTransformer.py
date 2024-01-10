"""Transformer implementation in the SpeechBrain style.
Authors
* Jianyuan Zhong 2020
* Samuele Cornell 2021
"""
import math
import torch
import torch.nn as nn
import speechbrain as sb
from typing import Optional
import numpy as np

from speechbrain.nnet.attention import RelPosEncXL
from speechbrain.nnet.CNN import Conv1d


def causal_linear_attention(qs, ks, vs, key_padding_mask=None):

    if key_padding_mask is not None:
        # is a binary mask of the shape B, T
        broadcast_mask = key_padding_mask.unsqueeze(-1).broadcast_to(vs.shape)
        vs = vs.masked_fill(broadcast_mask, 0.0)

    key_values = ks.unsqueeze(-1) @ vs.unsqueeze(-2)
    data_matrix = torch.cumsum(key_values, dim=-3)
    return (qs.unsqueeze(-2) @ data_matrix).flatten(-2, -1)


def linear_attention(qs, ks, vs, key_padding_mask=None):

    if key_padding_mask is not None:
        # is a binary mask of the shape B, T
        broadcast_mask = key_padding_mask.unsqueeze(-1).broadcast_to(vs.shape)
        vs = vs.masked_fill_(broadcast_mask, 0.0)

    key_values = ks.unsqueeze(-1) @ vs.unsqueeze(-2)
    data_matrix = torch.sum(key_values, dim=-3, keepdim=True)
    return (qs.unsqueeze(-2) @ data_matrix).flatten(-2, -1)


class GenericMultiHeadedAttention(nn.Module):
    """ The class wraps an attention function for MultiHeadedAttention

        Arguments
        ----------
        d_model: int
            total number of features in input vectors
        nhead : int
            parallel attention heads.
        attention_fn : supports __apply__
            attention function that takes batched queries, keys and values
        kdim : int (optional)
            total number of features in key
        vdim : int (optional)
            total number of features in value
        """

    def __init__(
            self,
            d_model=None,
            nhead=None,
            attention_fn=None,
            kdim=None,
            vdim=None
    ):
        super().__init__()

        self.nhead = nhead
        self.d_model = d_model
        self.attention_fn = attention_fn

        self.kdim = kdim if kdim is not None else d_model  # to replicate nn.MultiheadAttention
        self.vdim = vdim if vdim is not None else d_model  # to replicate nn.MultiheadAttention

        assert self.kdim % self.nhead == 0, "number of heads must divide key-dim evenly"
        assert self.vdim % self.nhead == 0, "number of heads must divide value-dim evenly"

        self.query_proj = nn.Parameter(torch.randn(1, self.d_model, self.kdim))
        self.key_proj = nn.Parameter(torch.randn(1, self.d_model, self.kdim))
        self.value_proj = nn.Parameter(torch.randn(1, self.d_model, self.vdim))
        self.output_proj = nn.Parameter(torch.randn(1, self.vdim, self.d_model))

    def forward(
            self,
            query,
            key,
            value,
            key_padding_mask=None
    ):
        """
            Arguments
            ----------
            query : torch.Tensor
                (B, L, E) where L is the target sequence length,
                B is the batch size, E is the embedding dimension.
            key : torch.Tensor
                (B, S, E) where S is the source sequence length,
                B is the batch size, E is the embedding dimension.
            value : torch.Tensor
                (B, S, E) where S is the source sequence length,
                B is the batch size, E is the embedding dimension.
            """

        B, L, E = query.shape
        _, S, _ = key.shape
        H, E_q, E_k, E_v = self.nhead, self.kdim // self.nhead, self.kdim // self.nhead, self.vdim // self.nhead

        # Apply head projections to query, keys and values
        qs = (query @ self.query_proj).view(B, L, H, E_q).transpose(-3, -2)  # -> B, H, L, E_q
        ks = (key   @   self.key_proj).view(B, S, H, E_k).transpose(-3, -2)  # -> B, H, S, E_k
        vs = (value @ self.value_proj).view(B, S, H, E_v).transpose(-3, -2)  # -> B, H, S, E_v

        # duplicate mask for each head (if there is a mask)
        if key_padding_mask is not None:
            head_mask = key_padding_mask.unsqueeze(-2).broadcast_to(ks.shape[0:3])
            head_mask = head_mask.flatten(0, 1)
        else:
            head_mask = None

        # Merge head dimension into batch dimension
        qs = qs.flatten(0, 1)
        ks = ks.flatten(0, 1)
        vs = vs.flatten(0, 1)

        # Apply attention function
        output = self.attention_fn(qs, ks, vs, head_mask)

        # Reshape to reintroduce head dimension
        output = output.reshape(B, H, L, E_v)

        # Project back up to model dim
        output = output.transpose(-3, -2).flatten(-2, -1)  # -> B, L, H*E_v
        output = output @ self.output_proj

        return output, None  # currently doesn't support returning attention weights


class MultiHeadedLinearAttention(GenericMultiHeadedAttention):
    """ The class implements Multi-headed linear attention from

        Arguments
        ----------
        d_model: int
            total number of features in input vectors
        nhead : int
            parallel attention heads.
        kdim : int
            total number of features in key (default: None).
        vdim : int
            total number of features in value (default: None).
    """

    def __init__(
            self,
            d_model=None,
            nhead=None,
            kdim=None,
            vdim=None,
            causal=True
    ):

        if causal:
            attention_fn = causal_linear_attention
        else:
            attention_fn = linear_attention

        super().__init__(d_model=d_model, nhead=nhead, attention_fn=attention_fn, kdim=kdim, vdim=vdim)


class LinearTransformerEncoderLayer(nn.Module):
    """This is an implementation of self-attention encoder layer.
    Arguments
    ----------
    d_ffn: int, optional
        The dimension of the feedforward network model hidden layer.
    nhead: int
        The number of heads in the multi-head attention models (default=8).
    d_model: int
        The number of expected features in the encoder/decoder inputs (default=512).
    kdim: int, optional
        Dimension of the key.
    vdim: int, optional
        Dimension of the value.
    dropout: int, optional
        The dropout value.
    activation: torch.nn.Module, optional
        The activation function for Feed-Forward Network layer,
        e.g., relu or gelu or swish.
    normalize_before: bool, optional
        Whether normalization should be applied before or after MHA or FFN in Transformer layers.
        Defaults to True as this was shown to lead to better performance and training stability.
    attention_type: str, optional
        Type of attention layer used in all Transformer or Conformer layers.
        e.g. regularMHA or RelPosMHA.
    ffn_type: str
        type of ffn: regularFFN/1dcnn
    ffn_cnn_kernel_size_list: list of int
        kernel size of 2 1d-convs if ffn_type is 1dcnn
    causal: bool, optional
        Whether the encoder should be causal or not (the decoder is always causal).
        If causal the Conformer convolutional layer is causal.
    Example
    -------
    >>> import torch
    >>> x = torch.rand((8, 60, 512))
    >>> net = TransformerEncoderLayer(512, 8, d_model=512)
    >>> output = net(x)
    >>> output[0].shape
    torch.Size([8, 60, 512])
    """

    def __init__(
            self,
            d_ffn,
            nhead,
            d_model,
            kdim=None,
            vdim=None,
            dropout=0.0,
            activation=nn.ReLU,
            normalize_before=False,
            ffn_type="regularFFN",
            ffn_cnn_kernel_size_list=[3, 3],
            causal=False,
    ):
        super().__init__()

        self.self_att = MultiHeadedLinearAttention(
            nhead=nhead,
            d_model=d_model,
            kdim=kdim,
            vdim=vdim,
            causal=causal
        )

        if ffn_type == "regularFFN":
            self.pos_ffn = sb.nnet.attention.PositionalwiseFeedForward(
                d_ffn=d_ffn,
                input_size=d_model,
                dropout=dropout,
                activation=activation,
            )
        elif ffn_type == "1dcnn":
            self.pos_ffn = nn.Sequential(
                Conv1d(
                    in_channels=d_model,
                    out_channels=d_ffn,
                    kernel_size=ffn_cnn_kernel_size_list[0],
                    padding="causal" if causal else "same",
                ),
                nn.ReLU(),
                Conv1d(
                    in_channels=d_ffn,
                    out_channels=d_model,
                    kernel_size=ffn_cnn_kernel_size_list[1],
                    padding="causal" if causal else "same",
                ),
            )

        self.norm1 = sb.nnet.normalization.LayerNorm(d_model, eps=1e-6)
        self.norm2 = sb.nnet.normalization.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

        self.normalize_before = normalize_before
        self.pos_ffn_type = ffn_type

    def forward(
            self,
            src,
            padding_mask=None
    ):
        """
        Arguments
        ----------
        src : torch.Tensor
            The sequence to the encoder layer.
        src_mask : torch.Tensor
            The areas of the src that are masked
        """

        if self.normalize_before:
            src1 = self.norm1(src)
        else:
            src1 = src

        output, attention_weights = self.self_att(
            src1,
            src1,
            src1,
            key_padding_mask=padding_mask
        )

        # add & norm
        src = src + self.dropout1(output)
        if not self.normalize_before:
            src = self.norm1(src)

        if self.normalize_before:
            src1 = self.norm2(src)
        else:
            src1 = src
        output = self.pos_ffn(src1)

        # add & norm
        output = src + self.dropout2(output)
        if not self.normalize_before:
            output = self.norm2(output)

        return output, attention_weights


class LinearTransformerEncoder(nn.Module):
    """This class implements the transformer encoder.
    Arguments
    ---------
    num_layers : int
        Number of transformer layers to include.
    nhead : int
        Number of attention heads.
    d_ffn : int
        Hidden size of self-attention Feed Forward layer.
    d_model : int
        The dimension of the input embedding.
    kdim : int
        Dimension for key (Optional).
    vdim : int
        Dimension for value (Optional).
    dropout : float
        Dropout for the encoder (Optional).
    input_module: torch class
        The module to process the source input feature to expected
        feature dimension (Optional).
    activation: torch.nn.Module, optional
        The activation function for Feed-Forward Netowrk layer,
        e.g., relu or gelu or swish.
    normalize_before: bool, optional
        Whether normalization should be applied before or after MHA or FFN in Transformer layers.
        Defaults to True as this was shown to lead to better performance and training stability.
    causal: bool, optional
        Whether the encoder should be causal or not (the decoder is always causal).
        If causal the Conformer convolutional layer is causal.
    layerdrop_prob: float
        The probability to drop an entire layer
    ffn_type: str
        type of ffn: regularFFN/1dcnn
    ffn_cnn_kernel_size_list: list of int
        conv kernel size of 2 1d-convs if ffn_type is 1dcnn
    Example
    -------
    >>> import torch
    >>> x = torch.rand((8, 60, 512))
    >>> net = TransformerEncoder(1, 8, 512, d_model=512)
    >>> output, _ = net(x)
    >>> output.shape
    torch.Size([8, 60, 512])
    """

    """
    encoder: !new:speechbrain.lobes.models.transformer.Transformer.TransformerEncoder
        d_model: !ref <embedding_dim>
        num_layers: 12
        nhead: 8
        d_ffn: 3072
        dropout: 0.1
        layerdrop_prob: !ref <encoder_layerdrop>
        normalize_before: True
        activation: !name:torch.nn.GELU
    
    """

    def __init__(
            self,
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
            layerdrop_prob=0.0,
            ffn_type="regularFFN",
            ffn_cnn_kernel_size_list=[3, 3],
    ):
        super().__init__()

        self.layers = torch.nn.ModuleList(
            [
                LinearTransformerEncoderLayer(
                    d_ffn=d_ffn,
                    nhead=nhead,
                    d_model=d_model,
                    kdim=kdim,
                    vdim=vdim,
                    dropout=dropout,
                    activation=activation,
                    normalize_before=normalize_before,
                    causal=causal,
                    ffn_type=ffn_type,
                    ffn_cnn_kernel_size_list=ffn_cnn_kernel_size_list,
                )
                for i in range(num_layers)
            ]
        )
        self.norm = sb.nnet.normalization.LayerNorm(d_model, eps=1e-6)
        self.layerdrop_prob = layerdrop_prob
        self.rng = np.random.default_rng()

    def forward(
            self,
            src,
            src_key_padding_mask=None
            # in the future we may want to support positional attention weights?
    ):
        """
        Arguments
        ----------
        src : tensor
            The sequence to the encoder layer (required).
        src_key_padding_mask : torch.Tensor
            The areas of the src that are masked
        """
        output = src
        if self.layerdrop_prob > 0.0:
            keep_probs = self.rng.random(len(self.layers))
        else:
            keep_probs = None

        attention_weights_list = []
        for i, enc_layer in enumerate(self.layers):
            if (
                    not self.training
                    or self.layerdrop_prob == 0.0
                    or keep_probs[i] > self.layerdrop_prob
            ):
                output, weights = enc_layer(output, padding_mask=src_key_padding_mask)
                attention_weights_list.append(weights)

        output = self.norm(output)
        return output, attention_weights_list


if __name__ == '__main__':
    encoder = LinearTransformerEncoder(num_layers=4, nhead=4, d_ffn=128, d_model=64, dropout=0.0)
    src = torch.randn(2, 16, 64)
    out, _ = encoder(src)
    print(out)
    toy_loss = torch.sum(out)
    print(toy_loss)
    toy_loss.backward()
    print("'backwards()' ran successfully")
