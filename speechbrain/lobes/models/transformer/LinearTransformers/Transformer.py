import numpy as np
import torch
from torch import nn
import speechbrain as sb

from speechbrain.nnet.activations import Swish
from speechbrain.nnet.attention import RelPosEncXL
from speechbrain.nnet.CNN import Conv1d


class MultiHeadedAttention(nn.Module):
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
        H, E_k, E_v = self.nhead, self.kdim // self.nhead, self.vdim // self.nhead

        # Apply head projections to query, keys and values
        qs = (query @ self.query_proj).view(B, L, H, E_k).transpose(-3, -2)  # -> B, H, L, E_k
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
        output = self.attention_fn(qs, ks, vs, key_padding_mask=head_mask)

        # Reshape to reintroduce head dimension
        output["values"] = output["values"].reshape(B, H, L, E_v)

        # Project back up to model dim
        output["values"] = output["values"].transpose(-3, -2).flatten(-2, -1)  # -> B, L, H*E_v
        output["values"] = output["values"] @ self.output_proj

        return output


class TransformerEncoderLayer(nn.Module):
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
        The activation function for Feed-Forward Netowrk layer,
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
            src,
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

        if self.normalize_before:
            src1 = self.norm1(src)
        else:
            src1 = src

        output = self.self_att(
            src1,
            src1,
            src1,
            # attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            # pos_embs=pos_embs,
        )

        # add & norm
        src = src + self.dropout1(output["values"])
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

        output["values"] = output_values
        return output


class TransformerEncoder(nn.Module):
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
    attention_type: str, optional
        Type of attention layer used in all Transformer or Conformer layers.
        e.g. regularMHA or RelPosMHA.
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

    def __init__(
            self,
            num_layers,
            d_model=None,
            layerdrop_prob=0.0,
            layer_factory=None,
    ):
        super().__init__()

        self.layers = torch.nn.ModuleList(
            [
                layer_factory()
                for _ in range(num_layers)
            ]
        )
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

        output = {"values": src}

        if self.layerdrop_prob > 0.0:
            keep_probs = self.rng.random(len(self.layers))
        else:
            keep_probs = None

        attention_lst = []

        for i, enc_layer in enumerate(self.layers):

            if (
                    not self.training
                    or self.layerdrop_prob == 0.0
                    or keep_probs[i] > self.layerdrop_prob
            ):

                output = enc_layer(
                    output["values"],
                    src_mask=src_mask,
                    src_key_padding_mask=src_key_padding_mask,
                    pos_embs=pos_embs,
                )

                attention_lst.append(output.get("weights", None))

        output_values = self.norm(output["values"])

        return output_values, attention_lst
