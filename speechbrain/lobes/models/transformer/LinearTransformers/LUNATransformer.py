import numpy as np
import torch
from torch import nn
import speechbrain as sb
from speechbrain.lobes.models.transformer.LinearTransformers.LinearTransformer import MultiHeadedAttention
from speechbrain.lobes.models.transformer.LinearTransformers.SoftmaxTransformer import softmax_attention


class LUNATransformerEncoderLayer(nn.Module):
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
            src,
            context,
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

        # --------
        packed = self.pack_attention(context, src1, src1, key_padding_mask=padding_mask)

        unpacked = self.unpack_attention(src1, packed["values"], packed["values"])
        # --------

        # add & norm
        src = src + self.dropout1(unpacked["values"])
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
        context = context + packed["values"]
        context = self.norm3(context)

        return {"values": output, "context": context, "weights": [packed["weights"], [unpacked["weights"]]]}


class LUNATransformerEncoder(nn.Module):
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
            d_model=None,
            layerdrop_prob=0.0,
            luna_factory=None,
            luna_context_size=8
    ):
        super().__init__()

        self.layers = torch.nn.ModuleList(
            [
                luna_factory()
                for _ in range(num_layers)
            ]
        )
        self.norm = sb.nnet.normalization.LayerNorm(d_model, eps=1e-6)
        self.layerdrop_prob = layerdrop_prob
        self.rng = np.random.default_rng()
        self.initial_context = nn.Parameter(torch.randn(1, luna_context_size, d_model))

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

        if self.layerdrop_prob > 0.0:
            keep_probs = self.rng.random(len(self.layers))
        else:
            keep_probs = None

        attention_lst = []
        batch_size = src.shape[0]
        output = {"values": src, "context": self.initial_context.expand(batch_size, -1, -1)}

        for i, enc_layer in enumerate(self.layers):

            if (
                    not self.training
                    or self.layerdrop_prob == 0.0
                    or keep_probs[i] > self.layerdrop_prob
            ):

                output = enc_layer(
                    output["values"],
                    output["context"],
                    # src_mask=src_mask,
                    padding_mask=src_key_padding_mask,
                    # pos_embs=pos_embs,
                )

                attention_lst.append(output["weights"])

        output_values = self.norm(output["values"])

        return output_values, attention_lst


def build_LUNA_transformer_encoder(
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
        causal=True,
        luna_context_size=8
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
        LUNATransformerEncoderLayer(
            d_model=d_model,
            dropout=dropout,
            normalize_before=normalize_before,
            multihead_factory=multihead_factory,
            ffn=ffn_factory(),
        )

    return LUNATransformerEncoder(
        num_layers=num_layers,
        d_model=d_model,
        layerdrop_prob=layerdrop_prob,
        luna_factory=layer_factory,
        luna_context_size=luna_context_size
    )


if __name__ == "__main__":  # fix tensor shape error on line 80

    encoder = build_LUNA_transformer_encoder(
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
        layerdrop_prob=0.0,
        luna_context_size=8
    )

    src = torch.randn(2, 16, 8)
    encoder(src)