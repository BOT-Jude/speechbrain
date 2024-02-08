import torch
from torch import nn
from fairseq.modules.luna_layer import LunaEncoderLayer


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