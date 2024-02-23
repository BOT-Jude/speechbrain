import torch
from torch import nn
from fairseq.models.luna import LunaEncoder

from fairseq.examples.linformer.linformer_src.multihead_linear_attention import MultiheadLinearAttention

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
    """

    def __init__(
            self,
            num_layers,
            d_model=None,
            layerdrop_prob=0.0,
            luna_context_size=8
    ):
        super().__init__()

        self.model = LunaEncoder()

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


class EncoderWrapper(nn.Module):
    """A wrapper that adds positional information,
    masks the input and then runs the latent encoder.
    Arguments
    ---------
    in_dim : int
        Last dimension of input tensor.
    embedding_dim : int
        Dimension to project input to and that the latent encoder will use.
    latent_encoder : torch.nn.module
        Initialized latent encoder object.
    positional_encoding : torch.nn.module
        Uninitialized nn.module for adding positional information, will use ``embedding_dim``.
    dropout_encoder_input : float
        Dropout on encoder input.

    Example
    -------
    >>> from speechbrain.lobes.models.transformer.Transformer import TransformerEncoder
    >>> encoder = TransformerEncoder(d_model=768, num_layers=4, nhead=4, d_ffn=1024)
    >>> wrapper = EncoderWrapper(1024, 768, encoder)
    >>> inputs = torch.rand(10, 12, 1024)
    >>> outputs = wrapper(inputs)
    >>> outputs["embeddings"].shape
    torch.Size([10, 12, 768])
    """

    def __init__(
            self,
            in_dim,
            embedding_dim,
            latent_encoder,
            positional_encoding=PositionalEncoding,
            dropout_encoder_input=0.05,
    ):
        super().__init__()
        self.input_projector = nn.Linear(in_dim, embedding_dim)
        self.latent_encoder = latent_encoder
        self.positional_encoding = positional_encoding(embedding_dim)
        self.dropout_encoder_input = nn.Dropout(dropout_encoder_input)
        self.mask_emb = nn.Parameter(
            torch.FloatTensor(embedding_dim).uniform_(), requires_grad=True
        )

    def forward(
            self, latents, wav_lens=None, padding_mask=None, mask=None,
    ):
        """
        Arguments
        ---------
        latents : torch.Tensor, shape (B, T, C)
            Batch of latent representations (AKA frames) output from latent extractor.
        wav_lens : torch.Tensor, shape (B,)
            The actual (unpadded) relative lengths for each sample of the batch (0<wav_lens<1).
        padding_mask : Torch.Tensor, shape (B, T,)
            Can be provided instead of wav_lens.
        mask : torch.Tensor, shape (B, T)
            Boolean mask which decides which latent frames will be masked.
        """
        results = {}
        T = latents.size(1)
        latents = self.input_projector(latents)
        latents = self.dropout_encoder_input(latents)

        if mask is not None:
            latents[mask] = self.mask_emb.to(latents.dtype)
            num_masked = mask.sum()
            results["num_masked"] = num_masked
            results["ratio_masked"] = num_masked / mask.numel()

        if wav_lens is not None:
            wav_lens = torch.round(wav_lens * T)
            padding_mask = ~length_to_mask(wav_lens, dtype=bool)

        latents = latents + self.positional_encoding(latents)
        feats, _ = self.latent_encoder(
            latents, src_key_padding_mask=padding_mask
        )

        results["embeddings"] = feats
        return results