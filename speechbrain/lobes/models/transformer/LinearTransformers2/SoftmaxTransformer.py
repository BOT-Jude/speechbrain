import torch
from torch import nn
import speechbrain as sb
from speechbrain.lobes.models.transformer.LinearTransformers2.Transformer import TransformerEncoderLayerWrapper, \
    TransformerEncoder


def softmax_attention(qs, ks, vs, key_padding_mask=None, causal=False):
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

    return attention_weights @ vs, attention_weights


class SoftmaxMultiHeadedAttention(nn.Module):

    def __init__(
            self,
            d_model=None,
            nhead=None,
            kdim=None,
            vdim=None
    ):
        super().__init__()

        self.nhead = nhead
        self.d_model = d_model
        self.attention_fn = softmax_attention

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
        ks = (key @ self.key_proj).view(B, S, H, E_k).transpose(-3, -2)  # -> B, H, S, E_k
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
        output, weights = softmax_attention(qs, ks, vs, key_padding_mask=head_mask)

        # Reshape to reintroduce head dimension
        output = output.reshape(B, H, L, E_v)

        # Project back up to model dim
        output = output.transpose(-3, -2).flatten(-2, -1)  # -> B, L, H*E_v
        output = output @ self.output_proj

        return output, weights


class SoftmaxMultiHeadedAttentionWithWeights(SoftmaxMultiHeadedAttention):

    def forward(self,
                query,
                key,
                value,
                other=[],
                key_padding_mask=None
                ):

        output, attention_weights = super()(
            query,
            key,
            value,
            key_padding_mask=None
        )
        other.append(attention_weights)

        return output, other


def add_empty_list(src):
    return (src, [])


def extract_src_and_list(data):
    return data[0], data[1]


class SoftmaxTransformerEncoder(TransformerEncoder):

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
                 layerdrop_prob=0.0
                 ):

        multihead_factory = lambda: \
            SoftmaxMultiHeadedAttentionWithWeights(
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
            TransformerEncoderLayerWrapper(
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
            epilog=add_empty_list,
            layer_factory=layer_factory,
            prolog=extract_src_and_list
        )


if __name__ == "__main__":
    encoder = SoftmaxTransformerEncoder(
        num_layers=2,
        nhead=4,
        d_ffn=16,
        d_model=8,
        kdim=None,
        vdim=None,
        dropout=0.0,
        activation=nn.ReLU,
        normalize_before=False,
        layerdrop_prob=0.0
    )

    src = torch.randn(2, 16, 8)
    encoder(src)
