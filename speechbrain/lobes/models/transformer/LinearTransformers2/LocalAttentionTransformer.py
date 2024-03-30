import torch
from torch import nn
import torch.nn.functional as F
import speechbrain as sb
from speechbrain.lobes.models.transformer.LinearTransformers2.Transformer import TransformerEncoder, \
    TransformerEncoderLayerWrapper


class LocalMultiHeadedAttention(nn.Module):

    def __init__(
            self,
            d_model=None,
            nhead=None,
            kdim=None,
            vdim=None,
            kernel_size=None
    ):
        super().__init__()

        assert kernel_size % 2 == 0, "TODO: Support odd kernel sizes"

        self.nhead = nhead
        self.kernel_size = kernel_size
        self.d_model = d_model

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
            other,
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
        K = self.kernel_size

        assert S == L, "Local attention only supported when queries and keys have the same sequence length"

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

        # -----------

        if head_mask is None:
            head_mask = torch.zeros(B*H, L, dtype=torch.bool)

        padded_ks = F.pad(ks, (0, 0, K//2, (K//2)-1), value=0.0)
        padded_head_mask = F.pad(head_mask, (K//2, (K//2)-1), value=True)
        padded_vs = F.pad(vs, (0, 0, K//2, (K//2)-1), value=0.0)

        unfold_ks = padded_ks.unfold(dimension=-2, size=K, step=1).transpose(-2, -1)
        unfold_head_mask = padded_head_mask.unfold(dimension=-1, size=K, step=1)
        unfold_vs = padded_vs.unfold(dimension=-2, size=K, step=1).transpose(-2, -1)

        weights = (qs.unsqueeze(-2) @ unfold_ks.transpose(-2, -1)).flatten(-2, -1)  # B*H, L, K
        weights = weights.masked_fill(unfold_head_mask, -torch.inf)  # remember to masked_fill BEFORE the softmax
        weights = F.softmax(weights, dim=-1)

        weighted_values = unfold_vs * weights.unsqueeze(-1)  # B*H, L, K, E_v
        output = torch.sum(weighted_values, dim=-2)  # B*H, L, E_v

        # -----------

        # Reshape to reintroduce head dimension
        output = output.reshape(B, H, L, E_v)

        # Project back up to model dim
        output = output.transpose(-3, -2).flatten(-2, -1)  # -> B, L, H*E_v
        output = output @ self.output_proj

        return output, _


class LocalTransformerEncoder(TransformerEncoder):

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
                 kernel_size=32
                 ):

        multihead_factory = lambda: \
            LocalMultiHeadedAttention(
                d_model=d_model,
                nhead=nhead,
                kdim=kdim,
                vdim=vdim,
                kernel_size=kernel_size
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
            layer_factory=layer_factory
        )


if __name__ == "__main__":
    encoder = LocalTransformerEncoder(
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
        kernel_size=32
    )

    src = torch.randn(2, 16, 8)
    encoder(src)
