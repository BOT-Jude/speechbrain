import math

import torch
from torch import nn
import torch.nn.functional as F
import speechbrain as sb
from speechbrain.lobes.models.transformer.LinearTransformers2.SoftmaxTransformer import softmax_attention
from speechbrain.lobes.models.transformer.LinearTransformers2.Transformer import TransformerEncoder, \
    TransformerEncoderLayerWrapper


class EGAMultiHeadedAttention(nn.Module):

    def __init__(
            self,
            d_model=None,
            nhead=None,
            kdim=None,
            vdim=None,
            nglobal=None
    ):
        super().__init__()

        self.nhead = nhead
        self.nglobal = nglobal
        self.d_model = d_model

        self.kdim = kdim if kdim is not None else d_model  # to replicate nn.MultiheadAttention
        self.vdim = vdim if vdim is not None else d_model  # to replicate nn.MultiheadAttention

        assert self.kdim % self.nhead == 0, "number of heads must divide key-dim evenly"
        assert self.vdim % self.nhead == 0, "number of heads must divide value-dim evenly"

        self.query_proj = nn.Parameter(torch.randn(1, self.d_model, self.kdim))
        self.key_proj = nn.Parameter(torch.randn(1, self.d_model, self.kdim))
        self.value_proj = nn.Parameter(torch.randn(1, self.d_model, self.vdim))
        self.importance_proj = nn.Parameter(torch.randn(1, self.d_model, self.nhead*self.nglobal))
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
        G = self.nglobal

        # Apply head projections to query, keys and values
        qs = (query @ self.query_proj).view(B, L, H, E_k).transpose(-3, -2)  # -> B, H, L, E_k
        ks = (key @ self.key_proj).view(B, S, H, E_k).transpose(-3, -2)  # -> B, H, S, E_k
        vs = (value @ self.value_proj).view(B, S, H, E_v).transpose(-3, -2)  # -> B, H, S, E_v

        # Compute importance for each token for each head for each global channel
        ips = (query @ self.importance_proj).view(B, L, H, G).permute(0, 2, 3, 1)  # -> B, H, G, L
        ips = ips / math.sqrt(query.shape[-1])  # normalize
        if key_padding_mask is not None:
            ips = ips.masked_fill(key_padding_mask.unsqueeze(-2).unsqueeze(-2), -torch.inf)
        ips = F.softmax(ips, dim=-1)

        # Create global queries by using softmax weighted importance of queries
        gqs = ips @ qs  # -> B, H, G, E_k

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
        gqs = gqs.flatten(0, 1)

        # find results from global queries
        global_keys, _ = softmax_attention(gqs, ks, ks, key_padding_mask=head_mask)
        global_values, _ = softmax_attention(gqs, ks, vs, key_padding_mask=head_mask)

        # use global results to compute output values
        output, _ = softmax_attention(qs, global_keys, global_values)

        # Reshape to reintroduce head dimension
        output = output.reshape(B, H, L, E_v)

        # Project back up to model dim
        output = output.transpose(-3, -2).flatten(-2, -1)  # -> B, L, H*E_v
        output = output @ self.output_proj

        return output, _


def softmax_weights(*weights):

    all_weights = torch.cat(weights, dim=-1)
    all_weights = F.softmax(all_weights, dim=-1)
    return torch.split(all_weights, split_size_or_sections=[w.shape[-1] for w in weights], dim=-1)


class EGAMultiHeadedAttention2(nn.Module):

    def __init__(
            self,
            d_model=None,
            nhead=None,
            kdim=None,
            vdim=None,
            nlocal=32,
            nglobal_keys=16,
            nglobal_queries=8,
            nstatic=1
    ):
        super().__init__()

        self.nhead = nhead
        self.nlocal = nlocal
        self.nglobal_keys = nglobal_keys
        self.nglobal_querys = nglobal_queries
        self.nstatic = nstatic

        assert nlocal != 0 or nglobal_keys != 0 or nglobal_queries != 0 or nstatic != 0, \
            "Must have some attention!?"

        self.d_model = d_model

        self.kdim = kdim if kdim is not None else d_model  # to replicate nn.MultiheadAttention
        self.vdim = vdim if vdim is not None else d_model  # to replicate nn.MultiheadAttention

        assert self.kdim % self.nhead == 0, "number of heads must divide key-dim evenly"
        assert self.vdim % self.nhead == 0, "number of heads must divide value-dim evenly"

        self.query_proj = nn.Parameter(torch.randn(1, self.d_model, self.kdim))
        self.key_proj = nn.Parameter(torch.randn(1, self.d_model, self.kdim))
        self.value_proj = nn.Parameter(torch.randn(1, self.d_model, self.vdim))

        if self.nglobal_keys != 0:
            self.key_importance_proj = nn.Parameter(torch.randn(1, self.d_model, self.nhead*self.nglobal_keys))

        if self.nglobal_querys != 0:
            self.query_importance_proj = nn.Parameter(torch.randn(1, self.d_model, self.nhead*self.nglobal_querys))

        if self.nstatic != 0:
            self.static_keys = nn.Parameter(torch.randn(1, self.nhead, self.nstatic, self.kdim//self.nhead))
            self.static_values = nn.Parameter(torch.randn(1, self.nhead, self.nstatic, self.vdim//self.nhead))

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
        Lk, Gk, Gq, Sk = self.nlocal, self.nglobal_keys, self.nglobal_querys, self.nstatic

        # Apply head projections to query, keys and values
        qs = (query @ self.query_proj).view(B, L, H, E_k).transpose(-3, -2)  # -> B, H, L, E_k
        ks = (key @ self.key_proj).view(B, S, H, E_k).transpose(-3, -2)  # -> B, H, S, E_k
        vs = (value @ self.value_proj).view(B, S, H, E_v).transpose(-3, -2)  # -> B, H, S, E_v

        # we will be computing a varying number of weights depending on the hyperparameters
        weights = []
        outputs = []

        # Compute local attention log weights
        if self.nlocal != 0:

            assert L == S, "Local attention requires target and source sequence length to be the same"

            if key_padding_mask is None:
                local_key_padding_mask = torch.zeros(B, H, L, dtype=torch.bool)
            else:
                local_key_padding_mask = key_padding_mask

            padded_ks = F.pad(ks, (0, 0, Lk//2, (Lk//2)-1), value=0.0)
            padded_mask = F.pad(local_key_padding_mask, (Lk//2, (Lk//2)-1), value=True)
            padded_vs = F.pad(vs, (0, 0, Lk//2, (Lk//2)-1), value=0.0)

            unfold_ks = padded_ks.unfold(dimension=-2, size=Lk, step=1).transpose(-2, -1)
            unfold_mask = padded_mask.unfold(dimension=-1, size=Lk, step=1)
            unfold_vs = padded_vs.unfold(dimension=-2, size=Lk, step=1).transpose(-2, -1)

            local_log_weights = (qs.unsqueeze(-2) @ unfold_ks.transpose(-2, -1)).flatten(-2, -1)  # B*H, L, Lk
            local_log_weights = local_log_weights.masked_fill(unfold_mask, -torch.inf)
            weights.append(local_log_weights)

        # Compute important keys and corresponding log weights
        if self.nglobal_keys != 0:

            key_imps = (key @ self.key_importance_proj).view(B, S, H, Gk).permute(0, 2, 3, 1)  # B, H, Gk, S
            key_imps = key_imps / math.sqrt(E)  # normalize
            if key_padding_mask is not None:
                key_imps = key_imps.masked_fill(key_padding_mask.unsqueeze(-2).unsqueeze(-2), -torch.inf)
            key_imps = F.softmax(key_imps, dim=-1)
            global_keys = key_imps @ ks

            # B, H, L, E_k @ B, H, E_k, Gk -> B, H, L, Gk
            global_keys_log_weights = qs @ global_keys.transpose(-2, -1)
            weights.append(global_keys_log_weights)

        # Compute important queries, find keys and compute corresponding log weights
        if self.nglobal_querys != 0:

            query_imps = (query @ self.query_importance_proj).view(B, S, H, Gq).permute(0, 2, 3, 1)  # B, H, Gq, L
            query_imps = query_imps / math.sqrt(E)  # normalize
            if key_padding_mask is not None:
                query_imps = query_imps.masked_fill(key_padding_mask.unsqueeze(-2).unsqueeze(-2), -torch.inf)
            query_imps = F.softmax(query_imps, dim=-1)
            global_querys = query_imps @ qs  # B, H, Gq, E_k

            intermediate_weights = global_querys @ ks.transpose(-2, -1)  # B, H, Gq, E_k @ B, H, E_k, S -> B, H, Gq, S
            intermediate_weights = intermediate_weights / math.sqrt(E_k)
            if key_padding_mask is not None:
                intermediate_weights = intermediate_weights.masked_fill(key_padding_mask.unsqueeze(-2).unsqueeze(-2), -torch.inf)

            global_querys_keys = intermediate_weights @ ks  # B, H, Gq, E_k

            # B, H, L, E_k @ B, H, E_k, Gq -> B, H, L, Gq
            global_querys_log_weights = qs @ global_querys_keys.transpose(-2, -1)
            weights.append(global_querys_log_weights)

        # compute static attention log weights
        if self.nstatic != 0:

            static_log_weights = qs @ self.static_keys.transpose(-2, -1)  # B, H, L, Sk
            weights.append(static_log_weights)

        # softmax all weights

        weights = softmax_weights(*weights)
        weights = list(weights)

        # Compute local attention output values
        if self.nlocal != 0:

            local_weights = weights.pop(0)
            weighted_values = unfold_vs * local_weights.unsqueeze(-1)  # B*H, L, K, E_v
            local_output_values = torch.sum(weighted_values, dim=-2)  # B*H, L, E_v
            outputs.append(local_output_values)

        # Compute global-key output values
        if self.nglobal_keys != 0:

            global_keys_weights = weights.pop(0)
            global_keys_values = key_imps @ vs  # B, H, Gk, E_v
            global_keys_output_values = global_keys_weights @ global_keys_values
            outputs.append(global_keys_output_values)

        # Compute global-query output values
        if self.nglobal_querys != 0:

            global_querys_weights = weights.pop(0)
            global_querys_values = intermediate_weights @ vs
            global_querys_output_values = global_querys_weights @ global_querys_values
            outputs.append(global_querys_output_values)

        # Compute static attention values
        if self.nstatic != 0:

            static_weights = weights.pop(0)
            static_output_values = static_weights @ self.static_values
            outputs.append(static_output_values)

        # Sum to get output value for each input

        output = outputs[0]
        for values in outputs[1:]:
            output += values

        # Project back up to model dim
        output = output.transpose(-3, -2).flatten(-2, -1)  # -> B, L, H*E_v
        output = output @ self.output_proj

        return output, _


class EGATransformerEncoder(TransformerEncoder):

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
                 nlocal=32,
                 nglobal_keys=16,
                 nglobal_queries=8,
                 nstatic=1
                 ):

        multihead_factory = lambda: \
            EGAMultiHeadedAttention2(
                d_model=d_model,
                nhead=nhead,
                kdim=kdim,
                vdim=vdim,
                nlocal=nlocal,
                nglobal_keys=nglobal_keys,
                nglobal_queries=nglobal_queries,
                nstatic=nstatic
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
    encoder = EGATransformerEncoder(
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
        nlocal=32,
        nglobal_keys=16,
        nglobal_queries=8,
        nstatic=1
    )

    src = torch.randn(2, 16, 8)
    encoder(src)
