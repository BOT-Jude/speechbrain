import torch
from torch import nn
from transformers import LongformerSelfAttention
import speechbrain as sb

from speechbrain.lobes.models.transformer.LinearTransformers2.Transformer \
    import TransformerEncoderLayerWrapper, TransformerEncoder
from argparse import Namespace


class LongformerSelfAttentionWrapper(nn.Module):

    def __init__(self, hidden_size, num_attention_heads, attention_window, attention_dilation=1, global_indices=None, causal=False, attention_mode="sliding_chunks"):
        super().__init__()

        assert attention_mode in ['tvm', 'sliding_chunks', 'n2', 'sliding_chunks_no_overlap']
        self.static_global_indices = torch.tensor(global_indices, dtype=torch.int)
        self.window_size = attention_window
        config = Namespace(
            attention_probs_dropout_prob=0.0,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            attention_window=[attention_window],
            attention_dilation=[attention_dilation],
            autoregressive=causal,
            attention_mode=attention_mode)
        self.longformer = LongformerSelfAttention(config=config, layer_id=0)

    def forward(self,
                query,
                key,
                value,
                other,
                key_padding_mask=None):

        assert query is key is value, "Performing self attention, query, key and value tensors must be the same"

        B, T, E = query.shape
        W = self.window_size

        is_index_masked = torch.zeros(B, T, dtype=torch.int) if key_padding_mask is None else key_padding_mask

        is_index_global_attn = torch.zeros(T)
        is_index_global_attn[self.static_global_indices] = 1
        is_index_global_attn = is_index_global_attn.expand(B, T)

        remove_from_windowed_attention_mask = is_index_masked + is_index_global_attn

        # query = F.pad(query, (0, 0, W//2, (W//2)-1), value=0.0)

        attn, = self.longformer(
            query,
            attention_mask=remove_from_windowed_attention_mask,
            is_index_masked=is_index_masked.bool(),
            is_index_global_attn=is_index_global_attn.bool(),
            output_attentions=False
        )

        return attn, None


class LongformerTransformerEncoder(TransformerEncoder):

    def __init__(
            self,
            num_layers,
            nhead,
            attention_window,
            global_indices,
            d_ffn,
            d_model=None,
            kdim=None,
            vdim=None,
            dropout=0.0,
            activation=nn.ReLU,
            normalize_before=False,
            causal=False,
            layerdrop_prob=0.0
    ):

        multihead_factory = lambda: \
            LongformerSelfAttentionWrapper(
                hidden_size=d_model,
                num_attention_heads=nhead,
                attention_window=attention_window,
                global_indices=global_indices,
                causal=causal,
                attention_mode="sliding_chunks")

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

    encoder = LongformerTransformerEncoder(
        num_layers=2,
        nhead=4,
        d_ffn=16,
        attention_window=4,
        global_indices=[15, 14, 13, 12],
        d_model=8,
        kdim=None,
        vdim=None,
        dropout=0.0,
        activation=nn.ReLU,
        normalize_before=False,
        causal=False,
        layerdrop_prob=0.0
    )

    src = torch.randn(2, 16, 8)
    encoder(src)
