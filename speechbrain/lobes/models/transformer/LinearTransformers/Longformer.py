import torch
from torch import nn
from transformers import LongformerSelfAttention

import speechbrain as sb

from speechbrain.lobes.models.transformer.LinearTransformers.Transformer \
    import TransformerEncoderLayer, TransformerEncoder


class LongformerSelfAttentionWrapper(nn.Module):

    def __init__(self, hidden_size, num_attention_heads, attention_window, attention_dilation, causal, attention_mode="sliding_chunks"):
        super().__init__()

        assert attention_mode in ['tvm', 'sliding_chunks', 'n2', 'sliding_chunks_no_overlap']
        self.longformer = LongformerSelfAttention(
            layer_id=1,
            config={1: {
                "hidden_size": hidden_size,
                "num_attention_heads": num_attention_heads,
                "attention_window": attention_window,
                "attention_dilation": attention_dilation,
                "autoregressive": causal,
                "attention_mode": attention_mode,
            }})

    def forward(self,
                query,
                key,
                value,
                key_padding_mask=None):


        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
    '''
    The `attention_mask` is changed in `BertModel.forward` from 0, 1, 2 to
        -ve: no attention
          0: local attention
        +ve: global attention
    '''
    assert encoder_hidden_states is None, "`encoder_hidden_states` is not supported"
    assert encoder_attention_mask is None, "`encoder_attention_mask` is not supported"



def build_longformer_transformer_encoder(
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
        layerdrop_prob=0.0
):

    multihead_factory = lambda: \
        LongformerSelfAttention()
    # takes a config and a layer id, we are going to have to rewrite the layer encoder to pass through a layer_id
    # and rewrite the encoder to pass in layer id's

    # also maybe everything should use configs cause that seems better

    ffn_factory = lambda: \
        sb.nnet.attention.PositionalwiseFeedForward(
            d_ffn=d_ffn,
            input_size=d_model,
            dropout=dropout,
            activation=activation,
        )

    layer_factory = lambda: \
        TransformerEncoderLayer(
            d_model=d_model,
            dropout=dropout,
            normalize_before=normalize_before,
            attention=multihead_factory(),
            ffn=ffn_factory(),
        )

    return TransformerEncoder(
        num_layers=num_layers,
        d_model=d_model,
        layerdrop_prob=layerdrop_prob,
        layer_factory=layer_factory
    )


if __name__ == "__main__":

    encoder = build_softmax_transformer_encoder(
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
        layerdrop_prob=0.0
    )

    src = torch.randn(2, 16, 8)
    encoder(src)
