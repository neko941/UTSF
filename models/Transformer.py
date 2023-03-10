from layers.Transformer_EncDec import Decoder__Pytorch, DecoderLayer__Pytorch, Encoder__Pytorch, EncoderLayer__Pytorch, ConvLayer__Pytorch
from layers.Transformer_EncDec import Decoder__Tensorflow, DecoderLayer__Tensorflow, Encoder__Tensorflow, EncoderLayer__Tensorflow, ConvLayer__Tensorflow
from layers.SelfAttention_Family import FullAttention__Pytorch, AttentionLayer__Pytorch
from layers.SelfAttention_Family import FullAttention__Tensorflow, AttentionLayer__Tensorflow
from layers.Embed import DataEmbedding__Pytorch
from layers.Embed import DataEmbedding__Tensorflow

import torch
import torch.nn as nn
import tensorflow as tf

class VanillaTransformer__Pytorch(nn.Module):
    """
    Vanilla Transformer with O(L^2) complexity
    """
    def __init__(self, configs):
        super(VanillaTransformer__Pytorch, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Embedding
        self.enc_embedding = DataEmbedding__Pytorch(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.dec_embedding = DataEmbedding__Pytorch(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        # Encoder
        self.encoder = Encoder__Pytorch(
            [
                EncoderLayer__Pytorch(
                    AttentionLayer__Pytorch(
                        FullAttention__Pytorch(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder__Pytorch(
            [
                DecoderLayer__Pytorch(
                    AttentionLayer__Pytorch(
                        FullAttention__Pytorch(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer__Pytorch(
                        FullAttention__Pytorch(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, y_dec=None,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        
class _VanillaTransformer__Tensorflow(tf.keras.Model):
    """
    Vanilla Transformer with O(L^2) complexity
    """
    def __init__(self, 
                 pred_len,
                 output_attention,
                 enc_in,
                 dec_in,
                 d_model,
                 embed,
                 factor,
                 freq,
                 n_heads,
                 d_layers,
                 e_layers,
                 c_out,
                 d_ff,
                 activations,
                 dropouts):
        super(_VanillaTransformer__Tensorflow, self).__init__()
        self.pred_len = pred_len
        self.output_attention = output_attention

        # Embedding
        self.enc_embedding = DataEmbedding__Tensorflow(enc_in, d_model[0], embed, freq, dropouts[0])
        self.dec_embedding = DataEmbedding__Tensorflow(dec_in, d_model[1], embed, freq, dropouts[1])
        # Encoder
        self.encoder = Encoder__Tensorflow(
            [
                EncoderLayer__Tensorflow(
                    AttentionLayer__Tensorflow(
                        FullAttention__Tensorflow(False, 
                                                  factor, 
                                                  attention_dropout=dropouts[2],
                                                  output_attention=output_attention), 
                        d_model[2], 
                        n_heads),
                    d_model[3],
                    d_ff,
                    dropout=dropouts[3],
                    activation=activations[0]
                ) for l in range(e_layers)
            ],
            norm_layer=tf.keras.layers.LayerNormalization(epsilon=1e-6)
        )
        # Decoder
        self.decoder = Decoder__Tensorflow(
            [
                DecoderLayer__Tensorflow(
                    AttentionLayer__Tensorflow(
                        FullAttention__Tensorflow(True, factor, attention_dropout=dropouts[4], output_attention=False),
                        d_model[4], n_heads),
                    AttentionLayer__Tensorflow(
                        FullAttention__Tensorflow(False, factor, attention_dropout=dropouts[5], output_attention=False),
                        d_model[5], n_heads),
                    d_model[6],
                    d_ff,
                    dropout=dropouts[6],
                    activation=activations[1],
                )
                for l in range(d_layers)
            ],
            norm_layer=tf.keras.layers.LayerNormalization(epsilon=1e-6),
            projection=tf.keras.layers.Dense(c_out)
        )

    def call(self, x_enc, x_mark_enc, x_dec, x_mark_dec, y_dec=None,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]

from models.Base import TensorflowModel
class VanillaTransformer__Tensorflow(TensorflowModel):
    def build(self):
        self.model = _VanillaTransformer__Tensorflow(pred_len=self.output_shape,
                                                     output_attention=True,
                                                     enc_in=self.input_shape[-1],
                                                     dec_in=self.input_shape[-1],
                                                     d_model=self.units,
                                                     embed='timeF',
                                                     factor=1,
                                                     freq='d',
                                                     n_heads=8,
                                                     d_layers=1,
                                                     e_layers=2,
                                                     c_out=self.output_shape,
                                                     d_ff=2048,
                                                     activations=self.activations,
                                                     dropouts=self.dropouts)