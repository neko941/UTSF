import torch.nn as nn
import torch.nn.functional as F
from keras import layers
import tensorflow as tf

class ConvLayer__Pytorch(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer__Pytorch, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x

class ConvLayer__Tensorflow(tf.keras.Model):
    def __init__(self, c_in):
        super(ConvLayer__Tensorflow, self).__init__()
        self.downConv = layers.Conv1D(filters=c_in,
                                      kernel_size=3,
                                      padding="valid",
                                      dilation_rate=2)
        self.norm = layers.BatchNormalization()
        self.activation = layers.ELU()
        self.maxPool = layers.MaxPooling1D(pool_size=3, strides=2, padding="same")

    def call(self, x):
        x = self.downConv(tf.transpose(x, perm=[0, 2, 1]))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = tf.transpose(x, perm=[0, 2, 1])
        return x

class EncoderLayer__Pytorch(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer__Pytorch, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn

class EncoderLayer__Tensorflow(tf.keras.Model):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer__Tensorflow, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = layers.Conv1D(filters=d_ff, kernel_size=1, activation=activation)
        self.conv2 = layers.Conv1D(filters=d_model, kernel_size=1)
        self.norm1 = layers.LayerNormalization()
        self.norm2 = layers.LayerNormalization()
        self.dropout = layers.Dropout(dropout)

    def call(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.conv1(tf.transpose(y, perm=[0, 2, 1])))
        y = self.dropout(self.conv2(tf.transpose(y, perm=[0, 2, 1])))
        y = tf.transpose(y, perm=[0, 2, 1])

        return self.norm2(x + y), attn

class Encoder__Pytorch(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder__Pytorch, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns

class Encoder__Tensorflow(tf.keras.Model):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder__Tensorflow, self).__init__()
        self.attn_layers = attn_layers
        self.conv_layers = conv_layers if conv_layers is not None else None
        self.norm = norm_layer

    def call(self, x, attn_mask=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns

class DecoderLayer__Pytorch(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer__Pytorch, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)

class DecoderLayer__Tensorflow(tf.keras.Model):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer__Tensorflow, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = tf.keras.layers.Conv1D(filters=d_ff, kernel_size=1, activation=activation)
        self.conv2 = tf.keras.layers.Conv1D(filters=d_model, kernel_size=1)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            [x, x, x],
            mask=x_mask
        )[0])
        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            [x, cross, cross],
            mask=cross_mask
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.conv1(y))
        y = self.dropout(self.conv2(y))

        return self.norm3(x + y)

class Decoder__Pytorch(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder__Pytorch, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x
    
class Decoder__Tensorflow(tf.keras.Model):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder__Tensorflow, self).__init__()
        self.layers = tf.keras.layers.LayerList(layers)
        self.norm = norm_layer
        self.projection = projection

    def call(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)

        return x