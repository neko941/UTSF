import torch
import torch.nn as nn
import torch.nn.functional as F


import tensorflow as tf
from models.Base import TensorflowModel
from keras.layers import Conv1D
from keras.layers import Dropout
from keras.layers import Multiply
from keras.layers import Add
from keras.layers import LSTM
from keras.layers import Dense
from keras.initializers import GlorotUniform

class LSTNet__Tensorflow(TensorflowModel):
    """
    LSTNet model implementation in TensorFlow.

    Args:
        input_shape: A tuple representing the input shape of the model.
        output_shape: A tuple representing the output shape of the model.
        num_channels: A list of integers representing the number of channels for each layer in the model.
        kernel_size: A list of integers representing the kernel size for each layer in the model.
        dropout_rate: A float representing the dropout rate used in the model.
        window_size: An integer representing the length of the sliding window used for input data.
        horizon: An integer representing the number of time steps to be predicted.

    Returns:
        A compiled TensorFlow model.
    """
    def __init__(self, input_shape, output_shape, units, activations, kernels, normalize_layer=None, seed=941, **kwargs):
        super().__init__(input_shape, output_shape, units, activations, normalize_layer, seed)
        self.kernels = kernels

    def body(self):
        # Normalization
        if self.normalize_layer: self.model.add(self.normalize_layer)
        
        # Add CNN layer
        self.model.add(Conv1D(name='CNN_layer',
                              filters=self.units[0],
                              kernel_size=self.kernels[0], 
                              activation=self.activations[0]))
        # self.model.add(tf.keras.layers.Dropout(rate=dropout_rate))

        # Add Residual layer
        self.model.add(Conv1D(name='Residual_layer',
                              filters=self.units[1], 
                              kernel_size=self.kernels[1], 
                              activation=self.activations[1]))
        # self.model.add(tf.keras.layers.Dropout(rate=dropout_rate))

        # Add Highway layer
        self.model.add(Conv1D(name='Highway_layer',
                              filters=self.units[2],  
                              kernel_size=self.kernels[2], 
                              activation=self.activations[2]))
        # self.model.add(Multiply())
        # self.model.add(Add())
        # self.model.add(tf.keras.layers.Dropout(rate=dropout_rate))

        # Add LSTM layer
        self.model.add(LSTM(name='LSTM_layer',
                            units=self.units[3], 
                            return_sequences=True,  
                            kernel_initializer=GlorotUniform(seed=self.seed), 
                            activation=self.activations[3]))
        # self.model.add(tf.keras.layers.Dropout(rate=dropout_rate))

        # Add Output layer
        self.model.add(Dense(name='Output_layer',
                             units=self.output_shape,  
                             kernel_initializer=GlorotUniform(seed=self.seed), 
                             activation=self.activations[4]))


"""
LSTNet: https://github.com/laiguokun/LSTNet/blob/master/models/LSTNet.py
"""

# class LSTNet__Pytorch(nn.Module):
#     def __init__(
#         self,
#         window: int,
#         m: int,
#         hidRNN: int,
#         hidCNN: int,
#         hidSkip: int,
#         CNN_kernel: int,
#         skip: int,
#         highway_window: int,
#         dropout: float,
#         output_fun: str,
#     ):
#         super(LSTNet, self).__init__()
#         self.P = window
#         self.m = m
#         self.hidR = hidRNN
#         self.hidC = hidCNN
#         self.hidS = hidSkip
#         self.Ck = CNN_kernel
#         self.skip = skip

#         # cast to int based on github issue
#         self.pt = int((self.P - self.Ck) / self.skip)
#         self.hw = highway_window
#         self.conv1 = nn.Conv2d(1, self.hidC, kernel_size=(self.Ck, self.m))
#         self.GRU1 = nn.GRU(self.hidC, self.hidR)
#         self.dropout = nn.Dropout(p=dropout)
#         if self.skip > 0:
#             self.GRUskip = nn.GRU(self.hidC, self.hidS)
#             self.linear1 = nn.Linear(self.hidR + self.skip * self.hidS, self.m)
#         else:
#             self.linear1 = nn.Linear(self.hidR, self.m)
#         if self.hw > 0:
#             self.highway = nn.Linear(self.hw, 1)
#         self.output = None
#         if output_fun == "sigmoid":
#             self.output = torch.sigmoid
#         if output_fun == "tanh":
#             self.output = torch.tanh

#     def forward(self, y_c):
#         batch_size = y_c.size(0)

#         # CNN
#         c = y_c.view(-1, 1, self.P, self.m)
#         c = F.relu(self.conv1(c))
#         c = self.dropout(c)
#         c = torch.squeeze(c, 3)

#         # RNN
#         r = c.permute(2, 0, 1).contiguous()
#         _, r = self.GRU1(r)
#         r = self.dropout(torch.squeeze(r, 0))

#         # skip-rnn

#         if self.skip > 0:
#             s = c[:, :, int(-self.pt * self.skip) :].contiguous()
#             s = s.view(batch_size, self.hidC, self.pt, self.skip)
#             s = s.permute(2, 0, 3, 1).contiguous()
#             s = s.view(self.pt, batch_size * self.skip, self.hidC)
#             _, s = self.GRUskip(s)
#             s = s.view(batch_size, self.skip * self.hidS)
#             s = self.dropout(s)
#             r = torch.cat((r, s), 1)

#         res = self.linear1(r)

#         # highway
#         if self.hw > 0:
#             z = y_c[:, -self.hw :, :]
#             z = z.permute(0, 2, 1).contiguous().view(-1, self.hw)
#             z = self.highway(z)
#             z = z.view(-1, self.m)
#             res = res + z

#         if self.output:
#             res = self.output(res)

#         return res