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

class LSTNet__TensorFlow(TensorflowModel):
    """
        https://github.com/flaviagiammarino/lstnet-tensorflow/blob/d38dc67d0ebf8476c7cd20dc0e436c64b4358105/lstnet_tensorflow/model.py
    """
    def __init__(self, input_shape, output_shape, units, activations, kernels, filters, dropouts, lag, normalize_layer=None, seed=941, **kwargs):
        super().__init__(input_shape, output_shape, units, activations, dropouts, normalize_layer=normalize_layer, seed=seed)
        self.kernels = kernels
        self.filters = filters
        self.gru_units = 100
        self.skip_gru_units = 50
        self.skip = 1
        self.lag = lag
        self.regularizer = 'L2'
        self.regularization_factor = 0.01

    def build(self):
    # def build_fn(self, n_targets, n_lookback,
    #          filters,
    #          kernel_size,
    #          gru_units,
    #          skip_gru_units,
    #          skip,
    #          lags,
    #          dropout,
    #          regularizer,
    #          regularization_factor):

        '''
        Build the model, see Section 3 in the LSTNet paper.
        Parameters:
        __________________________________
        n_targets: int.
            Number of time series.
        n_lookback: int.
            Number of past time steps to use as input.
        filters: int.
            Number of filters (or channels) of the convolutional layer.
        kernel_size: int.
            Kernel size of the convolutional layer.
        gru_units: list.
            Hidden units of GRU layer.
        skip_gru_units: list.
            Hidden units of Skip GRU layer.
        skip: int.
            Number of skipped hidden cells in the Skip GRU layer.
        lags: int.
            Number of autoregressive lags.
        dropout: float.
            Dropout rate.
        regularizer: str.
            Regularizer, either 'L1', 'L2' or 'L1L2'.
        regularization_factor: float.
            Regularization factor.
        '''

        # Inputs.
        x = tf.keras.layers.Input(shape=self.input_shape)

        # Convolutional component, see Section 3.2 in the LSTNet paper.
        c = tf.keras.layers.Conv1D(filters=self.filters[0], kernel_size=self.kernels[0], activation='relu')(x)
        c = tf.keras.layers.Dropout(rate=self.dropouts[0])(c)

        # Recurrent component, see Section 3.3 in the LSTNet paper.
        r = tf.keras.layers.GRU(units=self.gru_units, activation='relu')(c)
        r = tf.keras.layers.Dropout(rate=self.dropouts[1])(r)

        # Recurrent-skip component, see Section 3.4 in the LSTNet paper.
        s = SkipGRU(units=self.skip_gru_units, activation='relu', return_sequences=True)(c)
        s = tf.keras.layers.Dropout(rate=self.dropouts[2])(s)
        # s = tf.keras.layers.Dropout(rate=self.dropouts[2])(c)
        s = tf.keras.layers.Lambda(function=lambda x: x[:, - self.skip:, :])(s)
        s = tf.keras.layers.Reshape(target_shape=(s.shape[1] * s.shape[2],))(s)
        d = tf.keras.layers.Concatenate(axis=1)([r, s])
        d = tf.keras.layers.Dense(units=self.output_shape, kernel_regularizer=kernel_regularizer(self.regularizer, self.regularization_factor))(d)

        # Autoregressive component, see Section 3.6 in the LSTNet paper.
        l = tf.keras.layers.Flatten()(x[:, - self.lag:, :])
        l = tf.keras.layers.Dense(units=self.output_shape, kernel_regularizer=kernel_regularizer(self.regularizer, self.regularization_factor))(l)

        # Outputs.
        y = tf.keras.layers.Add()([d, l])

        self.model = tf.keras.models.Model(x, y)

def kernel_regularizer(regularizer, regularization_factor):

    '''
    Define the kernel regularizer.
    Parameters:
    __________________________________
    regularizer: str.
        Regularizer, either 'L1', 'L2' or 'L1L2'.
    regularization_factor: float.
        Regularization factor.
    '''

    if regularizer == 'L1':
        return tf.keras.regularizers.L1(l1=regularization_factor)

    elif regularizer == 'L2':
        return tf.keras.regularizers.L2(l2=regularization_factor)

    elif regularizer == 'L1L2':
        return tf.keras.regularizers.L1L2(l1=regularization_factor, l2=regularization_factor)

    else:
        raise ValueError('Undefined regularizer {}.'.format(regularizer))

class SkipGRU(tf.keras.layers.Layer):

    def __init__(self,
                 units,
                 p=1,
                 activation='relu',
                 return_sequences=False,
                 return_state=False,
                 **kwargs):

        '''
        Recurrent-skip layer, see Section 3.4 in the LSTNet paper.
        
        Parameters:
        __________________________________
        units: int.
            Number of hidden units of the GRU cell.
        p: int.
            Number of skipped hidden cells.
        activation: str, function.
            Activation function, see https://www.tensorflow.org/api_docs/python/tf/keras/activations.
        return_sequences: bool.
            Whether to return the last output or the full sequence.
        return_state: bool.
            Whether to return the last state in addition to the output.
        **kwargs: See https://www.tensorflow.org/api_docs/python/tf/keras/layers/GRUCell.
        '''

        if p < 1:
            raise ValueError('The number of skipped hidden cells cannot be less than 1.')

        self.units = units
        self.p = p
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.timesteps = None
        self.cell = tf.keras.layers.GRUCell(units=units, activation=activation, **kwargs)

        super(SkipGRU, self).__init__()

    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'units': self.units,
            'p': self.p,
            'return_sequences': self.return_sequences,
            'return_state': self.return_state,
            'timesteps': self.timesteps,
            'cell': self.cell,
        })
        return config
    
    def build(self, input_shape):

        if self.timesteps is None:
            self.timesteps = input_shape[1]

            if self.p > self.timesteps:
                raise ValueError('The number of skipped hidden cells cannot be greater than the number of timesteps.')

    def call(self, inputs):

        '''
        Parameters:
        __________________________________
        inputs: tf.Tensor.
            Layer inputs, 2-dimensional tensor with shape (n_samples, filters) where n_samples is the batch size
            and filters is the number of channels of the convolutional layer.
        Returns:
        __________________________________
        outputs: tf.Tensor.
            Layer outputs, 2-dimensional tensor with shape (n_samples, units) if return_sequences == False,
            3-dimensional tensor with shape (n_samples, n_lookback, units) if return_sequences == True where
            n_samples is the batch size, n_lookback is the number of past time steps used as input and units
            is the number of hidden units of the GRU cell.
        states: tf.Tensor.
            Hidden states, 2-dimensional tensor with shape (n_samples, units) where n_samples is the batch size
            and units is the number of hidden units of the GRU cell.
        '''

        outputs = tf.TensorArray(
            element_shape=(inputs.shape[0], self.units),
            size=self.timesteps,
            dynamic_size=False,
            dtype=tf.float32,
            clear_after_read=False
        )

        states = tf.TensorArray(
            element_shape=(inputs.shape[0], self.units),
            size=self.timesteps,
            dynamic_size=False,
            dtype=tf.float32,
            clear_after_read=False
        )

        initial_states = tf.zeros(
            shape=(tf.shape(inputs)[0], self.units),
            dtype=tf.float32
        )

        for t in tf.range(self.timesteps):

            if t < self.p:
                output, state = self.cell(
                    inputs=inputs[:, t, :],
                    states=initial_states
                )

            else:
                output, state = self.cell(
                    inputs=inputs[:, t, :],
                    states=states.read(t - self.p)
                )

            outputs = outputs.write(index=t, value=output)
            states = states.write(index=t, value=state)

        outputs = tf.transpose(outputs.stack(), [1, 0, 2])
        states = tf.transpose(states.stack(), [1, 0, 2])

        if not self.return_sequences:
            outputs = outputs[:, -1, :]

        if self.return_state:
            states = states[:, -1, :]
            return outputs, states

        else:
            return outputs
        
# class LSTNet__Tensorflow(TensorflowModel):
#     """
#     LSTNet model implementation in TensorFlow.

#     Args:
#         input_shape: A tuple representing the input shape of the model.
#         output_shape: A tuple representing the output shape of the model.
#         num_channels: A list of integers representing the number of channels for each layer in the model.
#         kernel_size: A list of integers representing the kernel size for each layer in the model.
#         dropout_rate: A float representing the dropout rate used in the model.
#         window_size: An integer representing the length of the sliding window used for input data.
#         horizon: An integer representing the number of time steps to be predicted.

#     Returns:
#         A compiled TensorFlow model.
#     """
#     def __init__(self, input_shape, output_shape, units, activations, kernels, normalize_layer=None, seed=941, **kwargs):
#         super().__init__(input_shape, output_shape, units, activations, normalize_layer, seed)
#         self.kernels = kernels

#     def body(self):
#         # Normalization
#         if self.normalize_layer: self.model.add(self.normalize_layer)
        
#         # Add CNN layer
#         self.model.add(Conv1D(name='CNN_layer',
#                               filters=self.units[0],
#                               kernel_size=self.kernels[0], 
#                               activation=self.activations[0]))
#         # self.model.add(tf.keras.layers.Dropout(rate=dropout_rate))

#         # Add Residual layer
#         self.model.add(Conv1D(name='Residual_layer',
#                               filters=self.units[1], 
#                               kernel_size=self.kernels[1], 
#                               activation=self.activations[1]))
#         # self.model.add(tf.keras.layers.Dropout(rate=dropout_rate))

#         # Add Highway layer
#         self.model.add(Conv1D(name='Highway_layer',
#                               filters=self.units[2],  
#                               kernel_size=self.kernels[2], 
#                               activation=self.activations[2]))
#         # self.model.add(Multiply())
#         # self.model.add(Add())
#         # self.model.add(tf.keras.layers.Dropout(rate=dropout_rate))

#         # Add LSTM layer
#         self.model.add(LSTM(name='LSTM_layer',
#                             units=self.units[3], 
#                             return_sequences=True,  
#                             kernel_initializer=GlorotUniform(seed=self.seed), 
#                             activation=self.activations[3]))
#         # self.model.add(tf.keras.layers.Dropout(rate=dropout_rate))

#         # Add Output layer
#         self.model.add(Dense(name='Output_layer',
#                              units=self.output_shape,  
#                              kernel_initializer=GlorotUniform(seed=self.seed), 
#                              activation=self.activations[4]))


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