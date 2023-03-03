import tensorflow as tf
from models.Base import TensorflowModel

from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Conv1D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import RepeatVector
from keras.layers import MaxPooling1D
from keras.layers import Bidirectional
from keras.layers import TimeDistributed
from keras.initializers import GlorotUniform
class EncoderDecoder__Tensorflow(TensorflowModel):
    """
        https://github.com/davide-burba/forecasting_models/blob/master/python_models.ipynb
    """
    def body(self):
        # Normalization
        if self.normalize_layer: self.model.add(self.normalize_layer)
        # model.add(tf.keras.layers.LSTM(256, activation='relu'))
        self.model.add(LSTM(name='LSTM_layer_1',
                            units=self.units[0],
                            kernel_initializer=GlorotUniform(seed=self.seed), 
                            activation=self.activations[0]))
        self.model.add(RepeatVector(name='RepeatVector_layer',
                                    n=self.output_shape))
        self.model.add(LSTM(name='LSTM_layer_2',
                            units=self.units[1],
                            return_sequences=True,
                            kernel_initializer=GlorotUniform(seed=self.seed), 
                            activation=self.activations[1]))
        self.model.add(TimeDistributed(name='TimeDistributed_layer',
                                       layer=Dense(units=self.output_shape, 
                                                   kernel_initializer=GlorotUniform(seed=self.seed),
                                                   activation=self.activations[2])))

class BiEncoderDecoder__Tensorflow(TensorflowModel):
    def body(self):
        # Normalization
        if self.normalize_layer: self.model.add(self.normalize_layer)
        # model.add(tf.keras.layers.LSTM(256, activation='relu'))
        self.model.add(Bidirectional(name='BiLSTM_layer_1',
                                     layer=LSTM(units=self.units[0],
                                                kernel_initializer=GlorotUniform(seed=self.seed),
                                                activation=self.activations[0])))
        self.model.add(RepeatVector(name='RepeatVector_layer',
                                    n=self.output_shape))
        self.model.add(Bidirectional(name='BiLSTM_layer_2',
                                     layer=LSTM(units=self.units[0],
                                                return_sequences=True,
                                                kernel_initializer=GlorotUniform(seed=self.seed),
                                                activation=self.activations[0])))
        self.model.add(TimeDistributed(name='TimeDistributed_layer',
                                       layer=Dense(units=self.output_shape, 
                                                   kernel_initializer=GlorotUniform(seed=self.seed),
                                                   activation=self.activations[2])))
        
# def BiEncoderDecoder__Tensorflow(input_shape, output_size, normalize_layer=None, seed=941):
#     model = tf.keras.Sequential(layers=None, name='BiEncoderDecoder__Tensorflow')
#     model.add(tf.keras.Input(shape=input_shape, name='input_layer'))
#     if normalize_layer: model.add(normalize_layer)
#     model.add(tf.keras.layers.LSTM(256, return_sequences=False))
#     model.add(tf.keras.layers.RepeatVector(output_size))
#     model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True)))
#     model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)))
#     model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(output_size)))
#     return model

# def CNNcLSTMcEncoderDecoder__Tensorflow(input_shape, output_size, normalize_layer=None, seed=941):
#     """
#         https://github.com/davide-burba/forecasting_models/blob/master/python_models.ipynb
#     """
#     model = tf.keras.Sequential(layers=None, name='CNNcLSTMcEncoderDecoder__Tensorflow')
#     model.add(tf.keras.Input(shape=input_shape, name='input_layer'))
#     if normalize_layer: model.add(normalize_layer)
#     model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
#     model.add(tf.keras.layers.Dropout(0.1))
#     model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
#     model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
#     model.add(tf.keras.layers.Dropout(0.1))
#     model.add(tf.keras.layers.Flatten())
#     model.add(tf.keras.layers.RepeatVector(output_size))
#     model.add(tf.keras.layers.LSTM(128, activation='relu', return_sequences=True))
#     model.add(tf.keras.layers.Dropout(0.1))
#     model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(64, activation='relu')))
#     model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(output_size)))
#     return model

class CNNcLSTMcEncoderDecoder__Tensorflow(TensorflowModel):
    def __init__(self, input_shape, output_shape, units, activations, kernels, filters, dropouts, normalize_layer=None, seed=941, **kwargs):
        super().__init__(input_shape, output_shape, units, activations, dropouts, normalize_layer=normalize_layer, seed=seed)
        self.kernels = kernels
        self.filters = filters

    def body(self):
        # Normalization
        if self.normalize_layer: self.model.add(self.normalize_layer)

        self.model.add(Conv1D(filters=self.filters[0], 
                              kernel_size=self.kernels[0], 
                              activation=self.activations[0]))
        self.model.add(Dropout(rate=self.dropouts[0]))
        self.model.add(Conv1D(filters=self.filters[1], 
                              kernel_size=self.kernels[1], 
                              activation=self.activations[1]))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Dropout(rate=self.dropouts[1]))
        self.model.add(Flatten())
        self.model.add(RepeatVector(self.output_shape))
        self.model.add(LSTM(units=self.units[0], 
                            activation=self.activations[2], 
                            return_sequences=True))
        self.model.add(Dropout(rate=self.dropouts[2]))
        self.model.add(TimeDistributed(layer=Dense(units=self.units[1], 
                                                   activation=self.activations[3])))
        self.model.add(TimeDistributed(layer=Dense(self.output_shape)))