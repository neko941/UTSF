import tensorflow as tf
from models.Base import TensorflowModel

from keras.layers import GRU
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.layers import Bidirectional
from keras.initializers import GlorotUniform

class RNNcLSTM__Tensorflow(TensorflowModel):
    def build(self):
        input_layer = tf.keras.Input(shape=self.input_shape, name='Input_layer')

        if self.normalize_layer: x = self.normalize_layer(input_layer)
        else: x = input_layer

        for n_unit in self.units[:-1]:
            rnn_x = SimpleRNN(units=n_unit, 
                              return_sequences=True, 
                              kernel_initializer=GlorotUniform(seed=self.seed), 
                              activation=self.activations[0])(x)
            
            lstm_x = LSTM(units=n_unit, 
                          return_sequences=True,
                          kernel_initializer=GlorotUniform(seed=self.seed),
                          activation=self.activations[1])(x)

            x = tf.concat([rnn_x, lstm_x], axis=-1)  

        rnn_x = SimpleRNN(units=n_unit, 
                          return_sequences=False, 
                          kernel_initializer=GlorotUniform(seed=self.seed),
                          activation=self.activations[2])(x)
        
        lstm_x = LSTM(units=n_unit, 
                      return_sequences=False,
                      kernel_initializer=GlorotUniform(seed=self.seed),
                      activation=self.activations[3])(x)

        x = tf.concat([rnn_x, lstm_x], axis=-1)  
                            
        x = Dense(name='Fully_Connected_layer',
                  units=self.units[-1],
                  kernel_initializer=GlorotUniform(seed=self.seed),
                  activation=self.activations[4])(x)
        
        output_layer = Dense(name='Output_layer',
                             units=self.output_shape,
                             kernel_initializer=GlorotUniform(seed=self.seed),
                             activation=self.activations[5])(x)

        self.model = tf.keras.Model(input_layer, output_layer, name=self.__class__.__name__)
        self.model.summary()

class BiRNNcBiLSTM__Tensorflow(TensorflowModel):
    def build(self):
        input_layer = tf.keras.Input(shape=self.input_shape, name='Input_layer')

        if self.normalize_layer: x = self.normalize_layer(input_layer)
        else: x = input_layer

        for n_unit in self.units[:-1]:
            rnn_x = Bidirectional(layer=SimpleRNN(units=n_unit,
                                                  return_sequences=True,
                                                  kernel_initializer=GlorotUniform(seed=self.seed),
                                                  activation=self.activations[0]))(x)
            
            lstm_x = Bidirectional(layer=LSTM(units=n_unit,
                                              return_sequences=True,
                                              kernel_initializer=GlorotUniform(seed=self.seed),
                                              activation=self.activations[1]))(x)

            x = tf.concat([rnn_x, lstm_x], axis=-1)  

        rnn_x = Bidirectional(layer=SimpleRNN(units=n_unit, 
                                              return_sequences=False, 
                                              kernel_initializer=GlorotUniform(seed=self.seed),
                                              activation=self.activations[2]))(x)
        
        lstm_x = Bidirectional(layer=LSTM(units=n_unit, 
                                          return_sequences=False,
                                          kernel_initializer=GlorotUniform(seed=self.seed),
                                          activation=self.activations[3]))(x)

        x = tf.concat([rnn_x, lstm_x], axis=-1)  
                            
        x = Dense(name='Fully_Connected_layer',
                  units=self.units[-1],
                  kernel_initializer=GlorotUniform(seed=self.seed),
                  activation=self.activations[4])(x)
        
        output_layer = Dense(name='Output_layer',
                             units=self.output_shape,
                             kernel_initializer=GlorotUniform(seed=self.seed),
                             activation=self.activations[5])(x)

        self.model = tf.keras.Model(input_layer, output_layer, name=self.__class__.__name__)
        self.model.summary()

# def RNNcLSTM__Tensorflow(input_shape, output_size, normalize_layer=None, seed=941):
#     input_layer = tf.keras.Input(shape=input_shape, name='input_layer')

#     if normalize_layer: x = normalize_layer(input_layer)

#     for n_unit in [128, 64]:
#         rnn_x = tf.keras.layers.SimpleRNN(n_unit, 
#                                     return_sequences=True, 
#                                     kernel_initializer=tf.initializers.GlorotUniform(seed=seed)
#                                     )(x)
        
#         lstm_x = tf.keras.layers.LSTM(n_unit, 
#                                     return_sequences=True, 
#                                     kernel_initializer=tf.initializers.GlorotUniform(seed=seed)
#                                     )(x)

#         x = tf.concat([rnn_x, lstm_x], axis=-1)  

#     rnn_x = tf.keras.layers.SimpleRNN(n_unit, 
#                                 return_sequences=False, 
#                                 kernel_initializer=tf.initializers.GlorotUniform(seed=seed)
#                                 )(x)
    
#     lstm_x = tf.keras.layers.LSTM(n_unit, 
#                                 return_sequences=False, 
#                                 kernel_initializer=tf.initializers.GlorotUniform(seed=seed)
#                                 )(x)

#     x = tf.concat([rnn_x, lstm_x], axis=-1)  
                        
#     x = tf.keras.layers.Dense(32,
#                             activation='relu',
#                             kernel_initializer=tf.initializers.GlorotUniform(seed=seed),
#                             name='fc_layer_1'
#                             )(x)
    
#     output_layer = tf.keras.layers.Dense(output_size, 
#                             kernel_initializer=tf.initializers.GlorotUniform(seed=seed),
#                             name='output_layer')(x)

#     model = tf.keras.Model(input_layer, output_layer, name='RNNcLSTM__Tensorflow')

#     return model    

class LSTMcGRU__Tensorflow(TensorflowModel):
    def build(self):
        input_layer = tf.keras.Input(shape=self.input_shape, name='Input_layer')

        if self.normalize_layer: x = self.normalize_layer(input_layer)
        else: x = input_layer

        for n_unit in self.units[:-1]:
            lstm_x = LSTM(units=n_unit,
                          return_sequences=True, 
                          kernel_initializer=GlorotUniform(seed=self.seed),
                          activation=self.activations[0])(x)
            
            gru_x = GRU(units=n_unit,
                        return_sequences=True,
                        kernel_initializer=GlorotUniform(seed=self.seed),
                        activation=self.activations[1])(x)

            x = tf.concat([lstm_x, gru_x], axis=-1)  

        lstm_x = LSTM(units=n_unit,
                      return_sequences=False,
                      kernel_initializer=GlorotUniform(seed=self.seed),
                      activation=self.activations[2])(x)
        
        gru_x = GRU(units=n_unit, 
                    return_sequences=False,
                    kernel_initializer=GlorotUniform(seed=self.seed),
                    activation=self.activations[3])(x)

        x = tf.concat([lstm_x, gru_x], axis=-1)  
                            
        x = Dense(name='Fully_Connected_layer',
                  units=self.units[-1],
                  kernel_initializer=GlorotUniform(seed=self.seed),
                  activation=self.activations[4])(x)
        
        output_layer = Dense(name='Output_layer',
                             units=self.output_shape,
                             kernel_initializer=GlorotUniform(seed=self.seed),
                             activation=self.activations[5])(x)

        self.model = tf.keras.Model(input_layer, output_layer, name=self.__class__.__name__)
        self.model.summary()

class BiLSTMcBiGRU__Tensorflow(TensorflowModel):
    def build(self):
        input_layer = tf.keras.Input(shape=self.input_shape, name='Input_layer')

        if self.normalize_layer: x = self.normalize_layer(input_layer)
        else: x = input_layer

        for n_unit in self.units[:-1]:
            lstm_x = Bidirectional(layer=LSTM(units=n_unit,
                                              return_sequences=True, 
                                              kernel_initializer=GlorotUniform(seed=self.seed),
                                              activation=self.activations[0]))(x)
            
            gru_x = Bidirectional(layer=GRU(units=n_unit,
                                  return_sequences=True,
                                  kernel_initializer=GlorotUniform(seed=self.seed),
                                  activation=self.activations[1]))(x)

            x = tf.concat([lstm_x, gru_x], axis=-1)  

        lstm_x = Bidirectional(layer=LSTM(units=n_unit,
                               return_sequences=False,
                               kernel_initializer=GlorotUniform(seed=self.seed),
                               activation=self.activations[2]))(x)
        
        gru_x = Bidirectional(layer=GRU(units=n_unit, 
                              return_sequences=False,
                              kernel_initializer=GlorotUniform(seed=self.seed),
                              activation=self.activations[3]))(x)

        x = tf.concat([lstm_x, gru_x], axis=-1)  
                            
        x = Dense(name='Fully_Connected_layer',
                  units=self.units[-1],
                  kernel_initializer=GlorotUniform(seed=self.seed),
                  activation=self.activations[4])(x)
        
        output_layer = Dense(name='Output_layer',
                             units=self.output_shape,
                             kernel_initializer=GlorotUniform(seed=self.seed),
                             activation=self.activations[5])(x)

        self.model = tf.keras.Model(input_layer, output_layer, name=self.__class__.__name__)
        self.model.summary()

# def GRUcLSTM__Tensorflow(input_shape, output_size, normalize_layer=None, seed=941):
#     input_layer = tf.keras.Input(shape=input_shape, name='input_layer')

#     if normalize_layer: x = normalize_layer(input_layer)

#     for n_unit in [128, 64]:
#         gru_x = tf.keras.layers.GRU(n_unit, 
#                                     return_sequences=True, 
#                                     kernel_initializer=tf.initializers.GlorotUniform(seed=seed)
#                                     )(x)
        
#         lstm_x = tf.keras.layers.LSTM(n_unit, 
#                                     return_sequences=True, 
#                                     kernel_initializer=tf.initializers.GlorotUniform(seed=seed)
#                                     )(x)

#         x = tf.concat([lstm_x, gru_x], axis=-1)  

#     gru_x = tf.keras.layers.GRU(n_unit, 
#                                 return_sequences=False, 
#                                 kernel_initializer=tf.initializers.GlorotUniform(seed=seed)
#                                 )(x)
    
#     lstm_x = tf.keras.layers.LSTM(n_unit, 
#                                 return_sequences=False, 
#                                 kernel_initializer=tf.initializers.GlorotUniform(seed=seed)
#                                 )(x)

#     x = tf.concat([lstm_x, gru_x], axis=-1)  
                        
#     x = tf.keras.layers.Dense(32,
#                             activation='relu',
#                             kernel_initializer=tf.initializers.GlorotUniform(seed=seed),
#                             name='fc_layer_1'
#                             )(x)
    
#     output_layer = tf.keras.layers.Dense(output_size, 
#                             kernel_initializer=tf.initializers.GlorotUniform(seed=seed),
#                             name='output_layer')(x)

#     model = tf.keras.Model(input_layer, output_layer, name='GRUcLSTM__Tensorflow')

#     return model  