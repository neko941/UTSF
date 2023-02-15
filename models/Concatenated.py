import tensorflow as tf
from models.Base import TensorflowModel

class RNNcLSTM__Tensorflow(TensorflowModel):
    def __init__(self, input_shape, output_shape, units, normalize_layer=None, seed=941, **kwargs):
        super().__init__(input_shape, output_shape, units, normalize_layer, seed)

    def build(self, input_shape, output_shape, units):
        input_layer = tf.keras.Input(shape=input_shape, name='Input_layer')

        if self.normalize_layer: x = self.normalize_layer(input_layer)
        else: x = input_layer

        for n_unit in units[:-1]:
            rnn_x = tf.keras.layers.SimpleRNN(n_unit, 
                                        return_sequences=True, 
                                        kernel_initializer=tf.initializers.GlorotUniform(seed=self.seed)
                                        )(x)
            
            lstm_x = tf.keras.layers.LSTM(n_unit, 
                                        return_sequences=True, 
                                        kernel_initializer=tf.initializers.GlorotUniform(seed=self.seed)
                                        )(x)

            x = tf.concat([rnn_x, lstm_x], axis=-1)  

        rnn_x = tf.keras.layers.SimpleRNN(n_unit, 
                                    return_sequences=False, 
                                    kernel_initializer=tf.initializers.GlorotUniform(seed=self.seed)
                                    )(x)
        
        lstm_x = tf.keras.layers.LSTM(n_unit, 
                                    return_sequences=False, 
                                    kernel_initializer=tf.initializers.GlorotUniform(seed=self.seed)
                                    )(x)

        x = tf.concat([rnn_x, lstm_x], axis=-1)  
                            
        x = tf.keras.layers.Dense(units[-1],
                                activation='relu',
                                kernel_initializer=tf.initializers.GlorotUniform(seed=self.seed),
                                name='FullyConnected_layer'
                                )(x)
        
        output_layer = tf.keras.layers.Dense(output_shape, 
                                kernel_initializer=tf.initializers.GlorotUniform(seed=self.seed),
                                name='Output_layer')(x)

        self.model = tf.keras.Model(input_layer, output_layer, name=self.__class__.__name__)

          
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
    def __init__(self, input_shape, output_shape, units, normalize_layer=None, seed=941, **kwargs):
        super().__init__(input_shape, output_shape, units, normalize_layer, seed)

    def build(self, input_shape, output_shape, units):
        input_layer = tf.keras.Input(shape=input_shape, name='Input_layer')

        if self.normalize_layer: x = self.normalize_layer(input_layer)
        else: x = input_layer

        for n_unit in units[:-1]:
            gru_x = tf.keras.layers.GRU(n_unit, 
                                    return_sequences=True, 
                                    kernel_initializer=tf.initializers.GlorotUniform(seed=self.seed)
                                    )(x)
        
            lstm_x = tf.keras.layers.LSTM(n_unit, 
                                        return_sequences=True, 
                                        kernel_initializer=tf.initializers.GlorotUniform(seed=self.seed)
                                        )(x)

            x = tf.concat([lstm_x, gru_x], axis=-1)  

        gru_x = tf.keras.layers.GRU(n_unit, 
                                    return_sequences=False, 
                                    kernel_initializer=tf.initializers.GlorotUniform(seed=self.seed)
                                    )(x)
        
        lstm_x = tf.keras.layers.LSTM(n_unit, 
                                    return_sequences=False, 
                                    kernel_initializer=tf.initializers.GlorotUniform(seed=self.seed)
                                    )(x)

        x = tf.concat([lstm_x, gru_x], axis=-1)  
                            
        x = tf.keras.layers.Dense(units[-1],
                                activation='relu',
                                kernel_initializer=tf.initializers.GlorotUniform(seed=self.seed),
                                name='FullyConnected_layer'
                                )(x)
        
        output_layer = tf.keras.layers.Dense(output_shape, 
                                kernel_initializer=tf.initializers.GlorotUniform(seed=self.seed),
                                name='Output_layer')(x)

        self.model = tf.keras.Model(input_layer, output_layer, name=self.__class__.__name__)

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