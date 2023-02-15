import tensorflow as tf
from models.Base import TensorflowModel

class VanillaRNN__Tensorflow(TensorflowModel):
    def __init__(self, input_shape, output_shape, units, normalize_layer=None, seed=941, **kwargs):
        super().__init__(input_shape, output_shape, units, normalize_layer, seed)
        
    def build(self, input_shape, output_shape, units):
        self.model = tf.keras.Sequential(layers=None, 
                                         name=self.__class__.__name__)
        self.model.add(tf.keras.Input(shape=input_shape, 
                                      name='Input_layer'))
        # Normalization
        if self.normalize_layer: self.model.add(self.normalize_layer)
        # RNN Layer 1
        self.model.add(tf.keras.layers.SimpleRNN(units=self.units[0],
                                            kernel_initializer=tf.initializers.GlorotUniform(seed=self.seed), 
                                            name='RNN_layer'))
        # FC Layer
        self.model.add(tf.keras.layers.Dense(units=self.units[1],
                                             activation='relu',
                                             kernel_initializer=tf.initializers.GlorotUniform(seed=self.seed),
                                             name='Fully_Connected_layer'))
        # Output Layer
        self.model.add(tf.keras.layers.Dense(units=output_shape, 
                                             kernel_initializer=tf.initializers.GlorotUniform(seed=self.seed),
                                             name='Output_layer'))

class BiRNN__Tensorflow(TensorflowModel):
    def __init__(self, input_shape, output_shape, units, normalize_layer=None, seed=941, **kwargs):
        super().__init__(input_shape, output_shape, units, normalize_layer, seed)
        
    def build(self, input_shape, output_shape, units):
        self.model = tf.keras.Sequential(layers=None, 
                                         name=self.__class__.__name__)
        self.model.add(tf.keras.Input(shape=input_shape, 
                                      name='Input_layer'))
        # Normalization
        if self.normalize_layer: self.model.add(self.normalize_layer)

        # BiRNN Layer 1 
        self.model.add(tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(units=self.units[0], 
                                                                               return_sequences=True,
                                                                               kernel_initializer=tf.initializers.GlorotUniform(seed=self.seed)),
                                                     name='BiRNN_layer_1'))
        # BiRNN Layer 2 
        self.model.add(tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(units=self.units[1], 
                                                                               return_sequences=True,
                                                                               kernel_initializer=tf.initializers.GlorotUniform(seed=self.seed)),
                                                     name='BiRNN_layer_2'))
        # BiRNN Layer 3 
        self.model.add(tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(units=self.units[2], 
                                                                               return_sequences=False,
                                                                               kernel_initializer=tf.initializers.GlorotUniform(seed=self.seed)),
                                                     name='BiRNN_layer_3'))
        # FC Layer
        self.model.add(tf.keras.layers.Dense(units=self.units[3],
                                             activation='relu',
                                             kernel_initializer=tf.initializers.GlorotUniform(seed=self.seed),
                                             name='Fully_Connected_layer'))

        # Output Layer
        self.model.add(tf.keras.layers.Dense(units=output_shape, 
                                             kernel_initializer=tf.initializers.GlorotUniform(seed=self.seed),
                                             name='Output_layer'))


# def BiRNN__Tensorflow(input_shape, output_size, normalize_layer=None, seed=941):
#     model = tf.keras.Sequential(layers=None, 
#                                 name='BiRNN__Tensorflow')
#     model.add(tf.keras.Input(shape=input_shape, 
#                              name='Input_layer'))
#     # Normalization
#     if normalize_layer: model.add(normalize_layer)
#     # BiRNN Layer 1 
#     model.add(tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(units=128, 
#                                                                       return_sequences=True,
#                                                                       kernel_initializer=tf.initializers.GlorotUniform(seed=seed)),
#                                             name='BiRNN_layer_1'))
#     # BiRNN Layer 2 
#     model.add(tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(units=64, 
#                                                                       return_sequences=True,
#                                                                       kernel_initializer=tf.initializers.GlorotUniform(seed=seed)),
#                                             name='BiRNN_layer_2'))
#     # BiRNN Layer 3 
#     model.add(tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(units=32, 
#                                                                       return_sequences=False,
#                                                                       kernel_initializer=tf.initializers.GlorotUniform(seed=seed)),
#                                             name='BiRNN_layer_3'))
#     # FC Layer
#     model.add(tf.keras.layers.Dense(units=32,
#                                     activation='relu',
#                                     kernel_initializer=tf.initializers.GlorotUniform(seed=seed),
#                                     name='Fully_Connected_layer'))

#     # Output Layer
#     model.add(tf.keras.layers.Dense(units=output_size, 
#                                     kernel_initializer=tf.initializers.GlorotUniform(seed=seed),
#                                     name='Output_layer'))
#     return model

# def BiRNN__Tensorflow(input_shape, output_size, normalize_layer=None, seed=941):
# 	return tf.keras.Sequential([
#         # Input layer 
#         tf.keras.Input(shape=input_shape, name='input_layer'), 

#         normalize_layer,

#         # BiLSTM Layer 1 
#         tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(128, 
#                                                            return_sequences=True, 
#                                                            kernel_initializer=tf.initializers.GlorotUniform(seed=seed)),
#                                       name='BiRNN_layer_1'), 

#         # BiLSTM Layer 2
#         tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(64, 
#                                                            return_sequences=True, 
#                                                            kernel_initializer=tf.initializers.GlorotUniform(seed=seed)),
#                                       name='BiRNN_layer_2'),          

#         # BiLSTM Layer 3
#         tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(32, 
#                                                            return_sequences=False, 
#                                                            kernel_initializer=tf.initializers.GlorotUniform(seed=seed)),
#                                       name='BiRNN_layer_3'),   
                            
#         # FC Layer 1
#         tf.keras.layers.Dense(32,
#                               activation='relu',
#                               kernel_initializer=tf.initializers.GlorotUniform(seed=seed),
#                               name='fc_layer_1'
#                               ),
        
#         # Output Layer
#         tf.keras.layers.Dense(output_size, 
#                               kernel_initializer=tf.initializers.GlorotUniform(seed=seed),
#                               name='output_layer') 
#     ],
#     name='BiRNN__Tensorflow')