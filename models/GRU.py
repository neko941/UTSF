import tensorflow as tf
from models.Base import TensorflowModel

class VanillaGRU__Tensorflow(TensorflowModel):
    def __init__(self, input_shape, output_shape, units, normalize_layer=None, seed=941, **kwargs):
        super().__init__(input_shape, output_shape, units, normalize_layer, seed)
        
    def build(self, input_shape, output_shape, units):
        self.model = tf.keras.Sequential(layers=None, 
                                         name=self.__class__.__name__)
        self.model.add(tf.keras.Input(shape=input_shape, 
                                      name='Input_layer'))
        # Normalization
        if self.normalize_layer: self.model.add(self.normalize_layer)

        # GRU Layer 1
        self.model.add(tf.keras.layers.GRU(units=self.units[0],
                                           kernel_initializer=tf.initializers.GlorotUniform(seed=self.seed), 
                                           name='GRU_layer'))
        # FC Layer
        self.model.add(tf.keras.layers.Dense(units=self.units[1],
                                             activation='relu',
                                             kernel_initializer=tf.initializers.GlorotUniform(seed=self.seed),
                                             name='Fully_Connected_layer'))
        # Output Layer
        self.model.add(tf.keras.layers.Dense(units=output_shape, 
                                             kernel_initializer=tf.initializers.GlorotUniform(seed=self.seed),
                                             name='Output_layer'))

# def VanillaGRU__Tensorflow(input_shape, output_size, normalize_layer=None, seed=941):
#     model = tf.keras.Sequential(layers=None, 
#                                 name='VanillaGRU__Tensorflow')
#     model.add(tf.keras.Input(shape=input_shape, 
#                              name='Input_layer'))
#     # Normalization
#     if normalize_layer: model.add(normalize_layer)
#     # GRU Layer 1
#     model.add(tf.keras.layers.GRU(units=128,
#                                    kernel_initializer=tf.initializers.GlorotUniform(seed=seed), 
#                                    name='GRU_layer'))
#     # FC Layer
#     model.add(tf.keras.layers.Dense(units=32,
#                                     # activation='custom_activation',
#                                     activation='relu',
#                                     kernel_initializer=tf.initializers.GlorotUniform(seed=seed),
#                                     name='Fully_Connected_layer'))
#     # Output Layer
#     model.add(tf.keras.layers.Dense(units=output_size, 
#                                     kernel_initializer=tf.initializers.GlorotUniform(seed=seed),
#                                     name='Output_layer'))
#     return model

# def VanillaGRU__Tensorflow(input_shape, output_size, normalize_layer=None, seed=941):
#   return tf.keras.Sequential([
#         # Input layer 
#         tf.keras.Input(shape=input_shape, name='input_layer'), 

#         normalize_layer,

#         # GRU Layer 1 
#         tf.keras.layers.GRU(128,
#                             # return_sequences=True,
#                             kernel_initializer=tf.initializers.GlorotUniform(seed=seed), 
#                             name='GRU_layer'),
#         # FC Layer 1
#         tf.keras.layers.Dense(32,
#                               activation='relu',
#                               kernel_initializer=tf.initializers.GlorotUniform(seed=seed),
#                               name='Fully_Connected_layer'
#                               ),
        
#         # Output Layer
#         tf.keras.layers.Dense(output_size, 
#                               kernel_initializer=tf.initializers.GlorotUniform(seed=seed),
#                               name='Output_layer') 
#     ],
#     name='VanillaGRU__Tensorflow')


class BiGRU__Tensorflow(TensorflowModel):
    def __init__(self, input_shape, output_shape, units, normalize_layer=None, seed=941, **kwargs):
        super().__init__(input_shape, output_shape, units, normalize_layer, seed)
        
    def build(self, input_shape, output_shape, units):
        self.model = tf.keras.Sequential(layers=None, 
                                         name=self.__class__.__name__)
        self.model.add(tf.keras.Input(shape=input_shape, 
                                      name='Input_layer'))
        # Normalization
        if self.normalize_layer: self.model.add(self.normalize_layer)

        # BiGRU Layer 1 
        self.model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=self.units[0], 
                                                                         return_sequences=True,
                                                                         kernel_initializer=tf.initializers.GlorotUniform(seed=self.seed)),
                                                     name='BiGRU_layer_1'))
        # BiGRU Layer 2 
        self.model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=self.units[1], 
                                                                         return_sequences=True,
                                                                         kernel_initializer=tf.initializers.GlorotUniform(seed=self.seed)),
                                                     name='BiGRU_layer_2'))
        # BiGRU Layer 3 
        self.model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=self.units[2], 
                                                                         return_sequences=False,
                                                                         kernel_initializer=tf.initializers.GlorotUniform(seed=self.seed)),
                                                     name='BiGRU_layer_3'))
        # FC Layer
        self.model.add(tf.keras.layers.Dense(units=self.units[3],
                                             activation='relu',
                                             kernel_initializer=tf.initializers.GlorotUniform(seed=self.seed),
                                             name='Fully_Connected_layer'))

        # Output Layer
        self.model.add(tf.keras.layers.Dense(units=output_shape, 
                                             kernel_initializer=tf.initializers.GlorotUniform(seed=self.seed),
                                             name='Output_layer'))

# def BiGRU__Tensorflow(input_shape, output_size, normalize_layer=None, seed=941):
#     model = tf.keras.Sequential(layers=None, 
#                                 name='BiGRU__Tensorflow')
#     model.add(tf.keras.Input(shape=input_shape, 
#                              name='Input_layer'))
#     # Normalization
#     if normalize_layer: model.add(normalize_layer)
#     # BiGRU Layer 1 
#     model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=128, 
#                                                                  return_sequences=True,
#                                                                  kernel_initializer=tf.initializers.GlorotUniform(seed=seed)),
#                                             name='BiGRU_layer_1'))
#     # BiGRU Layer 2 
#     model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=64, 
#                                                                  return_sequences=True,
#                                                                  kernel_initializer=tf.initializers.GlorotUniform(seed=seed)),
#                                             name='BiGRU_layer_2'))
#     # BiGRU Layer 3 
#     model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=32, 
#                                                                  return_sequences=False,
#                                                                  kernel_initializer=tf.initializers.GlorotUniform(seed=seed)),
#                                             name='BiGRU_layer_3'))
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


# def BiGRU__Tensorflow(input_shape, output_size, normalize_layer=None, seed=941):
#   return tf.keras.Sequential([
#         # Input layer 
#         tf.keras.Input(shape=input_shape, name='input_layer'), 

#         normalize_layer,

#         # BiGRU Layer 1 
#         tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128,
#                                                           return_sequences=True,
#                                                           kernel_initializer=tf.initializers.GlorotUniform(seed=seed)),
#                                       name='BiGRU_layer_1'), 

#         # BiGRU Layer 2
#         tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, 
#                                                            return_sequences=True, 
#                                                            kernel_initializer=tf.initializers.GlorotUniform(seed=seed)),
#                                       name='BiGRU_layer_2'),          

#         # BiGRU Layer 3
#         tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32, 
#                                                            return_sequences=False, 
#                                                            kernel_initializer=tf.initializers.GlorotUniform(seed=seed)),
#                                       name='BiGRU_layer_3'),   
                            
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
#     name='BiGRU__Tensorflow')