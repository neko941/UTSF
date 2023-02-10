import tensorflow as tf

def VanillaRNN__Tensorflow(input_shape, output_size, normalize_layer=None, seed=941):
    model = tf.keras.Sequential(layers=None, 
                                name='VanillaRNN__Tensorflow')
    model.add(tf.keras.Input(shape=input_shape, 
                             name='Input_layer'))
    # Normalization
    if normalize_layer: model.add(normalize_layer)
    # RNN Layer 1
    model.add(tf.keras.layers.SimpleRNN(units=128,
                                   kernel_initializer=tf.initializers.GlorotUniform(seed=seed), 
                                   name='RNN_layer'))
    # FC Layer
    model.add(tf.keras.layers.Dense(units=32,
                                    # activation='custom_activation',
                                    activation='relu',
                                    kernel_initializer=tf.initializers.GlorotUniform(seed=seed),
                                    name='Fully_Connected_layer'))
    # Output Layer
    model.add(tf.keras.layers.Dense(units=output_size, 
                                    kernel_initializer=tf.initializers.GlorotUniform(seed=seed),
                                    name='Output_layer'))

    return model

# def VanillaRNN__Tensorflow(input_shape, output_size, normalize_layer=None, seed=941):
#   return tf.keras.Sequential([
#         # Input layer 
#         tf.keras.Input(shape=input_shape, name='input_layer'), 

#         normalize_layer,

#         # RNN Layer 1 
#         tf.keras.layers.SimpleRNN(128,
#                             # return_sequences=True,
#                             kernel_initializer=tf.initializers.GlorotUniform(seed=seed), 
#                             name='RNN_layer'),
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
#     name='VanillaRNN__Tensorflow')

def BiRNN__Tensorflow(input_shape, output_size, normalize_layer=None, seed=941):
    model = tf.keras.Sequential(layers=None, 
                                name='BiRNN__Tensorflow')
    model.add(tf.keras.Input(shape=input_shape, 
                             name='Input_layer'))
    # Normalization
    if normalize_layer: model.add(normalize_layer)
    # BiRNN Layer 1 
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(units=128, 
                                                                      return_sequences=True,
                                                                      kernel_initializer=tf.initializers.GlorotUniform(seed=seed)),
                                            name='BiRNN_layer_1'))
    # BiRNN Layer 2 
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(units=64, 
                                                                      return_sequences=True,
                                                                      kernel_initializer=tf.initializers.GlorotUniform(seed=seed)),
                                            name='BiRNN_layer_2'))
    # BiRNN Layer 3 
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(units=32, 
                                                                      return_sequences=False,
                                                                      kernel_initializer=tf.initializers.GlorotUniform(seed=seed)),
                                            name='BiRNN_layer_3'))
    # FC Layer
    model.add(tf.keras.layers.Dense(units=32,
                                    activation='relu',
                                    kernel_initializer=tf.initializers.GlorotUniform(seed=seed),
                                    name='Fully_Connected_layer'))

    # Output Layer
    model.add(tf.keras.layers.Dense(units=output_size, 
                                    kernel_initializer=tf.initializers.GlorotUniform(seed=seed),
                                    name='Output_layer'))
    return model

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