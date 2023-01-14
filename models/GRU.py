import tensorflow as tf

def BiGRU__Tensorflow(input_shape, output_size, normalize_layer=None, seed=941):
	return tf.keras.Sequential([
        # Input layer 
        tf.keras.Input(shape=input_shape, name='input_layer'), 

        normalize_layer,

        # BiLSTM Layer 1 
        tf.keras.layers.Bidirectional(tf.keras.layers.GRU(128,
                                                          return_sequences=True,
                                                          kernel_initializer=tf.initializers.GlorotUniform(seed=seed)),
                                      name='BiGRU_layer_1'), 

        # BiLSTM Layer 2
        tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64, 
                                                           return_sequences=True, 
                                                           kernel_initializer=tf.initializers.GlorotUniform(seed=seed)),
                                      name='BiGRU_layer_2'),          

        # BiLSTM Layer 3
        tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32, 
                                                           return_sequences=False, 
                                                           kernel_initializer=tf.initializers.GlorotUniform(seed=seed)),
                                      name='BiGRU_layer_3'),   
                            
        # FC Layer 1
        tf.keras.layers.Dense(32,
                              activation='relu',
                              kernel_initializer=tf.initializers.GlorotUniform(seed=seed),
                              name='fc_layer_1'
                              ),
        
        # Output Layer
        tf.keras.layers.Dense(output_size, 
                              kernel_initializer=tf.initializers.GlorotUniform(seed=seed),
                              name='output_layer') 
    ],
    name='BiGRU__Tensorflow')