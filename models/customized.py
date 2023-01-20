import tensorflow as tf

def RNNcLSTM__Tensorflow(input_shape, output_size, normalize_layer=None, seed=941):
    input_layer = tf.keras.Input(shape=input_shape, name='input_layer')

    if normalize_layer: x = normalize_layer(input_layer)

    for n_unit in [128, 64]:
        rnn_x = tf.keras.layers.SimpleRNN(n_unit, 
                                    return_sequences=True, 
                                    kernel_initializer=tf.initializers.GlorotUniform(seed=seed)
                                    )(x)
        
        lstm_x = tf.keras.layers.LSTM(n_unit, 
                                    return_sequences=True, 
                                    kernel_initializer=tf.initializers.GlorotUniform(seed=seed)
                                    )(x)

        x = tf.concat([rnn_x, lstm_x], axis=-1)  

    rnn_x = tf.keras.layers.SimpleRNN(n_unit, 
                                return_sequences=False, 
                                kernel_initializer=tf.initializers.GlorotUniform(seed=seed)
                                )(x)
    
    lstm_x = tf.keras.layers.LSTM(n_unit, 
                                return_sequences=False, 
                                kernel_initializer=tf.initializers.GlorotUniform(seed=seed)
                                )(x)

    x = tf.concat([rnn_x, lstm_x], axis=-1)  
                        
    x = tf.keras.layers.Dense(32,
                            activation='relu',
                            kernel_initializer=tf.initializers.GlorotUniform(seed=seed),
                            name='fc_layer_1'
                            )(x)
    
    output_layer = tf.keras.layers.Dense(output_size, 
                            kernel_initializer=tf.initializers.GlorotUniform(seed=seed),
                            name='output_layer')(x)

    model = tf.keras.Model(input_layer, output_layer, name='combined_RNN_LSTM__Tensorflow')

    return model    
