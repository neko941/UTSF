import tensorflow as tf

def EncoderDecoder__Tensorflow(input_shape, output_size, normalize_layer=None, seed=941):
    model = tf.keras.Sequential(layers=None, name='EncoderDecoder_Model')
    model.add(tf.keras.Input(shape=input_shape, name='input_layer'))
    if normalize_layer: model.add(normalize_layer)
    model.add(tf.keras.layers.LSTM(256, activation='relu'))
    model.add(tf.keras.layers.RepeatVector(output_size))
    model.add(tf.keras.layers.LSTM(128, activation='relu', return_sequences=True))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(output_size)))
    return model

def BiEncoderDecoder__Tensorflow(input_shape, output_size, normalize_layer=None, seed=941):
    return tf.keras.models.Sequential(
            [
                tf.keras.Input(shape=input_shape, name='input_layer'),
                normalize_layer,
                tf.keras.layers.LSTM(256, return_sequences=False),
                tf.keras.layers.RepeatVector(output_size),
                tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(256, return_sequences=True)
                ),
                tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(128, return_sequences=True)
                ),
                tf.keras.layers.TimeDistributed(
                    tf.keras.layers.Dense(output_size)
                ),
                tf.keras.layers.Dense(output_size, 
                              kernel_initializer=tf.initializers.GlorotUniform(seed=seed),
                              name='output_layer') 
            ],
    name='BiEncoderDecoder__Tensorflow'
        )