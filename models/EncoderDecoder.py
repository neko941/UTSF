import tensorflow as tf

def EncoderDecoder__Tensorflow(input_shape, output_size, normalize_layer=None, seed=941):
    model = tf.keras.Sequential(layers=None, name='EncoderDecoder__Tensorflow')
    model.add(tf.keras.Input(shape=input_shape, name='input_layer'))
    if normalize_layer: model.add(normalize_layer)
    model.add(tf.keras.layers.LSTM(256, activation='relu'))
    model.add(tf.keras.layers.RepeatVector(output_size))
    model.add(tf.keras.layers.LSTM(128, activation='relu', return_sequences=True))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(output_size)))
    return model

def BiEncoderDecoder__Tensorflow(input_shape, output_size, normalize_layer=None, seed=941):
    model = tf.keras.Sequential(layers=None, name='BiEncoderDecoder__Tensorflow')
    model.add(tf.keras.Input(shape=input_shape, name='input_layer'))
    if normalize_layer: model.add(normalize_layer)
    model.add(tf.keras.layers.LSTM(256, return_sequences=False))
    model.add(tf.keras.layers.RepeatVector(output_size))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True)))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True)))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(output_size)))
    return model

def CNNcLSTMcEncoderDecoder__Tensorflow(input_shape, output_size, normalize_layer=None, seed=941):
    """
        https://github.com/davide-burba/forecasting_models/blob/master/python_models.ipynb
    """
    model = tf.keras.Sequential(layers=None, name='CNNcLSTMcEncoderDecoder__Tensorflow')
    model.add(tf.keras.Input(shape=input_shape, name='input_layer'))
    if normalize_layer: model.add(normalize_layer)
    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.RepeatVector(output_size))
    model.add(tf.keras.layers.LSTM(128, activation='relu', return_sequences=True))
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(64, activation='relu')))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(output_size)))
    return model