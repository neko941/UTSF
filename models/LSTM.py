import tensorflow as tf

def VanillaLSTM__Tensorflow(input_shape, output_size, normalize_layer=None, seed=941):
    return tf.keras.Sequential([
          # Input layer 
          tf.keras.Input(shape=input_shape, name='input_layer'), 

          normalize_layer,

          # LSTM Layer 1 
          tf.keras.layers.LSTM(128,
                              # return_sequences=True,
                              kernel_initializer=tf.initializers.GlorotUniform(seed=seed), 
                              name='LSTM_layer'),
          # FC Layer 1
          tf.keras.layers.Dense(32,
                                activation='relu',
                                kernel_initializer=tf.initializers.GlorotUniform(seed=seed),
                                name='Fully_Connected_layer'
                                ),
          
          # Output Layer
          tf.keras.layers.Dense(output_size, 
                                kernel_initializer=tf.initializers.GlorotUniform(seed=seed),
                                name='Output_layer') 
      ],
      name='VanillaLSTM__Tensorflow')

def BiLSTM__Tensorflow(input_shape, output_size, normalize_layer=None, seed=941):
  	return tf.keras.Sequential([
          # Input layer 
          tf.keras.Input(shape=input_shape, name='input_layer'), 

          normalize_layer,

          # BiLSTM Layer 1 
          tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, 
                                                             return_sequences=True, 
                                                             kernel_initializer=tf.initializers.GlorotUniform(seed=seed)),
                                        name='BiLSTM_layer_1'), 

          # BiLSTM Layer 2
          tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, 
                                                             return_sequences=True, 
                                                             kernel_initializer=tf.initializers.GlorotUniform(seed=seed)),
                                        name='BiLSTM_layer_2'),          

          # BiLSTM Layer 3
          tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, 
                                                             return_sequences=False, 
                                                             kernel_initializer=tf.initializers.GlorotUniform(seed=seed)),
                                        name='BiLSTM_layer_3'),   
                              
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
      name='BiLSTM__Tensorflow')

def ConvLSTM__Tensorflow(input_shape, output_size, normalize_layer=None, seed=941):
    # input_shape = list(input_shape)
    # while len(input_shape) < 5-1: input_shape.insert(0, None)
    model = tf.keras.Sequential(layers=None, name='ConvLSTM__Tensorflow')
    model.add(tf.keras.Input(shape=input_shape, name='input_layer'))
    if normalize_layer: model.add(normalize_layer)
    model.add(tf.keras.layers.ConvLSTM2D(filters=64, kernel_size=(1,2), activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(output_size))
    return model