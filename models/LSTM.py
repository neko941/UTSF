import tensorflow as tf
from models.Base import TensorflowModel

class VanillaLSTM__Tensorflow(TensorflowModel):
    def __init__(self, input_shape, output_shape, units, normalize_layer=None, seed=941, **kwargs):
        super().__init__(input_shape, output_shape, units, normalize_layer, seed)
        
    def build(self, input_shape, output_shape, units):
        self.model = tf.keras.Sequential(layers=None, 
                                         name=self.__class__.__name__)
        self.model.add(tf.keras.Input(shape=input_shape, 
                                      name='Input_layer',
                                      # default
                                      batch_size=None,
                                      dtype=None,
                                      sparse=None,
                                      tensor=None,
                                      ragged=None,
                                      type_spec=None))
        # Normalization
        if self.normalize_layer: self.model.add(self.normalize_layer)
        # LSTM Layer 1
        self.model.add(tf.keras.layers.LSTM(units=units[0],
                                            kernel_initializer=tf.initializers.GlorotUniform(seed=self.seed), 
                                            name='LSTM_layer',
                                            # defaut
                                            activation='tanh',
                                            recurrent_activation='sigmoid',
                                            use_bias=True,
                                            recurrent_initializer='orthogonal',
                                            bias_initializer='zeros',
                                            unit_forget_bias=True,
                                            kernel_regularizer=None,
                                            recurrent_regularizer=None,
                                            bias_regularizer=None,
                                            activity_regularizer=None,
                                            kernel_constraint=None,
                                            recurrent_constraint=None,
                                            bias_constraint=None,
                                            dropout=0.0,
                                            recurrent_dropout=0.0,
                                            return_sequences=False,
                                            return_state=False,
                                            go_backwards=False,
                                            stateful=False,
                                            time_major=False,
                                            unroll=False))
        # FC Layer
        self.model.add(tf.keras.layers.Dense(units=units[1],
                                             # activation='xsinsquared',
                                             # activation='snake_a5',
                                             activation='relu',
                                             kernel_initializer=tf.initializers.GlorotUniform(seed=self.seed),
                                             name='Fully_Connected_layer',
                                             # defaut
                                             use_bias=True,
                                             bias_initializer="zeros",
                                             kernel_regularizer=None,
                                             bias_regularizer=None,
                                             activity_regularizer=None,
                                             kernel_constraint=None,
                                             bias_constraint=None))
        # Output Layer
        self.model.add(tf.keras.layers.Dense(units=output_shape, 
                                             kernel_initializer=tf.initializers.GlorotUniform(seed=self.seed),
                                             # activation='snake_a1',
                                             name='Output_layer'))

class BiLSTM__Tensorflow(TensorflowModel):
    def __init__(self, input_shape, output_shape, units, normalize_layer=None, seed=941, **kwargs):
        super().__init__(input_shape, output_shape, units, normalize_layer, seed)
        
    def build(self, input_shape, output_shape, units):
        self.model = tf.keras.Sequential(layers=None, 
                                    name=self.__class__.__name__)
        self.model.add(tf.keras.Input(shape=input_shape, 
                                 name='Input_layer'))
        # Normalization
        if self.normalize_layer: self.model.add(self.normalize_layer)
        # BiLSTM Layer 1 
        self.model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=self.units[0], 
                                                                     return_sequences=True,
                                                                     kernel_initializer=tf.initializers.GlorotUniform(seed=self.seed)),
                                                name='BiLSTM_layer_1'))
        # BiLSTM Layer 2 
        self.model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=self.units[1], 
                                                                     return_sequences=True,
                                                                     kernel_initializer=tf.initializers.GlorotUniform(seed=self.seed)),
                                                name='BiLSTM_layer_2'))
        # BiLSTM Layer 3 
        self.model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=self.units[2], 
                                                                     return_sequences=False,
                                                                     kernel_initializer=tf.initializers.GlorotUniform(seed=self.seed)),
                                                name='BiLSTM_layer_3'))
        # FC Layer
        self.model.add(tf.keras.layers.Dense(units=self.units[3],
                                             activation='relu',
                                             kernel_initializer=tf.initializers.GlorotUniform(seed=self.seed),
                                             name='Fully_Connected_layer'))

        # Output Layer
        self.model.add(tf.keras.layers.Dense(units=output_shape, 
                                        kernel_initializer=tf.initializers.GlorotUniform(seed=self.seed),
                                        name='Output_layer'))

# def ConvLSTM__Tensorflow(input_shape, output_size, normalize_layer=None, seed=941):
#     # input_shape = list(input_shape)
#     # while len(input_shape) < 5-1: input_shape.insert(0, None)
#     model = tf.keras.Sequential(layers=None, name='ConvLSTM__Tensorflow')
#     model.add(tf.keras.Input(shape=input_shape, name='input_layer'))
#     if normalize_layer: model.add(normalize_layer)
#     model.add(tf.keras.layers.ConvLSTM2D(filters=64, kernel_size=(1,2), activation='relu'))
#     model.add(tf.keras.layers.Flatten())
#     model.add(tf.keras.layers.Dense(output_size))
#     return model