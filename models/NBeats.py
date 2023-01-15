import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn.preprocessing import StandardScaler

class NBeats():
    def __init__(self, input_shape, output_size, normalize_layer=None, seed=941, **kwargs):
        self.num_layers = kwargs.get('num_layers',4)
        self.num_neurons = kwargs.get('num_neurons',128)
        self.stacks = kwargs.get('stacks', 'generic, trend, seasonality, generic')
        if type(self.stacks)==str:
            self.stacks=self.stacks.split(',')
        self.activation = kwargs.get('activation','relu')
        self.input_shape = input_shape
        self.output_size = output_size


        """
        Build NBEATS model arhitecture
        """
        shape_t, shape_f = self.input_shape, self.output_size
        print(shape_t, shape_f)
        # inputs = tf.keras.layers.Input(shape=(shape_t, shape_f))
        inputs = tf.keras.layers.Input(shape=shape_t)

        # print(shape_t)
        # print(shape_f)
        # print(inputs)
        
        initial_block = self.NBeatsBlock(input_size=shape_t, theta_size=shape_f, horizon=1, n_neurons=self.num_neurons, n_layers=self.num_layers, stack_type=self.stacks[0])
        residuals, forecast = initial_block(inputs)
        for i in range(1, len(self.stacks)):
            backcast, block_forecast = self.NBeatsBlock(input_size=shape_t, theta_size=shape_f, horizon=1, n_neurons=self.num_neurons, n_layers=self.num_layers, stack_type=self.stacks[i])(residuals) 
            residuals = tf.keras.layers.subtract([residuals, backcast], name=f"subtract_{i}")
            forecast = tf.keras.layers.add([forecast, block_forecast], name=f"add_{i}")

        model = tf.keras.Model(inputs=inputs, outputs=forecast[0], name='NBeats')

        return model

    class NBeatsBlock(tf.keras.layers.Layer):
        def __init__(self, input_size: int, theta_size: int, horizon: int, n_neurons: int, n_layers: int, stack_type: str, **kwargs):
            super().__init__(**kwargs)
            self.input_size = input_size
            self.theta_size = theta_size
            self.horizon = horizon
            self.n_neurons = n_neurons
            self.n_layers = n_layers
            self.stack_type = stack_type 

            # by default block contains stack of 4 fully connected layers each has ReLU activation
            self.hidden = [tf.keras.layers.Dense(n_neurons, activation="relu") for _ in range(n_layers)]
            # Output of block is a theta layer with linear activation
            self.theta_layer = tf.keras.layers.Dense(theta_size, activation="linear", name="theta")

        def linear_space(self, backcast_length, forecast_length, is_forecast=True):
            ls = K.arange(-float(backcast_length), float(forecast_length), 1) / forecast_length
            return ls[backcast_length:] if is_forecast else K.abs(K.reverse(ls[:backcast_length], axes=0))

        def seasonality_model(self, thetas, backcast_length, forecast_length, is_forecast):
            p = thetas.get_shape().as_list()[-1]
            p1, p2 = (p // 2, p // 2) if p % 2 == 0 else (p // 2, p // 2 + 1)
            t = self.linear_space(backcast_length, forecast_length, is_forecast=is_forecast)
            s1 = K.stack([K.cos(2 * np.pi * i * t) for i in range(p1)])
            s2 = K.stack([K.sin(2 * np.pi * i * t) for i in range(p2)])
            if p == 1:
                s = s2
            else:
                s = K.concatenate([s1, s2], axis=0)
            s = K.cast(s, np.float32)
            return K.dot(thetas, s)

        def trend_model(self, thetas, backcast_length, forecast_length, is_forecast):
            p = thetas.shape[-1]           # take time dimension
            t = self.linear_space(backcast_length, forecast_length, is_forecast=is_forecast)
            t = K.transpose(K.stack([t for i in range(p)]))
            t = K.cast(t, np.float32)
            return K.dot(thetas, K.transpose(t)) 
            
        def call(self, inputs): 
            x = inputs 
            for layer in self.hidden:
                x = layer(x)
            theta = self.theta_layer(x)
                
            if self.stack_type == 'generic':
                backcast, forecast = theta[:, :self.input_size], theta[:, -self.horizon:]
            elif self.stack_type == 'seasonal':
                backcast = tf.keras.layers.Lambda(self.seasonality_model, arguments={'is_forecast': False, 'backcast_length': self.theta_size, 'forecast_length': self.horizon}, name='seasonal')(theta[:, :self.input_size])
                forecast = tf.keras.layers.Lambda(self.seasonality_model, arguments={'is_forecast': True, 'backcast_length': self.theta_size, 'forecast_length': self.horizon}, name='seasonal')(theta[:, -self.horizon:])
            else:
                backcast = tf.keras.layers.Lambda(self.trend_model, arguments={'is_forecast': False, 'backcast_length': self.theta_size, 'forecast_length': self.horizon}, name='trend')(theta[:, :self.input_size])
                forecast = tf.keras.layers.Lambda(self.trend_model, arguments={'is_forecast': True, 'backcast_length': self.theta_size, 'forecast_length': self.horizon}, name='trend')(theta[:, -self.horizon:])
            return backcast, forecast