import tensorflow as tf
import numpy as np
import math
# https://github.com/edebrouwer/gru_ode_bayes/blob/master/gru_ode_bayes/models.py
# GRU-ODE: Neural Negative Feedback ODE with Bayesian jumps

class GRUODECell(tf.keras.layers.Layer):
    def __init__(self, input_size, hidden_size, bias=True):
        """
        For p(t) modelling input_size should be 2x the x size.
        """
        super(GRUODECell, self).__init__()
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.bias        = bias

        self.lin_xz = tf.keras.layers.Dense(hidden_size, bias=bias)
        self.lin_xn = tf.keras.layers.Dense(hidden_size, bias=bias)

        self.lin_hz = tf.keras.layers.Dense(hidden_size, use_bias=False)
        self.lin_hn = tf.keras.layers.Dense(hidden_size, use_bias=False)


    def call(self, x, h):
        """
        Returns a change due to one step of using GRU-ODE for all h.
        The step size is given by delta_t.
        Args:
            x        input values
            h        hidden state (current)
            delta_t  time step
        Returns:
            Updated h
        """
        z = tf.sigmoid(self.lin_xz(x) + self.lin_hz(h))
        n = tf.tanh(self.lin_xn(x) + self.lin_hn(z * h))

        dh = (1 - z) * (n - h)
        return dh
    
class GRUODECell_Autonomous(tf.keras.layers.Layer):
    def __init__(self, hidden_size, bias=True):
        """
        For p(t) modelling input_size should be 2x the x size.
        """
        super(GRUODECell_Autonomous, self).__init__()
        self.hidden_size = hidden_size
        self.bias        = bias

        #self.lin_xz = tf.keras.layers.Dense(hidden_size, bias=bias)
        #self.lin_xn = tf.keras.layers.Dense(hidden_size, bias=bias)

        self.lin_hz = tf.keras.layers.Dense(hidden_size, use_bias=False)
        self.lin_hn = tf.keras.layers.Dense(hidden_size, use_bias=False)


    def call(self, t, h):
        """
        Returns a change due to one step of using GRU-ODE for all h.
        The step size is given by delta_t.
        Args:
            t        time
            h        hidden state (current)
        Returns:
            Updated h
        """
        x = tf.zeros_like(h)
        z = tf.sigmoid(x + self.lin_hz(h))
        n = tf.tanh(x + self.lin_hn(z * h))

        dh = (1 - z) * (n - h)
        return dh

class FullGRUODECell(tf.keras.layers.Layer):
    def __init__(self, input_size, hidden_size, bias=True):
        super(FullGRUODECell, self).__init__()

        self.lin_x = tf.keras.layers.Dense(hidden_size * 3, bias_initializer='zeros')

        self.lin_hh = tf.keras.layers.Dense(hidden_size, use_bias=False)
        self.lin_hz = tf.keras.layers.Dense(hidden_size, use_bias=False)
        self.lin_hr = tf.keras.layers.Dense(hidden_size, use_bias=False)

    def call(self, x, h):
        """
        Executes one step with GRU-ODE for all h.
        The step size is given by delta_t.
        Args:
            x        input values
            h        hidden state (current)
            delta_t  time step
        Returns:
            Updated h
        """

        xr, xz, xh = tf.split(self.lin_x(x), 3, axis=1)
        r = tf.sigmoid(xr + self.lin_hr(h))
        z = tf.sigmoid(xz + self.lin_hz(h))
        u = tf.tanh(xh + self.lin_hh(r * h))

        dh = (1 - z) * (u - h)
        return dh
    
import tensorflow as tf

class FullGRUODECell_Autonomous(tf.keras.layers.Layer):
    
    def __init__(self, hidden_size, bias=True):
        """
        For p(t) modelling input_size should be 2x the x size.
        """
        super(FullGRUODECell_Autonomous, self).__init__()

        #self.lin_xh = tf.keras.layers.Dense(hidden_size, bias=bias)
        #self.lin_xz = tf.keras.layers.Dense(hidden_size, bias=bias)
        #self.lin_xr = tf.keras.layers.Dense(hidden_size, bias=bias)

        #self.lin_x = tf.keras.layers.Dense(hidden_size * 3, bias=bias)

        self.lin_hh = tf.keras.layers.Dense(hidden_size, use_bias=False)
        self.lin_hz = tf.keras.layers.Dense(hidden_size, use_bias=False)
        self.lin_hr = tf.keras.layers.Dense(hidden_size, use_bias=False)

    def call(self, t, h):
        #xr, xz, xh = tf.split(self.lin_x(x), num_or_size_splits=3, axis=1)
        x = tf.zeros_like(h)
        r = tf.sigmoid(x + self.lin_hr(h))
        z = tf.sigmoid(x + self.lin_hz(h))
        u = tf.tanh(x + self.lin_hh(r * h))

        dh = (1 - z) * (u - h)
        return dh

class GRUObservationCellLogvar(tf.keras.layers.Layer):
    """Implements discrete update based on the received observations."""

    def __init__(self, input_size, hidden_size, prep_hidden, bias=True):
        super(GRUObservationCellLogvar, self).__init__()
        self.gru_d = tf.keras.layers.GRUCell(units=hidden_size)
        self.gru_debug = tf.keras.layers.GRUCell(units=hidden_size)

        ## prep layer and its initialization
        std = math.sqrt(2.0 / (4 + prep_hidden))
        self.w_prep = tf.Variable(initial_value=std * np.random.randn(input_size, 4, prep_hidden), dtype=tf.float32)
        self.bias_prep = tf.Variable(initial_value=0.1 + np.zeros((input_size, prep_hidden)), dtype=tf.float32)

        self.input_size = input_size
        self.prep_hidden = prep_hidden

    def call(self, h, p, X_obs, M_obs, i_obs):
        ## only updating rows that have observations
        p_obs = tf.gather(p, i_obs, axis=0)

        mean, logvar = tf.split(p_obs, num_or_size_splits=2, axis=1)
        sigma = tf.exp(0.5 * logvar)
        error = (X_obs - mean) / sigma

        ## log normal loss, over all observations
        log_lik_c = np.log(np.sqrt(2 * np.pi))
        losses = 0.5 * ((tf.pow(error, 2) + logvar + 2 * log_lik_c) * M_obs)
        if tf.reduce_sum(losses) != tf.reduce_sum(losses):
            import pdb; pdb.set_trace()

        ## TODO: try removing X_obs (they are included in error)
        gru_input = tf.stack([X_obs, mean, logvar, error], axis=2)
        gru_input = tf.matmul(gru_input, self.w_prep) + self.bias_prep
        gru_input = tf.nn.relu(gru_input)
        ## gru_input is (sample x feature x prep_hidden)
        gru_input = tf.transpose(gru_input, perm=[2, 0, 1])
        gru_input = tf.reshape(gru_input * M_obs, shape=(-1, self.prep_hidden * self.input_size), name="gru_input")

        temp = tf.identity(h)
        temp = tf.tensor_scatter_nd_update(temp, tf.expand_dims(i_obs, axis=1), self.gru_d(gru_input, tf.gather(h, i_obs, axis=0)))
        h = temp

        return h, losses
    
class GRUObservationCell(tf.keras.layers.Layer):
    def __init__(self, input_size, hidden_size, prep_hidden, bias=True, **kwargs):
        super().__init__(**kwargs)
        self.gru_d = tf.keras.layers.GRUCell(hidden_size, bias_initializer='ones', kernel_initializer='glorot_uniform')
        self.gru_debug = tf.keras.layers.GRUCell(hidden_size, bias_initializer='ones', kernel_initializer='glorot_uniform')

        # prep layer and its initialization
        std = math.sqrt(2.0 / (4 + prep_hidden))
        self.w_prep = tf.Variable(std * tf.random.normal(shape=(input_size, 4, prep_hidden)))
        self.bias_prep = tf.Variable(0.1 + tf.zeros(shape=(input_size, prep_hidden)))

        self.input_size = input_size
        self.prep_hidden = prep_hidden
        self.var_eps = 1e-6

    def call(self, h, p, X_obs, M_obs, i_obs):

        # only updating rows that have observations
        p_obs = tf.gather(p, i_obs, axis=0)
        mean, var = tf.split(p_obs, 2, axis=1)
        # making var non-negative and also non-zero (by adding a small value)
        var = tf.abs(var) + self.var_eps
        error = (X_obs - mean) / tf.sqrt(var)

        # log normal loss, over all observations
        loss = 0.5 * tf.reduce_sum((tf.pow(error, 2) + tf.math.log(var)) * M_obs)

        # TODO: try removing X_obs (they are included in error)
        gru_input = tf.stack([X_obs, mean, var, error], axis=2)
        gru_input = tf.matmul(gru_input, self.w_prep) + self.bias_prep
        gru_input = tf.nn.relu(gru_input)
        # gru_input is (sample x feature x prep_hidden)
        gru_input = tf.transpose(gru_input, perm=[2, 0, 1])
        gru_input = tf.reshape(gru_input * M_obs, shape=(-1, self.prep_hidden * self.input_size))
        
        temp = tf.identity(h)
        temp = tf.tensor_scatter_nd_update(temp, tf.expand_dims(i_obs, axis=1), self.gru_d(gru_input, tf.gather(h, i_obs, axis=0)))
        h = temp

        return h, loss

