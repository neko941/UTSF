from keras import backend as K
from keras.layers.core import Activation
from keras.utils.generic_utils import get_custom_objects

def SnakeActivation(x, a):
    return x - K.cos(2*a*x)/(2*a) + 1/(2*a)

def get_custom_activations():
    get_custom_objects().update({'xsinsquared': Activation(lambda x: x + (K.sin(x)) ** 2),
                                 'xsin': Activation(lambda x: x + (K.sin(x))),
                                 'snake_a.5': Activation(lambda x: SnakeActivation(x=x, a=0.5)),
                                 'snake_a1': Activation(lambda x: SnakeActivation(x=x, a=1)),
                                 'snake_a5': Activation(lambda x: SnakeActivation(x=x, a=5)),
                                 })  

