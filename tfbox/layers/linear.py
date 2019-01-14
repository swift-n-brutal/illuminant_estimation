import tensorflow as tf
import numpy as np
from ..config import NP_DTYPE, TF_DTYPE, LAYERS_VERBOSE
from .layer import Layer

class Linear(Layer):
    def __init__(self, x, output_dim, bias=True, name='fc', filler=('msra', 0., 1.)):
        super(Linear, self).__init__(name)
        self.inputs.append(x)
        with tf.variable_scope(name):
            shape = x.get_shape().as_list()
            fan_in = np.prod(shape[1:])
            x = tf.reshape(x, [-1, fan_in])
            
            initializer = None
            if filler[0] == 'uniform':
                initializer = tf.random_uniform_initializer(filler[1], filler[2])
            elif filler[0] == 'msra':
                stdev = np.sqrt(2. / ((filler[1]**2 + filler[2]**2) * fan_in))
                initializer = tf.random_normal_initializer(0., stdev)
            elif filler[0] == 'gaussian':
                initializer = tf.random_normal_initializer(filler[1], filler[2])
            else:
                raise ValueError('Invalid filler type: %s' % (filler[0]))

            weight = tf.get_variable('weight', shape=[fan_in, output_dim], dtype=TF_DTYPE, initializer=initializer)
            self.params.append(weight)
            x = tf.matmul(x, weight)

            if bias:
                b = tf.get_variable('bias', shape=[output_dim], dtype=TF_DTYPE, initializer=tf.constant_initializer(0.))
                self.params.append(b)
                x = tf.nn.bias_add(x, b)
        self.outputs.append(x)
        self.print_info(LAYERS_VERBOSE)

