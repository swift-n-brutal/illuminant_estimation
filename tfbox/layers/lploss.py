import tensorflow as tf
import numpy as np
from ..config import NP_DTYPE, TF_DTYPE, LAYERS_VERBOSE
from .layer import Layer

class LpLoss(Layer):
    def __init__(self, x, y, p, samplewise=False, name='lp_loss'):
        super(LpLoss, self).__init__(name)
        self.inputs.extend([x, y])
        with tf.variable_scope(name):
            if p != 1 and p != 2:
                raise ValueError('Unsupported LpLoss: p = %d' % p)
            if p == 1:
                eltwise = tf.abs(x-y)
            elif p == 2:
                eltwise = tf.square(x-y)
            if samplewise:
                rank = len(eltwise.shape.as_list())
                if rank > 1:
                    self.outputs.append(tf.reduce_mean(eltwise, axis=range(1,rank) ))
                else:
                    self.outputs.append(eltwise)
            else:
                self.outputs.append(tf.reduce_mean(eltwise))
        self.print_info(LAYERS_VERBOSE)

