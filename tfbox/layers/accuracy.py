import tensorflow as tf
import numpy as np
from ..config import NP_DTYPE, TF_DTYPE, LAYERS_VERBOSE
from .layer import Layer

class Accuracy(Layer):
    def __init__(self, logit, label, topk=1, name='accuracy'):
        super(Accuracy, self).__init__(name)
        self.inputs.extend([logit, label])
        with tf.variable_scope(name):
            assert topk == 1, 'Invalid parameter (topk = %s)' % str(topk)
            correct_prediction = tf.equal(
                    tf.cast(tf.argmax(logit, 1), tf.int32), tf.cast(label, tf.int32))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        self.outputs.append(accuracy)
        self.print_info(LAYERS_VERBOSE)
