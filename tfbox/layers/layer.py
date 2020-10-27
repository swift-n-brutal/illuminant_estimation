import tensorflow as tf
import numpy as np
from ..config import NP_DTYPE, TF_DTYPE

class Layer(object):
    """Base class of layer.

    Attributes:
        name (str): name of layer
        inputs (list of tf.tensor): input tensors
        outputs (list of tf.tensor): output tensors
        params (list of tf.tensor): trainable parameters used to compute outputs
    """
    def __init__(self, name='base'):
        self.name = tf.get_variable_scope().name + '/' + name
        self.inputs = list()
        self.outputs = list()
        self.params = list()
        
    def print_info(self, verbose):
        if verbose:
            print(self.name)
            print('\tIn', self.inputs)
            print('\tOut', self.outputs)
