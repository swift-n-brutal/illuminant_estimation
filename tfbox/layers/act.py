import tensorflow as tf
import numpy as np
from ..config import NP_DTYPE, TF_DTYPE, LAYERS_VERBOSE
from .layer import Layer

class Act(Layer):
    def __init__(self, x, args, name=None):
        kwargs = dict()
        #
        if name is not None:
            kwargs['name'] = name
        #
        act_type = args[0]
        if args[0] == 'ReLU':
            if len(args) == 2:
                kwargs['slope'] = args[1]
            self.ReLU(x, **kwargs)
        elif args[0] == 'ELU':
            if len(args) == 2:
                kwargs['alpha'] = args[1]
            self.ELU(x, **kwargs)
        elif args[0] == 'SoftPlus':
            if len(args) == 2:
                kwargs['scale'] = args[1]
            self.SoftPlus(x, **kwargs)
        else:
            raise ValueError('Invalid activation type: %s' % str(args[0]))
        self.print_info(LAYERS_VERBOSE)

    def ELU(self, x, alpha=1., name='elu'):
        super(Act, self).__init__(name)
        self.inputs.append(x)
        with tf.variable_scope(name):
            elu1 = tf.nn.elu(x)
            if alpha == 1.:
                self.outputs.append(elu1)
            else:
                self.outputs.append(tf.where(x >= 0, elu1, alpha * elu1))

    def ReLU(self, x, slope=0., name='relu'):
        super(Act, self).__init__(name)
        self.inputs.append(x)
        with tf.variable_scope(name):
            if slope == 0.:
                self.outputs.append(tf.nn.relu(x))
            else:
                self.outputs.append(tf.where(x >= 0, x, slope*x))

    def SoftPlus(self, x, scale=None, name='softplus'):
        """
            y = ln(exp(scale * x) + 1) / scale
        Parameters
        ----------
        scale : float
            The larger the scale is, the closer the shape of SoftPlus is to 
            that of ReLU.
        """
        super(Act, self).__init__(name)
        self.inputs.append(x)
        with tf.variable_scope(name):
            if scale is None or scale == 1:
                self.outputs.append(tf.nn.softplus(x))
            else:
                scale = NP_DTYPE
                assert scale != 0, "Scale for softplus should be positive. (%f given)" % scale
                reci_scale = NP_DTYPE(1. / scale)
                self.outputs.append(reci_scale * tf.nn.softplus(scale * x))
