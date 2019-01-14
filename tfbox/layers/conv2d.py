import tensorflow as tf
import numpy as np
from ..config import NP_DTYPE, TF_DTYPE, LAYERS_VERBOSE
from .layer import Layer

class Conv2d(Layer):
    """2D Convolutional Layer

    """
    def __init__(self, x, filter_shape, bias=True, stride=1, pad_size=0, pad_mode='CONSTANT',
            name='conv2d', filler=('msra', 0., 1.)):
        """__init__ method of Conv2d

        Parameters
        ----------
        x : tf.Tensor
            Input of 4D tensor.
        filter_shape : list of int
            Shape of convolutional filter. [filter_height, filter_width, chn_in]
            or [filter_height, filter_width, chn_in, chn_out]
        bias : bool
            Whether to add bias.
        stride : int
            Stride size.
        pad_size : int
        pad_mode : str
            One of 'CONSTANT', 'REFLECT', or 'SYMMETRIC'
        name : string
        filler : tuple
            Initializer for convolutional weight. One of
            -   ('msra', negative_slope, positive_slope)
            -   ('gaussian', mean, stdev)
            -   ('uniform', minval, maxval)
        """
        super(Conv2d, self).__init__(name)
        self.inputs.append(x)
        in_shape = x.shape.as_list()
        if len(filter_shape) == 3:
            # get chn_in from input tensor
            kin = in_shape[-1]
            kout = filter_shape[-1]
            filter_shape[-1] = kin
            filter_shape.append(kout)
        kh, kw, kin, kout = filter_shape
        with tf.variable_scope(name):
            # Use customized padding
            padding = 'VALID'
            if pad_size == -1:
                # 'SAME' padding
                if pad_mode == 'CONSTANT':
                    padding = 'SAME'
                else:
                    w_in = in_shape[-2]
                    if w_in % stride == 0:
                        pad_size_both = max(kw - stride, 0)
                    else:
                        pad_size_both = max(kw - (w_in % stride), 0)
                    if pad_size_both > 0:
                        pad_size = pad_size_both / 2
                        x = tf.pad(x, [[0,0], [pad_size, pad_size_both-pad_size],
                            [pad_size, pad_size_both-pad_size], [0,0]], pad_mode)
            elif pad_size > 0:
                # pad_size padding on both sides of each dimension
                x = tf.pad(x, [[0,0], [pad_size, pad_size], [pad_size, pad_size], [0,0]], pad_mode)
            # initializer for weight
            initializer = None
            if filler[0] == 'uniform':
                initializer = tf.random_uniform_initializer(filler[1], filler[2])
            elif filler[0] == 'msra':
                fan_in = kh * kw * kin
                stdev = np.sqrt(2. / ((filler[1]**2 + filler[2]**2) * fan_in))
                initializer = tf.random_normal_initializer(0., stdev)
            elif filler[0] == 'gaussian':
                initializer = tf.random_normal_initializer(filler[1], filler[2])
            else:
                raise ValueError('Invalid filler type: %s' % (filler[0]))
            # convolution
            weight = tf.get_variable('weight', shape=filter_shape, dtype=TF_DTYPE, initializer=initializer)
            self.params.append(weight)
            x = tf.nn.conv2d(x, weight, [1, stride, stride, 1], padding=padding)
            # add channel-wise bias
            if bias:
                b = tf.get_variable('bias', shape=kout, dtype=TF_DTYPE, initializer=tf.constant_initializer(0.))
                self.params.append(b)
                x = tf.nn.bias_add(x, b)
        self.outputs.append(x)
        self.print_info(LAYERS_VERBOSE)
