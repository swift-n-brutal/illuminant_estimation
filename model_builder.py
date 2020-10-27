import tensorflow as tf
import numpy as np
from argparse import ArgumentParser

from tfbox.config import TF_DTYPE
from tfbox.models import Model
from tfbox.layers import Act, Conv2d, Linear

class BaseNet(object):
    """Base Network
    
    Note
    ----
    The input should be in UV format:
        u,v = log(r/g), log(b/g)
    
    Attributes
    ----------
    branches : int
        The number of branches
    sub_mean : boolean
        Whether subtract mean from the input
    out_dim : int
        The dimension of output
    """
    def __init__(self, sub_mean, branches, out_dim,
                 filler=('msra', 0., 1.), act_config=('ReLU',), name='base_net'):
        self.name = name
        self._sub_mean = sub_mean
        self._branches = branches
        self._out_dim = out_dim
        self._filler = filler
        self._act_config = act_config
        
    def build_model(self, inputs, name, phase=None, model=None, update_collection=None):
        if model is None:
            model = Model()
            model.inputs.extend(inputs)
        #
        with tf.name_scope(name):
            with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
                if self._sub_mean:
                    # sub mean
                    mean_uv = tf.reduce_mean(inputs[-1], axis=[1,2], keepdims=True)
                    tops = [inputs[-1] - mean_uv]
                else:
                    tops = inputs
                #
                tops = self.append_base(model, tops)
                tops = self.append_head(model, tops)
                if self._sub_mean:
                    # add mean
                    mean_uv = tf.squeeze(mean_uv)
                    tops = [t + mean_uv for t in tops]
        model.outputs.extend(tops)
        assert len(model.outputs) == self._branches
        return model
        
    def append_base(self, model, inputs, name='base'):
        filler = self._filler
        config = self._act_config
        with tf.variable_scope(name):
            # conv 8x8/4
            tops = model.append(
                Conv2d(inputs[-1], [8, 8, 128], bias=True, pad_size=0, stride=4,
                       name='conv1', filler=filler))
            tops = model.append(Act(tops[-1], config, name='act1'))
            # conv 4x4/2
            tops = model.append(
                Conv2d(tops[-1], [4, 4, 256], bias=True, pad_size=0, stride=2,
                       name='conv2', filler=filler))
            # move the activation to the head part
            #tops = model.append(Act(tops[-1], config, name='act2'))
        return tops
    
    def append_head(self, model, inputs, name='head'):
        filler = self._filler
        config = self._act_config
        with tf.variable_scope(name):
            bottoms = model.append(Act(inputs[-1], config, name='act'))
            outputs = list()
            for b in range(self._branches):
                with tf.variable_scope('b%d' % b):
                    tops = model.append(Linear(bottoms[-1], 256,
                                               name='fc1', filler=filler))
                    tops = model.append(Act(tops[-1], config, name='act_fc'))
                    # output (uv - mean_uv) or logits
                    tops = model.append(Linear(tops[-1], self._out_dim,
                                               name='fc2', filler=filler))
                outputs.extend(tops)
        return outputs
                    
class HypNet(BaseNet):
    """Hypothesis Network"""
    def __init__(self, branches=2, name='hypnet'):
        BaseNet.__init__(self, True, branches, 2, name=name)
        
    @staticmethod
    def get_parser(ps=None):
        if ps is None: ps = ArgumentParser()
        g = ps.add_argument_group('hypnet')
        g.add_argument('--hn-branches', type=int, default=2)
        return ps
        
class SelNet(BaseNet):
    """Selection Network"""
    def __init__(self, branches=2, name='selnet'):
        BaseNet.__init__(self, False, 1, branches, name=name)
