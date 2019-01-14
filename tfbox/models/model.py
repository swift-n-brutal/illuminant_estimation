import tensorflow as tf
import numpy as np
from collections import defaultdict, OrderedDict

class Model(object):
    def __init__(self):
        self.inputs = list()
        self.outputs = list()
        self.stats = list()
        self.layers = list()
        self.params = list()
        self.param_lrs = list()
        self.param_by_lr_ = defaultdict(list) # pairs of lr and list of parameters
        self.update_params_ = defaultdict(list) # pairs of colloction and list of update params
        self._update_ops_ = defaultdict(list) # pairs of collection and list of update ops
        self.weight_decay_op = None

    def _add_params(self, params, lr):
        if len(params) > 0:
            self.params.extend(params)
            if type(lr) is float:
                # use the same learning rate for all params
                self.param_lrs.extend([lr]*len(params))
                self.param_by_lr_[lr].extend(params)
            elif type(lr) is list:
                # use individual learning rate for each param
                assert len(lr) == len(params), 'Number of lr and params must be equal (%d != %d)' % (len(lr), len(params))
                self.param_lrs.extend(lr)
                for (l, p) in zip(lr, params):
                    self.param_by_lr_[l].append(p)
            else:
                raise ValueError('The type of lr must be float or list (%s given)' % type(lr))

    def append(self, layer, lr=1.):
        self.layers.append(layer)
        self._add_params(layer.params, lr)
        l_update_params = getattr(layer, 'update_params', None)
        if l_update_params:
            self.update_params_[layer.update_collection].extend(l_update_params)
        return layer.outputs

    def extend(self, model):
        self.layers.extend(model.layers)
        self.params.extend(model.params)
        self.param_lrs.extend(model.param_lrs)
        for k in model.param_by_lr_.keys():
            self.param_by_lr_[k].extend(model.param_by_lr_[k])
        for k in model.update_params_.keys():
            self.update_params_[k].extend(model.update_params_[k])
        self._update_ops_.clear()

    def get_weight_decay(self, regularizer_type='L2'):
        if self.weight_decay_op is None:
            wd_list = list()
            fn = tf.square
            if regularizer_type == 'L1':
                fn = tf.abs
            with tf.name_scope('weight_decay'):
                for param in self.params:
                    param_name = param.name.rsplit(':', 1)[0]
                    wd_list.append(tf.reduce_sum(fn(param), name=param_name))
                self.weight_decay_op = tf.add_n(wd_list, name='loss')
        return self.weight_decay_op

    def get_update_ops(self, update_collection=None):
        if self._update_ops_.get(update_collection) is None:
            for layer in self.layers:
                update_ops_getter = getattr(layer, 'update_ops_getter', None)
                if update_ops_getter is not None and layer.update_collection == update_collection:
                    self._update_ops_[update_collection].extend(update_ops_getter())
        return self._update_ops_[update_collection]

    def get_update_params(self, update_collection=None):
        return self.update_params_[update_collection]

    def print_model(self):
        for layer in self.layers:
            layer.print_info(True)

    def save_model(self, sess, path):
        # params
        param_dict = OrderedDict()
        update_param_dict = OrderedDict()
        for layer in self.layers:
            name = layer.name
            if layer.params: # is not empty
                param_dict[name] = sess.run(layer.params) # numpy array
            l_update_params = getattr(layer, 'update_params', None)
            if l_update_params: # exists and is not empty
                update_param_dict[name] = sess.run(l_update_params) # numpy array 
        # save using numpy.savez
        np.savez(path, params=param_dict, update_params=update_param_dict)

    def load_model(self, sess, path):
        if path[-4:] != '.npz':
            path += '.npz'
        with np.load(path) as fd:
            param_dict = fd['params'][()]
            update_param_dict = fd['update_params'][()]
            for layer in self.layers:
                name = layer.name
                params = param_dict.get(name)
                if params:
                    assert len(params) == len(layer.params)
                    for (p_src, p_tgt) in zip(params, layer.params):
                        assert list(p_src.shape) == p_tgt.shape.as_list(), \
                                '{} {} {}'.format(name, p_src.shape, p_tgt)
                        sess.run(p_tgt.assign(p_src))
                update_params = update_param_dict.get(name)
                if update_params:
                    assert len(update_params) == len(layer.update_params)
                    for (up_src, up_tgt) in zip(update_params, layer.update_params):
                        assert list(up_src.shape) == up_tgt.shape.as_list(), \
                                '{} {} {}'.format(name, up_src.shape, up_tgt)
                        sess.run(up_tgt.assign(up_src))
