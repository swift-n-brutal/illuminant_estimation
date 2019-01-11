import numpy as np
from argparse import ArgumentParser

__all__ = ['Dataset', 'SqDataset']

def _stack(data, batch_shape):
    if isinstance(data[0], dict):
        data_stacked = dict()
        for k in data[0].keys():
            data_stacked[k] = np.stack([d[k] for d in data], axis=0).reshape(
                batch_shape + data[0][k].shape)
    elif isinstance(data[0], tuple):
        data_stacked = tuple(np.stack(d, axis=0).reshape(batch_shape+d[0].shape) \
                             for d in zip(*data))
    elif isinstance(data[0], list):
        data_stacked = list(np.stack(d, axis=0).reshape(batch_shape+d[0].shape) \
                            for d in zip(*data))
    elif isinstance(data[0], np.ndarray):
        data_stacked = np.stack(data, axis=0).reshape(batch_shape+data[0].shape)
    elif np.isscalar(data[0]):
        data_stacked = np.stack(data, axis=0).reshape(batch_shape)
    else:
        raise TypeError('Unsupported type: {}'.format(type(data[0])))
    return data_stacked

class Dataset(object):
    """Dataset returns indexed data.
    
    Any derived class should implement its own methods of:
        * _init()
        * _get(idx)
    """
    DEFAULT_ARGS = {
        'size': 100,
        'dtype': np.float32}
    def __init__(self, args=DEFAULT_ARGS, name="dataset"):
        self.name = name
        self._size = None
        self._args = args
        self._initialized = False
        
    def _init(self):
        self._size = self._args['size']
        self._data_shape = (1,)
        self._dtype = self._args['dtype']
        self._data = np.arange(self._size, dtype=self._dtype).reshape(-1, *(self._data_shape))
        
    def _get(self, idx):
        return self._data[idx]
        
    def _check_init(self):
        if not self._initialized:
            self._init()
            self._initialized = True
    
    def reset(self):
        pass
    
    def size(self):
        self._check_init()
        assert self._size is not None, \
                'The size of dataset must be provided in _init()'
        return self._size
    
    def get(self, idx):
        self._check_init()
        if not isinstance(idx, np.ndarray):
            idx = np.array(idx)
        batch_shape = idx.shape
        batch_size = np.prod(batch_shape, dtype=np.int)
        data = [self._get(i) for i in idx.reshape(batch_size)]
        return _stack(data, batch_shape)
    
    # To be compatible with the interface of _worker
    __call__ = get
    
    @staticmethod
    def get_parser(ps=None):
        if ps is None: ps = ArgumentParser()
        g = ps.add_argument_group('dataset')
        return ps

class SqDataset(Dataset):
    """Dataset returns the square of the index.
    """
    def __init__(self, args=Dataset.DEFAULT_ARGS, name="sq_dataset"):
        Dataset.__init__(self, args=args, name=name)
        
    def _get(self, idx):
        return np.square(self._data[idx])