import numpy as np

class Sampler(object):
    """Sampler always returns None.
    
    A sampler is an instance of iterator over the indices of a dataset. The
    internal state is initialized (by calling reset()) when __iter__ is called
    in the first run.
    
    Any derived class should implements its own methods of:
        * _init()
        * _reset()
        * _sample()
    NOTE: "dataset" should always be the first parameter in __init__ function.
    """
    def __init__(self, dataset, name='sampler'):
        self._dataset = dataset
        self.name = name
        self._initialized = False
    
    def _init(self):
        pass
    
    def _reset(self):
        pass
    
    def _sample(self):
        return None
    
    def _check_init(self):
        if not self._initialized:
            self._init()
            self._initialized = True
    
    def reset(self):
        self._check_init()
        self._reset()
    
    def __iter__(self):
        self._check_init()
        return self
    
    def __next__(self):
        self._check_init()
        return self._sample()
    
    next = __next__
    
    def __call__(self, inp=None):
        """To be compatible with the interface of _worker
        """
        return self.next()
        
class SerialSampler(Sampler):
    """Serial Sampler returns index from 0 to (dataset.size() - 1) in sequence.
    """
    def __init__(self, dataset, loop=False, name='serial_sampler'):
        Sampler.__init__(self, dataset, name)
        self._loop = loop
    
    def _init(self):
        self._size = self._dataset.size()
        self._reset()
    
    def _reset(self):
        self._index = 0
        
    def _sample(self):
        if self._index >= self._size:
            if self._loop:
                self._reset()
            else:
                raise StopIteration
        ret = np.array(self._index)
        self._index += 1
        return ret
    
class RandomSampler(Sampler):
    """Random Sampler returns random index in the range [0, dataset.size()-1].
    """
    def __init__(self, dataset, seed=None, name='random_sampler'):
        Sampler.__init__(self, dataset, name)
        self._seed = seed
    
    def _init(self):
        self._size = self._dataset.size()
        self._reset()
        
    def _reset(self):
        self._rand = np.random.RandomState(self._seed)
        
    def _sample(self):
        return self._rand.randint(self._size, size=())
    
class BatchSerialSampler(SerialSampler):
    """
    Parameters
    ----------
    keep_last : boolean
        Whether to keep the last fragment of data when it is not enough to form a batch.
    """
    def __init__(self, dataset, batch_shape, loop=False, keep_last=True, name='batch_serial_sampler'):
        SerialSampler.__init__(self, dataset, loop, name)
        if isinstance(batch_shape, int):
            self._batch_shape = (batch_shape,)
        else:
            self._batch_shape = batch_shape
        self._batch_size = np.prod(batch_shape, dtype=np.int)
        self._keep_last = keep_last
        
    def _sample(self):
        ret = list()
        try:
            for _ in xrange(self._batch_size):
                ret.append(SerialSampler._sample(self))
        except StopIteration:
            if self._keep_last and len(ret) > 0:
                ret = np.array(ret)
            else:
                raise
        else:
            ret = np.array(ret).reshape(self._batch_shape)
        return ret
    
class BatchRandomSampler(RandomSampler):
    """
    """
    def __init__(self, dataset, batch_shape, seed=None, replace=False, name='batch_random_sampler'):
        RandomSampler.__init__(self, dataset, seed, name)
        if isinstance(batch_shape, int):
            self._batch_shape = (batch_shape,)
        else:
            self._batch_shape = batch_shape
        self._replace = replace
        
    def _sample(self):
        return self._rand.choice(self._size, size=self._batch_shape, replace=self._replace)