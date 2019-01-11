
class DataLoader(object):
    """Data Loader iterates the data in dataset using sampler.
    
    Any derived class should implement its own methods of:
        * _init()
        * _reset()
        * _get(idx)
        * _get_next()
        
    """
    def __init__(self, dataset, sampler_cls, sampler_args=(), transformer=None,
                 name='data_loader'):
        self._dataset = dataset
        self._sampler_cls = sampler_cls
        self._sampler_args = sampler_args
        self._sampler = None
        self._transformer = transformer
        self.name = name
        #
        self._initialized = False
    
    def _init(self):
        if self._sampler is None:
            self._sampler = self._sampler_cls(self._dataset, *(self._sampler_args))
    
    def _reset(self):
        self._sampler.reset()
    
    def _get(self, idx):
        data = self._dataset(idx)
        if self._transformer is not None:
            data = self._transformer(data)
        return data
            
    def _get_next(self):
        idx = self._sampler.next()
        return self._get(idx)
        
    def _check_init(self):
        if not self._initialized:
            self._init()
            self._initialized = True
        
    def get(self, idx):
        self._check_init()
        return self._get(idx)
        
    def reset(self):
        self._check_init()
        self._reset()
        
    def __iter__(self):
        self._check_init()
        return self
        
    def __next__(self):
        self._check_init()
        return self._get_next()
    
    next = __next__