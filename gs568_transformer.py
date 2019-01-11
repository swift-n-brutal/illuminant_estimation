import numpy as np

from dataloaders.transformer import Transformer

class ToUV(Transformer):
    def __init__(self, eps=1e-8, name='to_uv'):
        Transformer.__init__(self, name)
        self._eps = eps
        
    def forward(self, x):
        assert x.shape[-1] == 3, 'Invalid shape: {}'.format(x.shape)
        y = np.log(x[...,(0,2)] / x[...,(1,)] + self._eps)
        return y
    
    def backward(self, x):
        return np.exp(x) - self._eps