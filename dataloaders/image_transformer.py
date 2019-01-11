import numpy as np

from .transformer import Transformer

class Normalize(Transformer):
    def __init__(self, mean, std, name='normalize'):
        Transformer.__init__(self, name)
        self._mean = mean
        self._std = std
        if mean is not None and len(mean.shape) == 1:
            self._mean = self._mean[np.newaxis, np.newaxis, :]
        if std is not None and len(std.shape) == 1:
            self._std = self._std[np.newaxis, np.newaxis, :]
        
    def forward(self, x):
        assert len(x.shape) == 3 and x.shape[-1] == self._mean.shape[-1] \
            and x.shape[-1] == self._mean.shape[-1], \
            'Invalid input shape: {}'.format(x.shape)
        return (x - self._mean) / self._std
        
    def backward(self, x):
        assert len(x.shape) == 3 and x.shape[-1] == self._mean.shape[-1], \
            'Invalid input shape: {}'.format(x.shape)
        return x * self._std + self._mean

class Pad(Transformer):
    def __init__(self, pad_size, pad_value=0., pad_mode='constant', name='pad'):
        Transformer.__init__(self, name)
        self._pad_size = pad_size
        self._pad_value = pad_value
        self._pad_mode = pad_mode
    
    def forward(self, x):
        assert len(x.shape) == 3, 'Invalid input shape: {}'.format(x.shape)
        p = self._pad_size
        x_padded = np.pad(x, ((p, p), (p, p), (0,0)), 
                         self._pad_mode, constant_values=self._pad_value)
        return x_padded
        
    def backward(self, x):
        return x

class Crop(Transformer):
    def __init__(self, crop_size, name='random_crop'):
        Transformer.__init__(self, name)
        self._crop_size = crop_size
        
    def forward(self, x):
        assert len(x.shape) == 3 and x.shape[0] >= self._crop_size \
            and x.shape[1] >= self._crop_size, \
            'Invalid input shape: {}'.format(x.shape)
        h_off = int((x.shape[0] - crop_size + 1) / 2)
        w_off = int((x.shape[1] - crop_size + 1) / 2)
        x_cropped = x[h_off:h_off+crop_size, w_off:w_off+crop_size, :]
        return x_cropped
    
    def backward(self, x):
        return x
    
class RandomCrop(Transformer):
    def __init__(self, crop_size, seed=None, name='random_crop'):
        Transformer.__init__(self, name)
        self._crop_size = crop_size
        self._seed = seed
        self._reset()
        
    def _reset(self):
        self._rand = np.random.RandomState(self._seed)
        
    def forward(self, x):
        assert len(x.shape) == 3 and x.shape[0] >= self._crop_size \
            and x.shape[1] >= self._crop_size, \
            'Invalid input shape: {}'.format(x.shape)
        h_off = self._rand.randint(x.shape[0] - self._crop_size + 1)
        w_off = self._rand.randint(x.shape[1] - self._crop_size + 1)
        x_cropped = x[h_off:h_off+self._crop_size, w_off:w_off+self._crop_size, :]
        return x_cropped
    
    def backward(self, x):
        return x
    
class RandomMirror(Transformer):
    def __init__(self, seed=None, name='random_mirror'):
        Transformer.__init__(self, name)
        self._seed = seed
        self._reset()
        
    def _reset(self):
        self._rand = np.random.RandomState(self._seed)
        
    def forward(self, x):
        assert len(x.shape) == 3, 'Invalid input shape: {}'.format(x.shape)
        if self._rand.randint(2) == 1:
            return x[:,::-1,:]
        return x
    
    def backward(self, x):
        return x

class Transpose(Transformer):
    def __init__(self, axes, name='transpose'):
        Transformer.__init__(self, name)
        assert len(axes) == 3, 'Invalid axes: {}'.format(axes)
        self._axes = axes
        self._inverse_axes = np.argsort(axes)
        
    def forward(self, x):
        assert len(x.shape) == 3, 'Invalid input shape: {}'.format(x.shape)
        return np.transpose(x, self._axes)
    
    def backward(self, x):
        return np.transpose(x, self._inverse_axes)

