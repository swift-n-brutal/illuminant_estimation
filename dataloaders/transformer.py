import numpy as np

class Transformer(object):
    """Transformer transforms an input to a specified format.
    """
    def __init__(self, name):
        self.name = name
        self._inner_fn = None
        
    def _reset(self):
        pass
        
    def reset(self):
        self._reset()
        if _inner_fn is not None:
            self._inner_fn.reset()
        
    def forward(self, x):
        raise NotImplemented
    
    def backward(self, x):
        raise NotImplemented
        
    def inverse(self, x):
        """Inverse the transform of x."""
        back_x = self.backward(x)
        if self._inner_fn is None:
            return back_x
        else:
            return self._inner_fn.inverse(back_x)
    
    def __add__(self, other):
        """Composite two transformers.
        For example, f+g denotes g(f(\cdot)) and reads "g following f",
        "g after f" or "g of f".
        """
        other._inner_fn = self
        return other
    
    def __str__(self):
        return "%s(%s)" % \
                (self.name, '_' if self._inner_fn is None else str(self._inner_fn))
        
    def __call__(self, x):
        """Evaluate the transformed x."""
        if self._inner_fn is None:
            return self.forward(x)
        else:
            return self.forward(self._inner_fn(x))

class InverseTransformer(Transformer):
    def __init__(self, transformer, name='inverse'):
        Transformer.__init__(self, name)
        self._transformer = transformer
        
    def inverse(self, x):
        return self._transformer(x)
    
    def __call__(self, x):
        return self._transformer.inverse(x)
            
class Add(Transformer):
    """Add a number
    """
    def __init__(self, a, name='add'):
        Transformer.__init__(self, name)
        self._a = a
        
    def forward(self, x):
        return x + self._a
    
    def backward(self, x):
        return x - self._a
    
class Mul(Transformer):
    """Multiply a number
    """
    def __init__(self, a, name='mul'):
        Transformer.__init__(self, name)
        self._a = a
        
    def forward(self, x):
        return x * self._a
    
    def backward(self, x):
        return x / self._a

class Log(Transformer):
    """Logarithm with small shift
    """
    def __init__(self, eps=1e-8, name='log'):
        Transformer.__init__(self, name)
        self._eps = eps
        
    def forward(self, x):
        return np.log(x + self._eps)
    
    def backward(self, x):
        return np.exp(x) - self._eps