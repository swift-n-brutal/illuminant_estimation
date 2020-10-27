import numpy as np

from dataloaders.sampler import Sampler

class GS568BatchSerialSampler(Sampler):
    """
    Parameters
    ----------
    batch_shape : tuple, list or array-like object
        The shape of mini-batch
    patches_per_image : int or None
        The number of patches sampled per image. If None, all patches are used.
    """
    def __init__(self, dataset, batch_shape, patches_per_image=128,
                 keep_last=True, seed=0, replace=False, name='gs568_batch_serial_sampler'):
        Sampler.__init__(self, dataset, name)
        if isinstance(batch_shape, int):
            self._batch_shape = (batch_shape,)
        else:
            self._batch_shape = batch_shape
        self._batch_size = np.prod(batch_shape, dtype=np.int)
        self._patches_per_image = patches_per_image
        self._keep_last = keep_last
        self._seed = seed
        self._replace = replace
        
    def _init(self):
        self._n_images = self._dataset.size()
        self._size = 0
        self._locs = list()
        rand = np.random.RandomState(self._seed)
        if self._patches_per_image is None:
            # self._locs stores the length of indices
            for i in range(self._n_images):
                n_locs = len(self._dataset.get_loc(i))
                self._locs.append(n_locs)
                self._size += n_locs
        else:
            # self._locs stores the array of indices
            for i in range(self._n_images):
                n_locs = len(self._dataset.get_loc(i))
                self._locs.append(
                    rand.choice(n_locs, self._patches_per_image, replace=self._replace))
                self._size += self._patches_per_image
        self._reset()
        self._dtype = np.dtype([('idx', 'i4'), ('loc', 'i4')])
           
    def _reset(self):
        self._img_index = 0
        self._loc_index = 0
        
    def _get_next(self):
        if self._img_index >= self._n_images:
            raise StopIteration
        if self._patches_per_image is None:
            ret = (self._img_index, self._loc_index)
            self._loc_index += 1
            if self._loc_index >= self._locs[self._img_index]:
                self._img_index += 1
                self._loc_index = 0
        else:
            ret = (self._img_index, self._locs[self._img_index][self._loc_index])
            self._loc_index += 1
            if self._loc_index >= self._patches_per_image:
                self._img_index += 1
                self._loc_index = 0
        return ret
        
    def _sample(self):
        ret = list()
        try:
            for _ in range(self._batch_size):
                ret.append(self._get_next())
        except StopIteration:
            if self._keep_last and len(ret) > 0:
                ret = np.array(ret, dtype=self._dtype)
            else:
                raise
        else:
            ret = np.array(ret, dtype=self._dtype).reshape(self._batch_shape)
        return ret
    
    def size(self):
        self._check_init()
        return self._size
                
class GS568BatchRandomSampler(Sampler):
    """
    Parameters
    ----------
    batch_shape : tuple, list or array-like object
        The shape of mini-batch
    """
    def __init__(self, dataset, batch_shape, seed=None, replace=False, name='gs568_batch_random_sampler'):
        Sampler.__init__(self, dataset, name)
        if isinstance(batch_shape, int):
            self._batch_shape = (batch_shape,)
        else:
            self._batch_shape = batch_shape
        self._batch_size = np.prod(batch_shape, dtype=np.int)
        self._seed = seed
        self._replace = replace
        
    def _init(self):
        self._n_images = self._dataset.size()
        self._size = 0
        self._locs = list()
        for i in range(self._n_images):
            n_locs = len(self._dataset.get_loc(i))
            self._locs.append(n_locs)
            self._size += n_locs
        self._reset()
        self._dtype = np.dtype([('idx', 'i4'), ('loc', 'i4')])
        
    def _reset(self):
        self._rand = np.random.RandomState(self._seed)
        
    def _sample(self):
        img_idx = self._rand.choice(self._n_images, size=self._batch_size, replace=self._replace)
        loc_idx = [self._rand.randint(self._locs[ii]) for ii in img_idx]
        return np.array(list(zip(img_idx, loc_idx)), dtype=self._dtype).reshape(self._batch_shape)
