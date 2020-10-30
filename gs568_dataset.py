import numpy as np
from argparse import ArgumentParser
import os.path as osp
import struct

from dataloaders.dataset import Dataset
from dataloaders.transformer import Log
from dataloaders.image_transformer import RandomMirror
from gs568_transformer import ToUV

EXT_BIN = '.bin'
EXT_LOC = '.npy'

def _get_transformer(phase, to_uv):
    if phase == 'train':
        if to_uv:
            return RandomMirror()+ToUV()
        else:
            return RandomMirror()
    elif phase == 'test':
        if to_uv:
            return ToUV()
        else:
            return None
    else:
        raise ValueError("phase should be in ['train', 'test'], %s given" % str(phase))

class GS568Dataset(Dataset):
    def __init__(self, phase, args, to_uv=True, cache=True, name='gs568_dataset'):
        """
        Parameters
        ----------
        cache : bool
            Whether to load all images into memory. Set cache=False if your memory
            is not large enough to fit your data. 
        """
        Dataset.__init__(self, args, name)
        self._cache = cache
        assert phase == 'train' or phase == 'test' or phase == 'all', \
            "phase should be in ['train', 'test', 'all'], %s given" % str(phase)
        self._phase = phase
        self._transformer = _get_transformer(phase, to_uv)
        self._illum_transformer = ToUV() if to_uv else None
        self._folder = args["gs_folder"]
    
    def _load_names(self, path):
        names = list()
        illums = list()
        with open(path, 'r') as fin:
            line = fin.readline().strip('\n\r')
            n_names = int(line)
            line = fin.readline().strip('\n\r') # header: name r g b
            for i in range(n_names):
                line = fin.readline().strip('\n\r')
                name, r, g, b = line.split(' ')
                names.append(name)
                illums.append(np.array([r,g,b], dtype=np.float32))
        return names, illums

    def _load_bin(self, name):
        with open(name, 'rb') as fin:
            data = fin.read()
            h, w, c = struct.unpack('iii', data[:12])
            img = np.frombuffer(data, dtype=np.float32, count=-1, offset=12)
            assert h*w*c == img.size, '%d * %d * %d != %d' % (h, w, c, img.size)
            return np.transpose(img.reshape((c, w, h)), (2, 1, 0)) # The binary file is in matlab format.
    
    def _load_locs(self, folder, names):
        locs = list()
        for nm in names:
            locs.append(np.load(osp.join(folder, nm + EXT_LOC)))
        return locs
    
    def _init(self):
        args = self._args
        folder = args['gs_folder']
        if self._phase == 'train':
            set_ids = list(range(3))
            set_ids.remove(args['gs_test_set'])
        elif self._phase == 'test':
            set_ids = [args['gs_test_set']]
        else:
            set_ids = list(range(3))
        # load names and ground truth
        names, illums = list(), list()
        for i in set_ids:
            nm, il = self._load_names(osp.join(folder, 'set_%d.txt' % i))
            names.extend(nm)
            illums.extend(il)
        self._names = names
        self._illums = illums
        self._size = len(names)
        self._data = None
        # if has_loc, load locations of all valid patches
        self._locs = None
        if args['gs_has_loc']:
            self._locs = self._load_locs(args['gs_loc_folder'], names)
    
    def _get(self, idx):
        # if cache, load all images
        if self._cache and self._data is None:
            data = list()
            for nm in self._names:
                data.append(self._load_bin(osp.join(self._args['gs_folder'], nm + EXT_BIN)))
            self._data = data
        if idx.dtype.names is None:
            # idx is a scalar
            if self._cache:
                img = self._data[idx]
            else:
                img = self._load_bin(osp.join(self._folder, self._names[idx] + EXT_BIN))
            illum = self._illums[idx]
        else:
            # idx is a structured array
            img_idx = idx['idx']
            loc = idx['loc']
            if idx['loc'].size == 1:
                # loc represents the index of locs
                loc = self._locs[img_idx][loc]
            if self._cache:
                img = self._data[img_idx][loc[0]:loc[1], loc[2]:loc[3], :]
            else:
                img = self._load_bin(osp.join(self._folder, self._names[img_idx] + EXT_BIN))
                img = img[loc[0]:loc[1], loc[2]:loc[3], :]
            illum = self._illums[img_idx]
        if self._transformer is not None:
            img = self._transformer(img)
        if self._illum_transformer is not None:
            illum = self._illum_transformer(illum)
        return img, illum
    
    def get_name(self, idx):
        return self._names[idx]
    
    def get_shape(self, idx):
        return self._data[idx].shape
    
    def get_loc(self, idx):
        return self._locs[idx]
    
    @staticmethod
    def get_parser(ps=None):
        if ps is None: ps = ArgumentParser()
        g = ps.add_argument_group('gs568_dataset')
        #
        g.add_argument('--gs-folder', type=str, default='data/gs568')
        g.add_argument('--gs-test-set', type=int, default=0, choices=[0,1,2],
                       help='One of three splits of data is regarded as test set. The other two are used for training.')
        g.add_argument('--gs-has-loc', action='store_true', default=False,
                       help='Whether the locations of valid (without zero values) patches are provided in the folder. \
                       If not provided, the location is sampled and checked at runtime.')
        g.add_argument('--gs-loc-folder', type=str, default='data/gs568/loc')
        return ps
