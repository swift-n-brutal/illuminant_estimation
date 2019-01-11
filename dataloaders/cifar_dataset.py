from argparse import ArgumentParser
import os.path as osp
import lmdb
import numpy as np
import caffe.proto as caproto

from .dataset import Dataset
from .transformer import Transformer
from .image_transformer import Normalize, Pad, Crop, RandomCrop, RandomMirror, Transpose
from tfbox.config import NP_DTYPE

MEAN = np.array([125.3, 123.0, 113.9], dtype=NP_DTYPE)
STD = np.array([63.0, 62.1, 66.7], dtype=NP_DTYPE)

class CifarDataset(Dataset):
    def __init__(self, n_classes, phase, args, key_length=5, name='cifar_dataset'):
        Dataset.__init__(self, args, name)
        assert n_classes == 10 or n_classes == 100, \
            "n_classes should be in [10, 100], %s given" % str(n_classes)
        assert phase == 'train' or phase == 'test', \
            "phase should be in ['train', 'test'], %s given" % str(phase)
        self._n_classes = n_classes
        self._phase = phase
        self._key_length = key_length
    
    def _init(self):
        path = osp.join(self._args['cd_lmdb_root'],
                        'cifar%d_%s_lmdb' % (self._n_classes, self._phase))
        self._env = lmdb.open(path, readonly=True)
        self._txn = self._env.begin()
        self._key_format = '%%0%dd' % self._key_length
        self._size = self._env.stat()['entries']
        self._transformer = CifarTransformer(self._phase)
    
    def _get(self, idx):
        # TODO get batch or get one
        raw_datum = self._txn.get(self._key_format % idx)
        datum = caproto.caffe_pb2.Datum()
        datum.ParseFromString(raw_datum)
        flat_x = np.fromstring(datum.data, dtype=np.uint8)
        x = flat_x.reshape(datum.channels, datum.height, datum.width)
        x_transformed = self._transformer(x)
        return np.array(idx), x_transformed, np.array(datum.label)
    
    def get_transformer(self):
        self._check_init()
        return self._transformer
    
    def reset(self):
        self._check_init()
        self._transformer.reset()
    
    @staticmethod
    def get_parser(ps=None):
        if ps is None: ps = ArgumentParser()
        g = ps.add_argument_group('cifar_dataset')
        #
        g.add_argument('--cd-lmdb-root', type=str, default='/home/sw015/data/cifar10')
        return ps
    
class CifarTransformer(Transformer):
    def __init__(self, phase, name='cifar_transformer'):
        Transformer.__init__(self, name)
        if phase == 'train':
            self._transformer = Transpose((1,2,0))+Normalize(MEAN, STD)+Pad(4)+RandomCrop(32)+RandomMirror()
        elif phase == 'test':
            self._transformer = Transpose((1,2,0))+Normalize(MEAN, STD)
        else:
            raise ValueError("phase should be in ['train', 'test'], %s given" % str(phase))
            
    def reset(self):
        self._transformer.reset()
            
    def inverse(self, x):
        return self._transformer.inverse(x)
            
    def __call__(self, x):
        return self._transformer(x)
        