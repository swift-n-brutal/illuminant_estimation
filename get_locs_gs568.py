import numpy as np
import os, os.path as osp
from gs568_dataset import GS568Dataset

def get_all_locs(ds, psize, stride, save_dir='.', drop_last=False):
    size = ds.size()
    for i in range(size):
        name = ds.get_name(i)
        shape = ds.get_shape(i)
        img, _ = ds.get(i)
        locs = list()
        if drop_last:
            has_last_col = False
            has_last_row = False
        else:
            has_last_col = ((shape[1]-psize) % stride) > 0
            has_last_row = ((shape[0]-psize) % stride) > 0
        for y in range(0, shape[0]-psize, stride):
            for x in range(0, shape[1]-psize, stride):
                if np.all(img[y:y+psize, x:x+psize, :]):
                    locs.append([y, y+psize, x, x+psize])
            if has_last_col:
                # last column
                if np.all(img[y:y+psize, shape[1]-psize:shape[1], :]):
                    locs.append([y, y+psize, shape[1]-psize, shape[1]])
        if has_last_row:
            # last row
            y = shape[0] - psize
            for x in range(0, shape[1]-psize, stride):
                if np.all(img[y:y+psize, x:x+psize, :]):
                    locs.append([y, y+psize, x, x+psize])
            if has_last_col:
                # last column
                if np.all(img[y:y+psize, shape[1]-psize:shape[1], :]):
                    locs.append([y, y+psize, shape[1]-psize, shape[1]])
        # save locs
        np.save(osp.join(save_dir, name), \
                np.array(locs, dtype=np.int32))
        print(i, name, shape, len(locs))

def get_parser(ps=None):
    ps = GS568Dataset.get_parser(ps)
    g = ps.add_argument_group('main')
    g.add_argument('--size', type=int, default=44)
    g.add_argument('--stride', type=int, default=4, help='Stride of adjacent patches.')
    g.add_argument('--save_dir', type=str, default='data/gs568/loc')
    return ps

def main():
    ps = get_parser()
    args = ps.parse_args()
    args = vars(args)
    #
    ds_train = GS568Dataset('train', args, False)
    ds_train._transformer = None # Disable random flipping
    ds_test = GS568Dataset('test', args, False)
    psize = args['size']
    stride = args['stride']
    save_dir = args['save_dir']
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    print('    Train    ')
    get_all_locs(ds_train, psize, stride, save_dir)
    print('    test    ')
    get_all_locs(ds_test, psize, stride, save_dir)

if __name__ == '__main__':
    main()
