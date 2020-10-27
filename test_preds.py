"""
Test the angular errors of predictions by (plain or weighted) median pooling of local estimations.
The local estimations is produced by running
    python solver.py --test_only --gs-test-set 0
    python solver.py --test_only --gs-test-set 1
    python solver.py --test_only --gs-test-set 2
"""

import numpy as np
import math
import os.path as osp
from argparse import ArgumentParser

from gs568_dataset import GS568Dataset

RAD2DEG = 180. / math.pi

def compute_angular_error(pred, target):
    # pred = [r/g, b/g], target = [r, g, b]
    assert pred.shape == (2,) and target.shape == (3,)
    ip = pred[0] * target[0] + 1. * target[1] + pred[1] * target[2]
    norm = np.sqrt((pred[0]*pred[0] + pred[1]*pred[1] + 1.) * \
                  (target[0]*target[0] + target[1]*target[1] + target[2]*target[2]))
    return math.acos(ip/norm) * RAD2DEG

def get_median(preds):
    return np.median(preds, axis=0)

def get_confidence_weights(img, locs):
    weights = np.zeros(locs.shape[0], dtype=np.float32)
    minchn = np.min(img, axis=-1)
    for i in range(locs.shape[0]):
        weights[i] = np.mean(minchn[locs[i][0]:locs[i][1], locs[i][2]:locs[i][3]])
    return weights / np.max(weights)
    
def get_weighted_median(preds, weights):
    n, c = preds.shape
    weighted_median = np.zeros(c)
    for i in range(c):
        temp = zip(preds[:,i], weights)
        sorted_pw = sorted(temp, key=lambda x: x[0])
        sorted_p, sorted_w = zip(*sorted_pw)
        cum_w = np.cumsum(sorted_w)
        med_w = cum_w[-1] *0.5
        index = np.searchsorted(cum_w, med_w)
        weighted_median[i] = (sorted_p[index-1] * (cum_w[index] - med_w) \
                + sorted_p[index] * (med_w - cum_w[index-1])) / sorted_w[index]
    return weighted_median

def get_selected_preds(hyp_preds, logits):
    n_preds = hyp_preds.shape[0]
    sel_idx = np.argmax(logits, axis=1)
    return hyp_preds[range(n_preds), sel_idx]

def test_split(args, test_set_id):
    args['gs_test_set'] = test_set_id
    ds = GS568Dataset('test', args, to_uv=False)
    n_imgs = ds.size()
    ae_list = list()
    for i in range(n_imgs):
        img, illum = ds.get(np.array(i))
        name = ds.get_name(i)
        data = np.load(osp.join(args['pred_dir'], name+'.npz'))
        hyp_preds = data['preds']
        logits = data['logits']
        preds = get_selected_preds(hyp_preds, logits)
        if args['weighted_median']:
            weights = get_confidence_weights(img, data['locs'])
            pred = get_weighted_median(preds, weights)
        else:
            pred = get_median(preds)
        ae = compute_angular_error(pred, illum)
        print('%s %.4f' % (name, ae),
                pred, illum / illum[1])
        ae_list.append(ae)
    return ae_list

def print_errors(ae_all):
    e_avg = np.mean(ae_all)
    e_med = np.median(ae_all)
    e_min = np.min(ae_all)
    e_max = np.max(ae_all) 
    e_tri = (np.percentile(ae_all, 25) + np.percentile(ae_all, 75) + e_med * 2) / 4
    e_p95 = np.percentile(ae_all, 95)
    p25 = np.percentile(ae_all, 25)
    p75 = np.percentile(ae_all, 75)
    e_l25 = np.mean(ae_all[ae_all <= p25])
    e_h25 = np.mean(ae_all[ae_all >= p75])
    err_all = {'avg':e_avg,
           'med':e_med,
           'tri':e_tri,
           'p95':e_p95,
           'min':e_min,
           'max':e_max,
           'l25':e_l25,
           'h25':e_h25}
    for (k,v) in err_all.items():
        print(k, ":", v)
    return err_all
        
def main():
    ps = get_parser()
    args = vars(ps.parse_args())
    #
    test_set_id = args['test_set_id']
    ae_list = list()
    if test_set_id in [0, 1, 2]:
        ae_list += test_split(args, test_set_id)
    else:
        for i in range(3):
            ae_list += test_split(args, i)
    ae_arr = np.array(ae_list)
    print_errors(ae_arr)

def get_parser(ps=None):
    if ps is None: ps = ArgumentParser()
    ps = GS568Dataset.get_parser(ps)
    g = ps.add_argument_group('test_preds')
    g.add_argument('--pred-dir', type=str, default='preds/')
    g.add_argument('--test-set-id', type=int, default=-1)
    g.add_argument('--weighted-median', action='store_true', default=False)
    return ps
    
if __name__ == '__main__':
    main()
