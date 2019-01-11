import tensorflow as tf
import numpy as np
import os, os.path as osp
from time import time
from collections import defaultdict
from argparse import ArgumentParser
import traceback

from tfbox.config import NP_DTYPE, TF_DTYPE
from tfbox.solvers import Solver
from tfbox.layers import LpLoss, Accuracy
try:
    from dataloaders.zmq_dataloader import ZMQDataLoader as DataLoader
except ImportError:
    print 'ZeroMQ is not detected. It is recommended for efficient training.'
    from dataloaders.zmq_dataloader import DataLoader

from gs568_dataset import GS568Dataset as Dataset
from gs568_sampler import GS568BatchSerialSampler, GS568BatchRandomSampler
from model_builder import HypNet, SelNet
    
def get_merged_summary_op(summary_dict, ext_list=None):
    ret_list = [tf.summary.scalar(k,v) for k,v in summary_dict.items()]
    if ext_list is not None:
        ret_list += ext_list
    return tf.summary.merge(ret_list)

def dict_add(a, b):
    for k in b.keys():
        a[k] += b[k]
        
def dict_div(a, b):
    for k in a.keys():
        a[k] /= b

def print_dict(d):
    for k in sorted(d.keys()):
        print k, ':', d[k]
        
def strf(f):
    s = str(f)
    if '.' in s and (not (('e' in s) or ('E' in s))):
        return s.rstrip('0').rstrip('.')
    return s
        
class Timeline(object):
    """
    Training interval: (), Test interval: []
        {...(..)(..)(..)(..[....])...}
    """
    def __init__(self, name='timeline'):
        self.name = name
        self._start_t = dict() # start time of timer
        self._stack_k = list() # key of open timer in the current timeline
        self._stack_t = list() # accumulative time that should be subtracted from the outer timer
        # place guards in both stacks
        self._stack_k.append('_guard')
        self._stack_t.append(0.)
        
    def tic(self, k=None):
        # k should either be the top of the stack or be not in the stack
        t = self.toc(k)
        if k not in self._start_t:
            self._start_t[k] = time()
            self._stack_k.append(k)
            self._stack_t.append(0.)
        else:
            raise ValueError('%s is not on the top of the stack' % str(k))
        return t
        
    def toc(self, k=None):
        # k should be the top of the stack
        if self._stack_k[-1] == k:
            t = time() - self._start_t.pop(k)
            self._stack_k.pop()
            sub_t = self._stack_t.pop()
            self._stack_t[-1] += t
            return t - sub_t
        else:
            return 0.
                
    
class GS568Solver(Solver):
    def __init__(self, args, sess):
        Solver.__init__(self, args, sess)
    
    def setup(self):
        args = self.args
        #
        mom = args['mom']
        decay = args['decay']
        psize = args['patch_size']
        batch_size = args['batch_size']
        branches = args['hn_branches']
        # input shapes
        input_shape = (batch_size, psize, psize, 2)
        illum_shape = (batch_size, 2)
        # inputs
        lr = tf.placeholder(TF_DTYPE, shape=(), name='lr')
        inputs = tf.placeholder(TF_DTYPE, shape=input_shape, name='inputs')
        illums = tf.placeholder(TF_DTYPE, shape=illum_shape, name='illums')
        zeros = tf.zeros(shape=[batch_size], dtype=tf.int32, name='const_zero') # for counting wta/sel indices
        # model builder
        hypnet = HypNet(branches)
        selnet = SelNet(branches)
        for device_id in xrange(1):
            print "~~~~~~~~ Device %d ~~~~~~~~" % device_id
            with tf.device('/%cpu:%d' % ('c' if args['cpu_only'] else 'g', device_id)):
                ######
                # model
                model_hypnet = hypnet.build_model([inputs], name='train')
                model_selnet = selnet.build_model([inputs], name='train')
                pred_branch = model_hypnet.outputs
                logits = model_selnet.outputs[-1]
                    # prediction by SelNet
                pred_stacked = tf.stack(pred_branch, axis=0)
                sel_idx = tf.argmax(logits, axis=-1, output_type=tf.int32)
                sel_zero = Accuracy(logits, zeros, name='sel_zero').outputs[-1]
                sample_idx = tf.range(batch_size)
                pred_sel = tf.gather_nd(pred_stacked,
                                        tf.stack((sel_idx, sample_idx), axis=-1),
                                        name='pred_by_sel')
                ######
                # loss
                with tf.name_scope('loss'):
                    loss_hypsel = LpLoss(pred_sel, illums, 2, name='loss_hypsel').outputs[-1]
                    # hypnet
                    loss_branch = list()
                    for b in xrange(branches):
                        loss_branch.append(
                            LpLoss(pred_branch[b], illums, 2,
                                   samplewise=True, name='loss_b%d' % b).outputs[-1])
                    loss_stacked = tf.stack(loss_branch, axis=-1)
                    loss_hypnet = tf.reduce_mean(tf.reduce_min(loss_stacked, axis=-1),
                                                name='loss_hypnet')
                    wta_idx = tf.argmin(loss_stacked, axis=-1, output_type=tf.int32)
                    wta_zero = Accuracy(-loss_stacked, zeros, name='wta_zero').outputs[-1]
                    # selnet
                    loss_selnet = tf.reduce_mean(
                        tf.nn.sparse_softmax_cross_entropy_with_logits(
                            labels=wta_idx, logits=logits),
                        name='loss_selnet')
                    acc_selnet = Accuracy(logits, wta_idx, name='acc_selnet').outputs[-1]
                    # weight decay
                    if decay > 0:
                        loss_decay_hypnet = decay*model_hypnet.get_weight_decay()
                        loss_decay_selnet = decay*model_selnet.get_weight_decay()
                    else:
                        loss_decay_hypnet = 0.
                        loss_decay_selnet = 0.
                        
                ######
                # optimizer
                opt = tf.train.MomentumOptimizer(lr, mom)
                gvs_hypnet = opt.compute_gradients(loss_hypnet+loss_decay_hypnet,
                    var_list=model_hypnet.params, colocate_gradients_with_ops=True)
                gvs_selnet = opt.compute_gradients(loss_selnet+loss_decay_selnet,
                    var_list=model_selnet.params, colocate_gradients_with_ops=True)
                # update ops
                with tf.control_dependencies([gv[0] for gv in gvs_hypnet+gvs_selnet]):
                    ags_hypnet = opt.apply_gradients(gvs_hypnet, name='update_hypnet')
                    ags_selnet = opt.apply_gradients(gvs_selnet, name='update_selnet')
                    
                ######
                # restore or init
                self.restore_or_init()
                
                ######
                # attributes
                self.lr = lr
                self.inputs = inputs
                self.illums = illums
                    #
                self.model_hypnet = model_hypnet
                self.model_selnet = model_selnet
                self.pred_hyp = pred_stacked
                self.sel_logits = logits
                self.sel_idx = sel_idx
                self.sel_zero = sel_zero
                self.pred_sel = pred_sel
                self.loss_hypnet = loss_hypnet
                self.loss_hypsel = loss_hypsel
                self.loss_selnet = loss_selnet
                self.acc_selnet = acc_selnet
                self.wta_idx = wta_idx
                self.wta_zero = wta_zero
                    #
                self.ags_hypnet = ags_hypnet
                self.ags_selnet = ags_selnet
        # summary
        if not args['test_only']:
            with tf.name_scope('summary'):
                s_train = dict()
                s_train['loss_hypnet'] = self.loss_hypnet
                s_train['loss_selnet'] = self.loss_selnet
                s_train['acc_selnet'] = self.acc_selnet
                s_train['loss_hypsel'] = self.loss_hypsel
                s_train['sel_zero'] = self.sel_zero
                s_train['wta_zero'] = self.wta_zero
                self.merged_s_train = get_merged_summary_op(s_train)
                s_test = dict()
                s_test['test_loss_hypnet'] = tf.placeholder(TF_DTYPE, shape=(), name='test_loss_hypnet_ph')
                s_test['test_loss_selnet'] = tf.placeholder(TF_DTYPE, shape=(), name='test_loss_selnet_ph')
                s_test['test_acc_selnet'] = tf.placeholder(TF_DTYPE, shape=(), name='test_acc_selnet_ph')
                s_test['test_loss_hypsel'] = tf.placeholder(TF_DTYPE, shape=(), name='test_loss_hypsel_ph')
                s_test['test_sel_zero'] = tf.placeholder(TF_DTYPE, shape=(), name='test_sel_zero_ph')
                s_test['test_wta_zero'] = tf.placeholder(TF_DTYPE, shape=(), name='test_wta_zero_ph')
                self.merged_s_test = get_merged_summary_op(s_test)
                self.s_test = s_test
                self.writer = tf.summary.FileWriter(args['save_dir'], self.sess.graph)
                    
    def get_dataloader(self, name):
        if self._dataloader.get(name) is None:
            args = self.args
            batch_size = args['batch_size']
            if name == 'train':
                ds = Dataset('train', args)
                dl = DataLoader(ds, GS568BatchRandomSampler, (args['batch_size'],))
                try:
                    dl.add_prefetch_process(args['nproc'])
                except AttributeError:
                    pass
            elif name == 'test':
                ds = Dataset('test', args)
                dl = DataLoader(ds, GS568BatchSerialSampler, (args['batch_size'], 128))
            else:
                dl = None
            self._dataloader[name] = dl
        return self._dataloader[name]
        
    def step(self, lr):
        # training step
        imgs, illums = self.get_dataloader('train').next()
        fetch_dict = {'loss_hypnet': self.loss_hypnet,
                      'loss_selnet': self.loss_selnet,
                      'acc_selnet': self.acc_selnet,
                      'loss_hypsel': self.loss_hypsel,
                      'sel_zero': self.sel_zero,
                      'wta_zero': self.wta_zero,
                      'summary': self.merged_s_train,
                      'ags_selnet': self.ags_selnet,
                      'ags_hypnet': self.ags_hypnet}
        feed_dict = {self.lr: lr,
                    self.inputs: imgs,
                    self.illums: illums}
        fetch_val = self.sess.run(fetch_dict, feed_dict)
        fetch_val.pop('ags_selnet')
        fetch_val.pop('ags_hypnet')
        return fetch_val
        
    def train(self):
        args = self.args
        lr = NP_DTYPE(args['lr'])
        saver = self.get_saver()
        timer = Timeline()
        
        timer.tic('train')
        for itr in xrange(args['mxit'] + 1):
            if itr >= args['lr_update_after'] and (itr - args['lr_update_after']) % args['lr_update_every'] == 0:
                lr *= args['lr_decay']
            # test
            if args['tsit'] > 0 and itr % args['tsit'] == 0 and (itr > 0 or args['test_first']):
                timer.tic('test')
                val = self.test(itr)
                self.writer.add_summary(val['summary'], itr)
                print 'Test [%d](%.2f)' % (itr, timer.toc('test')),
                print 'lhyp: %.4e|' % val['loss_hypnet'],
                print 'lsel: %.4e|' % val['loss_selnet'],
                print 'asel: %.4f|' % val['acc_selnet'],
                print 'lhs: %.4e|' % val['loss_hypsel'],
                print 'sel0: %.4f|' % val['sel_zero'],
                print 'wta0: %.4f|' % val['wta_zero'],
                print 'lr: %.4e|' % lr
            # snapshot
            if itr > 0 and itr % args['ssit'] == 0:
                timer.tic('snap')
                #self.save(self.sess, saver, self.save_dir, global_step=itr)
                self.model_hypnet.save_model(self.sess,
                        osp.join(self.save_dir, 'hypnet_%d.npz' % itr))
                self.model_selnet.save_model(self.sess,
                        osp.join(self.save_dir, 'selnet_%d.npz' % itr))
                print 'Snapshot (%.2f)' % timer.toc('snap')
            # train
            val = self.step(lr)
            self.writer.add_summary(val['summary'], itr)
            # display
            if itr > 0 and itr % args['dpit'] == 0:
                print '[%d](%.2f)' % (itr, timer.tic('train')),
                print 'lhyp: %.4e|' % val['loss_hypnet'],
                print 'lsel: %.4e|' % val['loss_selnet'],
                print 'asel: %.4f|' % val['acc_selnet'],
                print 'lhs: %.4e|' % val['loss_hypsel'],
                print 'sel0: %.4f|' % val['sel_zero'],
                print 'wta0: %.4f|' % val['wta_zero']
                
    def test(self, itr):
        args = self.args
        dl = self.get_dataloader('test')
        count = 0
        fetch_dict = {'loss_hypnet': self.loss_hypnet,
                      'loss_selnet': self.loss_selnet,
                      'acc_selnet': self.acc_selnet,
                      'loss_hypsel': self.loss_hypsel,
                      'sel_zero': self.sel_zero,
                      'wta_zero': self.wta_zero}
        val = defaultdict(float)
        for imgs, illums in dl:
            feed_dict = {self.inputs: imgs, self.illums: illums}
            fetch_val = self.sess.run(fetch_dict, feed_dict)
            dict_add(val, fetch_val)
            count += 1
        dl.reset()
        dict_div(val, count)
        s_feed_dict = dict([(self.s_test['test_'+k], val[k]) for k in fetch_dict.keys()])
        s = self.sess.run(self.merged_s_test, s_feed_dict)
        val['summary'] = s
        return val
        
    def test_only(self):
        args = self.args
        # In the training stage, we don't use TF's methods to snapshot models.
        # Need to load models manually.
        save_dir = args['save_dir']
        mxit = args['mxit']
        self.model_hypnet.load_model(self.sess,
                osp.join(save_dir, 'hypnet_%d.npz' % mxit))
        self.model_selnet.load_model(self.sess,
                osp.join(save_dir, 'selnet_%d.npz' % mxit))
        #
        batch_size = args['batch_size']
        patch_size = args['patch_size']
        chns = 2
        inputs = np.zeros((batch_size, patch_size, patch_size, chns), dtype=NP_DTYPE)
        ds = Dataset('test', args)
        transformer = ds._illum_transformer
        sampler = GS568BatchSerialSampler(ds, batch_size, None)
        #
        feed_dict = {self.inputs: inputs}
        fetch_dict = {'pred_hyp': self.pred_hyp,
                     'sel_logits': self.sel_logits}
        #
        pred_hyp = defaultdict(list)
        sel_logits = defaultdict(list)
        d_loc_idx = defaultdict(list)
        #
        size = sampler.size()
        step = size * 0.01
        count = 0
        prog = 0
        timer = Timeline()
        timer.tic('test')
        for idx in sampler:
            imgs, illums = ds.get(idx)
            n_imgs = imgs.shape[0]
            inputs[0:n_imgs,...] = imgs[...]
            fetch_val = self.sess.run(fetch_dict, feed_dict)
            pred = fetch_val['pred_hyp']
            if transformer is not None:
                pred = transformer.inverse(pred)
            logits = fetch_val['sel_logits']
            for i in xrange(n_imgs):
                img_idx, loc_idx = idx[i]
                pred_hyp[img_idx].append(pred[:,i,:])
                sel_logits[img_idx].append(logits[i,:])
                d_loc_idx[img_idx].append(loc_idx)
            count += n_imgs 
            if int(count / step) != prog:
                prog = int(count / step)
                print '[%.2f%%,%d/%d](%.2f)' % (count * 100. / size, count, size, timer.tic('test'))
        #
        if not osp.exists(args['test_dir']):
            os.makedirs(args['test_dir'])
        for img_idx in pred_hyp.keys():
            name = ds.get_name(img_idx)
            locs = ds.get_loc(img_idx)
            locs = locs[d_loc_idx[img_idx]]
            np.savez(osp.join(args['test_dir'], name+'.npz'),
                     preds=np.array(pred_hyp[img_idx], dtype=NP_DTYPE),
                     logits=np.array(sel_logits[img_idx], dtype=NP_DTYPE),
                     locs=locs)
        print 'Done(%.2f)' % (timer.toc('test'))
        
    @staticmethod
    def get_parser(ps=None):
        if ps is None: ps = ArgumentParser()
        ps = Dataset.get_parser(ps)
        ps = HypNet.get_parser(ps)
        #
        g = ps.add_argument_group('solver')
        g.add_argument('--name', type=str, default='gs568')
        g.add_argument('--patch-size', type=int, default=44)
        g.add_argument('--batch-size', type=int, default=64)
        g.add_argument('--nproc', type=int, default=2)
        g.add_argument('--lr', type=float, default=0.02)
        g.add_argument('--mom', type=float, default=0.9)
        g.add_argument('--mom2', type=float, default=0.99)
        g.add_argument('--decay', type=float, default=5e-6)
        g.add_argument('--save-dir', type=str, default='models')
        g.add_argument('--lr_update_after', type=int, default=2000000)
        g.add_argument('--lr_update_every', type=int, default=2000000)
        g.add_argument('--lr_decay', type=float, default=0.1)
        g.add_argument('--mxit', type=int, default=4000000)
        g.add_argument('--dpit', type=int, default=200)
        g.add_argument('--tsit', type=int, default=4000)
        g.add_argument('--ssit', type=int, default=200000)
        g.add_argument('--test-first', action='store_true', default=False)
        g.add_argument('--test-only', action='store_true', default=False)
        g.add_argument('--test-dir', type=str, default='preds')
        g.add_argument('--max-to-keep', type=int, default=20)
        g.add_argument('--cpu-only', action='store_true', default=False)
        
        return ps
    
def main():
    ps = GS568Solver.get_parser()
    args = ps.parse_args()
    args = vars(args)
    args['save_dir'] = osp.join(args['save_dir'],
            '%s-%d_bs%d_lr%s' % (
            args['name'], args['gs_test_set'],
            args['batch_size'], strf(args['lr'])))
    args['filler'] = ('msra', 0., 1.)
    
    print_dict(args)
    
    gpu_options = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    with tf.Session(config=config) as sess:
        solver = GS568Solver(args, sess)
        try:
            if args['test_only']:
                solver.test_only()
            else:
                solver.train()
        except KeyboardInterrupt:
            print 'Interrupted by user'
        except:
            print 'Unexpected error'
            traceback.print_exc()
            
if __name__ == '__main__':
    main()
