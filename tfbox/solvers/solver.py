import os
import os.path as osp
import tensorflow as tf

from ..utils import gen_seed

class Solver(object):
    def __init__(self, args, sess,
            name='solver', save_dir='results',
            max_to_keep=5):
        self.sess = sess
        self.args = args
        #
        self.name = args.get('name')
        if self.name is None:
            self.name = name
        #
        self.save_dir = args.get('save_dir')
        if self.save_dir is None:
            self.save_dir = save_dir
        #
        self.max_to_keep = args.get('max_to_keep')
        if self.max_to_keep is None:
            self.max_to_keep = max_to_keep
        #
        self._saver = None
        self._seed_generator = None
        self._dataloader = dict()
        with tf.name_scope(self.name):
            self.setup()
    
    def get_saver(self):
        if self._saver is None:
            self._saver = tf.train.Saver(max_to_keep=self.max_to_keep)
        return self._saver

    def next_seed(self, num=None):
        if self._seed_generator is None:
            self._seed_generator = gen_seed()
        if type(num) is int:
            return [self._seed_generator.next() for _ in xrange(num)]
        else:
            return self._seed_generator.next()

    def restore_or_init(self):
        sess = self.sess
        saver = self.get_saver()
        load_dir = self.args.get('load_dir')
        global_step = self.args.get('global_step')
        if global_step == -1:
            global_step = None
        if load_dir is None:
            load_dir = self.save_dir
        try:
            load_path = self.restore(sess, saver, load_dir, global_step=global_step)
            print 'Restored from', load_path
        except:
            sess.run(tf.global_variables_initializer())
            save_path = self.save(sess, saver, self.save_dir)
            print 'Initialization done. Saved to', save_path

    def setup(self):
        raise NotImplementedError

    def get_dataloader(self, name):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError
    
    def test(self):
        raise NotImplementedError

    @staticmethod
    def restore(sess, saver, load_dir, global_step=None):
        state = tf.train.get_checkpoint_state(load_dir)
        basename = osp.basename(state.model_checkpoint_path)
        if global_step is not None:
            assert type(global_step) is int, "global_step should be an integer."
            splited = basename.rsplit('-', 1)
            if len(splited) == 2:
                try:
                    latest_step = int(splited[1])
                    basename = "%s-%d" % (splited[0], global_step)
                except:
                    basename = "%s-%d" % (basename, global_step)
            else:
                basename = "%s-%d" % (basename, global_step)
        load_path = osp.join(load_dir, basename)
        print "Try to restore from", load_path
        saver.restore(sess, load_path)
        return load_path

    @staticmethod
    def save(sess, saver, save_dir, global_step=None):
        if not osp.exists(save_dir):
            os.makedirs(save_dir) 
        save_path = osp.join(save_dir, 'model')
        if global_step is None:
            return saver.save(sess, save_path, write_meta_graph=True)
        else:
            return saver.save(sess, save_path, global_step=global_step, write_meta_graph=False)
