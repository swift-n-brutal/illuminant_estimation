import numpy as np
import signal
import traceback, sys
import time
import zmq
import os

from multiprocessing import Process
from .utils.multiprocessing import ZMQQueue as Queue, Empty, empty_queue
from .handler import Handler
from .dataloader import DataLoader

class WorkerHandler(Handler):
    """Signal handler for workers
    
    Use SIGTERM as the signal to stop.
    Ignore SIGINT.
    """
    SIGSTOP = signal.SIGTERM
    def __init__(self, name="worker_handler"):
        Handler.__init__(self, name)
        self.register_signal(WorkerHandler.SIGSTOP)
        self.register_signal(signal.SIGINT)
        
    def is_alive(self):
        return not self._flag[WorkerHandler.SIGSTOP]

class ResettableWorkerHandler(WorkerHandler):
    """Signal handler for workers
    
    Use SIGTERM as the signal to stop.
    Use SIGUSR1 as the signal to reset.
    Use SIGUSR2 as the signal to start (after reset).
    Ignore SIGINT.
    """
    SIGRESET = signal.SIGUSR1
    SIGSTART = signal.SIGUSR2
    def __init__(self, name="resettable_worker_handler"):
        WorkerHandler.__init__(self, name)
        self.register_signal(ResettableWorkerHandler.SIGRESET)
        self.register_signal(ResettableWorkerHandler.SIGSTART)
    
    def to_reset(self):
        return self._flag[ResettableWorkerHandler.SIGRESET]
    
    def set_sig(self, sig):
        if self._check_sig(sig):
            self._flag[sig] = True
    
    def to_start(self):
        return self._flag[ResettableWorkerHandler.SIGSTART]

    
class _KeyError(Exception):
    """The duplicate of KeyError with properly formatted error message."""
    pass

class ExceptionWrapper(object):
    """Wraps an exception plus traceback to communicate across threads"""
    def __init__(self, exc_info):
        self.exc_type = exc_info[0]
        self.exc_msg = "".join(traceback.format_exception(*exc_info))

    def get_exc(self):
        if self.exc_type is KeyError:
            return _KeyError(self.exc_msg)
        else:
            return self.exc_type(self.exc_msg)

         
_WORKER_CHECK_ALIVE_INTERVAL = 1.
        
def _worker(queue_in, queue_out, map_fn, ndarray=False, name='worker'):
    """A worker method
        * gets input data from queue_in,
        * computes output using map_fn,
        * feeds output data to queue_out.
    Any exception encountered in the loop will be passed to queue_out.
    The worker will exit only when a SIGTERM is issued or the 
    terminate() function of its process is called.
        
    Parameters
    ----------
    queue_in : Queue or None
        In the case of None, map_fn requires no inputs.
    queue_out : Queue
        This must be provided, otherwise the worker is not productive.
    map_fn : list of callable
        map_fn applies a serial of functions that accepts a single input
        and returns the output corresponding to that input. The function
        can be None which corresponds to a "pass" function.
    ndarray : boolean
        Whether to treat the output data as (dict or list or tuple of)
        np.ndarray. Put np.ndarray data may improve efficiency.
    """
    handler = WorkerHandler()
    data = None
    while handler.is_alive():
        try:
            if queue_in is not None:
                try:
                    data = queue_in.get(_WORKER_CHECK_ALIVE_INTERVAL)
                except Empty:
                    continue
            if isinstance(data, ExceptionWrapper):
                raise data.get_exc()
            for fn in map_fn:
                if fn is not None:
                    data = fn(data)
            if ndarray and np.isscalar(data):
                data = np.array(data)
        except Exception:
            queue_out.put(ExceptionWrapper(sys.exc_info()))
        else:
            queue_out.put(data, ndarray)
            data = None
    if queue_in is not None:
        queue_in.close()
    queue_out.close()
    #print "Worker", name, "exits"
    
def _resettable_worker(queue_in, queue_out, map_fn, ndarray=False, name='resettable_worker'):
    """A resettable worker method
    
    In a resettable worker, map_fn may have some internal states (e.g.
    rand_seed and offset). In some cases, they will be resetted on demand.
    Use Resettable handler to capture SIGRESET and SIGSTART to suspend and
    start the inner loop.
    
    This worker is usuallly used for a finite state sampler. Exactly one
    worker is created and one StopIteration error is passed to the output 
    queue when the sampler runs out.
    
    Parameters
    ----------
    map_fn : callable
        It must have a reset() method.
    """
    handler = ResettableWorkerHandler()
    data = None
    while handler.is_alive():
        try:
            if handler.to_reset():
                if handler.to_start():
                    for fn in map_fn:
                        if fn is not None:
                            fn.reset()
                    handler.reset(ResettableWorkerHandler.SIGRESET)
                    handler.reset(ResettableWorkerHandler.SIGSTART)
                else:
                    time.sleep(_WORKER_CHECK_ALIVE_INTERVAL)
                    continue
            # get input
            if queue_in is not None:
                try:
                    data = queue_in.get(_WORKER_CHECK_ALIVE_INTERVAL)
                except Empty:
                    continue
            if isinstance(data, ExceptionWrapper):
                raise data.get_exc()
            # compute output
            try:
                for fn in map_fn:
                    if fn is not None:
                        data = fn(data)
                if ndarray and np.isscalar(data):
                    data = np.array(data)
            except StopIteration:
                handler.set_sig(ResettableWorkerHandler.SIGRESET)
                raise
        except Exception:
            queue_out.put(ExceptionWrapper(sys.exc_info()))
        else:
            queue_out.put(data, ndarray)
            data = None
    if queue_in is not None:
        queue_in.close()
    queue_out.close()
    #print "Worker", name, "exits"
        
    
class ZMQDataLoader(DataLoader):
    def __init__(self, dataset, sampler_cls, sampler_args=(), transformer=None,
                 name='zmq_data_loader'):
        DataLoader.__init__(self, dataset, sampler_cls, sampler_args, transformer, name)
        self._index_queue_size = 16
        self._data_queue_size = 16
        self._index_queue = None
        self._data_queue = None
        self._index_process = None
        self._data_processes = list()
        self._timeout = 1.
    
    def _get_index_queue(self):
        if self._index_queue is None:
            DataLoader._init(self) # self._sampler is required by self._index_process.
            q = Queue(self._index_queue_size, connect_mode='one2multi') # index as integer
            p = Process(name='/%s/index' % self.name, target=_resettable_worker,
                        args=(None, q, [self._sampler], True))
            self._index_queue = q
            self._index_process = p
            p.start()
        return self._index_queue
    
    def _get_data_queue(self):
        if self._data_queue is None:
            self._data_queue = Queue(self._data_queue_size, connect_mode='multi2one')
            # call "get" method to bind the socket in the main process
            empty_queue(self._data_queue, self._timeout)
        return self._data_queue
        
    def add_prefetch_process(self, n_proc=1, seeds=None):
        """
        Note
        ----
        self._sampler must be initialized before this method is called.
        Thus, the default prefetching process is added in the _init method,
        which will conflict with the purpose of this method.
        """
        iq = self._get_index_queue()
        dq = self._get_data_queue()
        if type(seeds) is not list:
            seeds = [seeds]*n_proc
        prev_n_proc = len(self._data_processes)
        for i in range(n_proc):
            p = Process(name='/%s/data%d' % (self.name, prev_n_proc + i),
                        target=_worker,
                        args=(iq, dq, [self._dataset, self._transformer], True,
                              'worker%d' % (prev_n_proc + i)))
            p.start()
            self._data_processes.append(p)
        self._initialized = True
        
    def _init(self):
        #assert self._data_queue is None or len(self._data_processes) == 0
        self.add_prefetch_process()
    
    def _reset(self):
        DataLoader._reset(self)
        os.kill(self._index_process.pid, ResettableWorkerHandler.SIGRESET)
        empty_queue(self._data_queue, self._timeout)
        os.kill(self._index_process.pid, ResettableWorkerHandler.SIGSTART)
    
    def _get_next(self):
        data = self._data_queue.get()
        if isinstance(data, ExceptionWrapper):
            raise data.exc_type(data.exc_msg)
        return data   
    
    def join_all(self):
        # terminate the data processes
        for dp in self._data_processes:
            dp.terminate()
        if self._data_queue is not None:
            empty_queue(self._data_queue, self._timeout)
            self._data_queue.close()
            self._data_queue.join_thread()
            self._data_queue = None
        for dp in self._data_processes:
            if dp.is_alive():
                dp.join()
            print("Joined", dp.name)
        self._data_processes = list()
        # terminate the index process
        if self._index_process is not None:
            # As _index_process is the only sender associated with _index_queue,
            # it should exist longer than main process. Set reset to temporarily
            # stop putting new index into _index_queue
            os.kill(self._index_process.pid, ResettableWorkerHandler.SIGRESET)
        if self._index_queue is not None:
            empty_queue(self._index_queue, self._timeout)
            self._index_queue.close()
            self._index_queue.join_thread()
            self._index_queue = None
        if self._index_process is not None:
            # Kill the _index_process after closing all getters of _index_queue.
            self._index_process.terminate()
            if self._index_process.is_alive():
                self._index_process.join()
            print("Joined", self._index_process.name)
            self._index_process = None
        self._initialized = False
            
    def __del__(self):
        print("Del", self.name)
        if self._initialized:
            self.join_all()
