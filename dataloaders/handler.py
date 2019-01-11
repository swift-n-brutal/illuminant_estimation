import signal

class Handler(object):
    """Mask signals for asychronous events.
    """
    ALL_SIGNALS = [signal.SIGINT, signal.SIGTERM,
                   signal.SIGUSR1, signal.SIGUSR2]
    def __init__(self, name="handler"):
        self.name = name
        self._flag = dict()
        
    def _check_sig(self, sig):
        return sig in self._flag
    
    def register_signal(self, sig):
        assert sig in Handler.ALL_SIGNALS, \
            "Signal is NOT allowed: {}".format(sig)
        signal.signal(sig, self.on_signal)
        self._flag[sig] = False
        
    def on_signal(self, sig, frame):
        self._flag[sig] = True
        
    def reset(self, sig=None):
        if sig is None:
            for s in self._flag.keys():
                self._flag[s] = False
        elif self._check_sig(sig):
            self._flag[sig] = False
        else:
            # unknown signal
            pass
