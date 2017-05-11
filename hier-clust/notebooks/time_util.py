from contextlib import contextmanager
from timeit import default_timer
from collections import OrderedDict

class TimingRegistry(object):
    def __init__(self, verbose = True):
        self.verbose = verbose
        self.reset()

    @contextmanager
    def timer(self, name, **kwargs):
        start = default_timer()
        def get_elapsed_time():
            return default_timer() - start
        yield get_elapsed_time
        elapsed = get_elapsed_time()
        if self.verbose:
            if name is not None:
                opt_stage = " ({})".format(name)
            else:
                opt_stage = ""
            print("Elapsed time: {:.3f} seconds{}".format(elapsed, opt_stage))
        if name is not None:
            self.registry[name] = elapsed

    def reset(self):
        self.registry = OrderedDict()

    def __str__(self):
        return "\n".join("{}: {}".format(k, v) for k, v in self.registry.items())

    def __repr__(self):
        return self.__str__()
