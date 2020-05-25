import cProfile, pstats, io


class Profiler(object):
    
    def __init__(self, text, sort='tottime'):
        self.text = text
        self.sortby = sort
        self.profiler = cProfile.Profile()

    def enable(self):
        self.profiler.enable()
        print(u'--------- Enabling Profiler %s ---------------' % self.text)
    
    def disable(self, print_stats = False):
        self.profiler.disable()
        print(u'--------- Disabling Profiler %s ---------------' % self.text)
        if print_stats:
            self.print_stats()

    def print_stats(self, max_stats=10):     
        s = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=s).sort_stats(self.sortby)
        ps.print_stats(max_stats)
        print(s.getvalue())
        print(u'--------- Disabled Profiler %s ---------------' % self.text)
        

def profile(function, *args, **kwargs):
    """ Returns performance statistics (as a string) for the given function.
    """
    def _run():
        function(*args, **kwargs)
    import cProfile as profile
    import pstats
    import os
    import sys; sys.modules['__main__'].__profile_run__ = _run
    id = function.__name__ + '()'
    profile.run('__profile_run__()', id)
    p = pstats.Stats(id)
    p.stream = open(id, 'w')
    p.sort_stats('time').print_stats(20)
    p.stream.close()
    s = open(id).read()
    os.remove(id)
    return s        
