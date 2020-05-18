import cProfile, pstats, StringIO


class Profiler(object):
    
    def __init__(self, text, sort='tottime'):
        self.text = text
        self.sortby = sort
        print(u'--------- Enabling Profiler %s ---------------' % self.text)
        self.profiler = cProfile.Profile()
        self.profiler.enable()
    
    def disable(self, max_stats=10):
        self.profiler.disable()
        print(u'--------- Disabling Profiler %s ---------------' % self.text)
        s = StringIO.StringIO()
        ps = pstats.Stats(self.profiler, stream=s).sort_stats(self.sortby)
        ps.print_stats(max_stats)
        print s.getvalue()
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
