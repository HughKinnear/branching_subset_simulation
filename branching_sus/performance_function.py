from functools import lru_cache



class PerformanceFunction:

    def __init__(self, performance_function):
        self.non_cache_performance_function = performance_function
        self.cached_performance_function = lru_cache(maxsize=None)(performance_function)

    def __call__(self, x):
        return self.cached_performance_function(tuple(x))

    @property
    def eval_count(self):
        return self.cached_performance_function.cache_info().currsize


def breitung(x):
    if x[0] > 3.5:
        g_1 = 4 - x[0]
    else:
        g_1 = 0.85 - (0.1 * x[0])
    if x[1] > 2:
        g_2 = 0.5 - (0.1 * x[1])
    else:
        g_2 = 2.3 - x[1]
    return -min([g_1, g_2])


def himmel(x):
    return -((x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2)
