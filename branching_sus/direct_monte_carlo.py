import numpy as np
from numpy.random import default_rng
from scipy.stats import multivariate_normal


class DirectMonteCarlo:

    def __init__(self,
                 dim,
                 performance_function,
                 sample_size,
                 seed):
        self.dim = dim
        self.performance_function = performance_function
        self.sample_size = sample_size
        self.seed = seed
        self.random_state = default_rng(seed)
        self.samples = None
        self.performances = None


    def run(self):
        self.samples = multivariate_normal.rvs(mean=np.zeros(self.dim),
                                               random_state=self.random_state,
                                               size=self.sample_size)
        self.performances = np.array([self.performance_function(sample)
                                      for sample in self.samples])

    def compute_exceedance_probability(self, threshold):
        return sum(self.performances > threshold) / len(self.samples)


