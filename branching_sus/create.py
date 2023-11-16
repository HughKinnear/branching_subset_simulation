from .indicator import Indicator
from scipy.stats import multivariate_normal
import numpy as np
from dataclasses import dataclass
from .level import InitialLevel, Level
from .markov_chain import ChainData


@dataclass
class Sample:
    array: ...
    performance: ...


class Creator:

    def __init__(self,
                 random_state,
                 performance_function,
                 level_probability,
                 markov_chain,
                 level_size,
                 dimension,
                 allocator,
                 log):
        self.random_state = random_state
        self.performance_function = performance_function
        self.level_probability = level_probability
        self.markov_chain = markov_chain
        self.level_size = level_size
        self.log = log
        self.dimension = dimension
        self.allocator = allocator
        self.allocator.add_budget('chains', self.number_of_chains)



    @property
    def number_of_chains(self):
        return int(self.level_probability * self.level_size)

    @property
    def chain_length(self):
        return int(self.level_size / self.number_of_chains)


    def create_initial(self):
        array = multivariate_normal.rvs(mean=np.zeros(self.dimension),
                                        cov=np.identity(self.dimension),
                                        size=self.level_size,
                                        random_state=self.random_state)
        sample_list = [Sample(array=arrayi,
                              performance=self.performance_function(arrayi))
                       for arrayi in array]
        indicator = Indicator(-np.inf,
                              self.performance_function,
                              lambda x: 1)
        level = InitialLevel(sample_list=sample_list,
                             indicator=indicator,
                             name='1')
        self.log.create_level(level)
        return level

    def create(self, indicators, level):

        self.log.partition(len(indicators))
        keys = self.allocator.fit(level, indicators)
        chain_number_list = [self.allocator.budget_dict[key]['chains']
                             for key in keys]
        allocation_info = [self.allocator.allocation_info_dict[key]
                           for key in keys]

        for info, key, chain_number in zip(allocation_info,
                                           keys,
                                           chain_number_list):

            if chain_number == 0:
                continue

            sorted_samples = sorted(info.samples,
                                    key=lambda x: x.performance,
                                    )
            seed_samples = sorted_samples[-chain_number:]
            threshold = seed_samples[0].performance
            seeds = [sample.array for sample in seed_samples]
            new_indicator = info.partition_indicator.new_threshold(threshold)
            chain_data = ChainData([[seed] for seed in seeds])
            self.markov_chain.indicator = new_indicator
            self.markov_chain.update(chain_data,
                                     self.chain_length - 1,
                                     )

            sample_chain_list = [[Sample(array=array,
                                         performance=self.performance_function(array))
                                  for array in chain]
                                 for chain in chain_data.chain_list]
            new_level = Level(
                              indicator=new_indicator,
                              chain_list=sample_chain_list,
                              name=key,
                              parent=level)
            self.log.create_level(new_level)







