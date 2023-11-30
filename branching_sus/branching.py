from anytree import findall_by_attr, PreOrderIter
from .create import Creator
from .performance_function import PerformanceFunction

#test
#test 2
#another test

class BranchingFramework:

    def __init__(self,
                 performance_function,
                 level_probability,
                 level_size,
                 dimension,
                 stopper,
                 partitioner,
                 markov_chain,
                 chooser,
                 allocator,
                 random_state,
                 log,
                 ):
        self.stopper = stopper
        self.chooser = chooser
        self.partitioner = partitioner
        self.creator = Creator(random_state=random_state,
                               performance_function=PerformanceFunction(performance_function),
                               level_probability=level_probability,
                               markov_chain=markov_chain,
                               level_size=level_size,
                               log=log,
                               allocator=allocator,
                               dimension=dimension)
        self.initial_level = None


    def run(self):
        self.initial_level = self.creator.create_initial()
        while not self.stopper.is_stop(self.leaves):
            current_level = self.chooser.choose(self.stopper.options(self.leaves))
            self.partitioner.fit(current_level)
            indicators = self.partitioner.get_indicators_for_level(current_level)
            self.creator.create(indicators, current_level)


    def find(self, name):
        return findall_by_attr(self.initial_level, name)[0]


    @property
    def leaves(self):
        return self.initial_level.leaves

    @property
    def all_levels(self):
        all_levels = list(PreOrderIter(self.initial_level))
        return sorted(all_levels, key=lambda x: x.order)

    @property
    def all_samples(self):
        return [samp for level in self.all_levels
                for samp in level.sample_list]

