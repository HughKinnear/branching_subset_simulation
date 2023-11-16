from anytree import Node
import numpy as np


class LevelBase(Node):

    def __init__(self,
                 indicator,
                 *args,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)
        self.indicator = indicator
        self.sample_list = None


    @property
    def order(self):
        return int(self.name)

    @property
    def sorted_list(self):
        return sorted(self.sample_list, key=lambda x: x.performance)

    @property
    def unique_list(self):
        return [self.sample_list[ind]
                for ind in np.unique([samp.array for samp in self.sample_list],
                                     axis=0,
                                     return_index=True)[1]]
    @property
    def level_size(self):
        return len(self.sample_list)

    @property
    def branch(self):
        return list(self.ancestors) + [self]



class InitialLevel(LevelBase):

    def __init__(self,
                 sample_list,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_list = sample_list



class Level(LevelBase):

    def __init__(self,
                 chain_list,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.chain_list = chain_list
        self.sample_list = [sample
                            for chain in chain_list
                            for sample in chain]



