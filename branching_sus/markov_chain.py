from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import bernoulli
from scipy.stats import norm


class ChainData:

    def __init__(self,
                 chain_list,
                 parameter_names=None):
        self.chain_list = chain_list

        if parameter_names is None:
            self.parameter_names = ['Parameter ' + str(i + 1)
                                    for i in range(self.number_of_params)]
        else:
            self.parameter_names = parameter_names

    @property
    def number_of_params(self):
        return len(self.chain_list[0][0])

    def trim(self, length):
        self.chain_list = [chain[length:] for chain in self.chain_list]


    @property
    def chain_dict(self):
        array_chain_list = [[sample for sample in chain]
                            for chain in self.chain_list]
        parameter_chain_list = [arr.T for arr in np.array(array_chain_list).T]
        chain_dict = {}
        for i in range(self.number_of_params):
            chain_dict[self.parameter_names[i]] = parameter_chain_list[i]
        return chain_dict

    @property
    def all_samples(self):
        return [item for chain in self.chain_list for item in chain]




class MarkovChain(ABC):


    def __init__(self,
                 random_state=None):
        self.random_state = random_state


    @abstractmethod
    def step(self, curr):
        pass

    def update(self, chain_data, length):
        for chain in chain_data.chain_list:
            for _ in range(length):
                chain.append(self.step(chain[-1]))

    def update_each(self, chain_data, length):
        for _ in range(length):
            for chain in chain_data.chain_list:
                chain.append(self.step(chain[-1]))





class LogProportionalMarkov(MarkovChain):

    def __init__(self,
                 log_proportional_target,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.log_proportional_target = log_proportional_target

    def step(self, curr):
        pass


class SubsetSimulationMarkov(MarkovChain):

    def __init__(self,
                 indicator,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.indicator = indicator

    def step(self, curr):
        pass


class Metropolis(LogProportionalMarkov):
    def __init__(self,
                 proposal_sampler,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.proposal_sampler = proposal_sampler

    def step(self, curr):
        return Metropolis.metropolis_step(self.log_proportional_target,
                                          self.proposal_sampler,
                                          curr,
                                          self.random_state)

    @staticmethod
    def metropolis_step(target, proposal_sampler, curr, random_state):
        cand = proposal_sampler(curr)
        log_alpha = target(cand) - target(curr)
        alpha = np.exp(log_alpha)
        if np.isnan(alpha):
            return curr
        if alpha >= 1:
            new = cand
        else:
            if bool(bernoulli.rvs(alpha, size=1, random_state=random_state)):
                new = cand
            else:
                new = curr
        return new



class ModifiedMetropolis(SubsetSimulationMarkov):
    def __init__(self,
                 std_list,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.std_list = std_list
        self.proposal_sampler_list = [lambda x, val=val: np.array([norm.rvs(loc=x,
                                                                          scale=val,
                                                                          random_state=self.random_state)])
                                      for val in self.std_list]

    def step(self, curr):
        cand_list = []
        for i, curr_i in enumerate(curr):
            cand_list.append(Metropolis.metropolis_step(proposal_sampler=self.proposal_sampler_list[i],
                                                        target=lambda x: norm.logpdf(x),
                                                        curr=np.array([curr_i]),
                                                        random_state=self.random_state)[0])
        cand = np.array(cand_list)
        if self.indicator(cand) == 1:
            new = cand
        else:
            new = curr
        return new


