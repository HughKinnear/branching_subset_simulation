from dataclasses import dataclass
from anytree import PreOrderIter
import numpy as np


@dataclass
class AllocationInfo:
    percent: ...
    samples: ...
    partition_indicator: ...


class Allocator:

    def __init__(self):
        self.allocation_info_dict = {}
        self.budget_dict = {'1': {}}

    def add_budget(self, name, budget):
        self.budget_dict['1'][name] = budget

    @property
    def budget_names(self):
        return list(self.budget_dict['1'].keys())

    def fit(self, level, indicators):
        partition_indicator_list = []
        indicator_sample_list = []
        for indicator in indicators:
            partition_indicator = level.indicator.new_partition(indicator)
            samples = [sample for sample, v
                       in zip(level.sample_list,
                              partition_indicator(level.sample_list))
                       if bool(v)]
            partition_indicator_list.append(partition_indicator)
            indicator_sample_list.append(samples)
        percentages = [len(samples) / level.level_size
                       for samples in indicator_sample_list]
        keys = [str(max([level.order
                         for level in list(PreOrderIter(level.root))]) + 1 + i)
                for i in range(len(indicators))]
        allocated_budgets = []
        level_budgets = self.budget_dict[level.name]
        for budget in level_budgets.values():
            allocated_budgets.append(self.budget_divider(percentages, budget))
        trans_allocated_budgets = list(zip(*allocated_budgets))
        for key, budgets in zip(keys, trans_allocated_budgets):
            self.budget_dict[key] = {budget_name: budget
                                     for budget_name, budget
                                     in zip(self.budget_names, budgets)}
        for partition_indicator, samples, percent, key in zip(partition_indicator_list,
                                                              indicator_sample_list,
                                                              percentages,
                                                              keys):
            allocation_info = AllocationInfo(percent=percent,
                                             samples=samples,
                                             partition_indicator=partition_indicator)
            self.allocation_info_dict[key] = allocation_info

        return keys

    @staticmethod
    def budget_divider(percentages, budget):
        float_results = [budget * per for per in percentages]
        results = [int(float_result) for float_result in float_results]
        remain = budget - sum(results)
        while remain > 0:
            discrep = [float_result - result
                       for float_result, result
                       in zip(float_results, results)]
            results[np.argmax(discrep)] += 1
            remain -= 1
        return results
