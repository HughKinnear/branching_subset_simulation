from abc import ABC, abstractmethod
from .estimate import probability_of_indicator


class Stopper:

    def __init__(self,
                 condition_list,
                 log):
        self.condition_list = condition_list
        self.stop_dict = {}
        self.log = log


    def is_stop(self, leaves):
        result = not bool(self.options(leaves))
        if result:
            self.log.terminate()
        return result


    def options(self, leaves):
        options = []
        for leaf in leaves:
            stop_data = self.stop_dict.get(leaf.name, None)
            if stop_data is None:
                stop_data = {condition.name: condition(leaf) for condition in self.condition_list}
                self.stop_dict[leaf.name] = stop_data
            if not any(stop_data.values()):
                options.append(leaf)
        return options


class Condition(ABC):

    def __init__(self,
                 log):
        self.log = log

    @property
    def name(self):
        return self._name()

    @abstractmethod
    def _name(self):
        pass

    @abstractmethod
    def __call__(self,leaf):
        pass

class Failure(Condition):

    def __init__(self,
                 threshold,
                 probability_limit,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold = threshold
        self.probability_limit = probability_limit

    def _name(self):
        return 'Failure'

    def __call__(self, leaf):
        indicator = leaf.indicator.new_threshold(self.threshold)
        probability = probability_of_indicator(indicator, leaf)
        result = probability > self.probability_limit
        if result:
            self.log.fail_stop(leaf)
        return probability > self.probability_limit



class Constant(Condition):


    def _name(self):
        return 'Constant'

    def __call__(self, leaf):
        result = leaf.sorted_list[0].performance == leaf.sorted_list[-1].performance
        if result:
            self.log.constant_stop(leaf)
        return result


