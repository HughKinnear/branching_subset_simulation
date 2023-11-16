from abc import ABC, abstractmethod

class PartitionInformation(ABC):

    @abstractmethod
    def get_indicators(self):
        pass


class Partitioner(ABC):

    def __init__(self):
        self.partition_information_dict = {}

    @abstractmethod
    def partition(self, level):
        pass

    def fit(self, level):
        self.partition_information_dict[level.name] = self.partition(level)


    def get_indicators_for_level(self, level):
        return self.get_partition_information_for_level(level).get_indicators()


    def get_partition_information_for_level(self, level):
        return self.partition_information_dict[level.name]


class NoPartitionInformation(PartitionInformation):

    def get_indicators(self):
        return [lambda x: 1]

class NoPartitioner(Partitioner):

    def partition(self, level):
        return NoPartitionInformation()



