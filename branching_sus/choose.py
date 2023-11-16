from abc import ABC, abstractmethod


class Chooser(ABC):

    @abstractmethod
    def choose(self, options):
        pass


class First(Chooser):

    def choose(self, options):
        return options[0]
