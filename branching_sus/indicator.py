class Indicator:

    def __init__(self,
                 threshold,
                 performance_function,
                 partition_indicator,
                 ):
        self.threshold = threshold
        self.partition_indicator = partition_indicator
        self.performance_function = performance_function


    def threshold_indicator(self, x):
        return int(self.performance_function(x)
                   >= self.threshold)


    def indicator(self, x):
        if self.partition_indicator(x) == 1:
            if self.threshold_indicator(x) == 1:
                return 1
        return 0


    def __call__(self, x):
        if isinstance(x, list):
            return [self.indicator(sample.array) for sample in x]
        else:
            return self.indicator(x)


    @staticmethod
    def combine_indicators(indicator_a, indicator_b):
        def combined_indicator(x):
            return indicator_a(x) * indicator_b(x)

        return combined_indicator


    def new_threshold(self, threshold):
        return Indicator(threshold=threshold,
                         performance_function=self.performance_function,
                         partition_indicator=self.partition_indicator)

    def new_partition(self, partition_indicator):

        comb_part_indicator = self.combine_indicators(self.partition_indicator,
                                                     partition_indicator)
        return Indicator(threshold=self.threshold,
                         performance_function=self.performance_function,
                         partition_indicator=comb_part_indicator)

