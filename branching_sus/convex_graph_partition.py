import networkx as nx
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from .partition import Partitioner, PartitionInformation
import numpy as np
from branching_sus.utils import seeded_choice_no_replace


class BarebonesPipeline(Pipeline):

    def predict_single(self, x):
        for _, transformer in self.steps[:-1]:
            x = transformer.transform_single(x)
        return self.steps[-1][1].predict_single(x)


class BarebonesStandardScaler(StandardScaler):

    def transform_single(self, x):


        return (x - self.mean_) / self.var_ ** .5


class BarebonesLinearSVC(LinearSVC):

    def predict_single(self, x):


        scores = np.dot(self.coef_, x) + self.intercept_


        if len(scores) == 1:
            return self.classes_[int(scores > 0)]
        else:
            return self.classes_[np.argmax(scores)]




class ConvexGraphPartitionInformation(PartitionInformation):

    def __init__(self,
                 indicator_list,
                 graph,
                 graph_samples,
                 classifier,
                 model,
                 partition,
                 sample_partition):
        self.indicator_list = indicator_list
        self.graph = graph
        self.graph_samples = graph_samples
        self.classifier = classifier
        self.partition = partition
        self.sample_partition = sample_partition
        self.model = model

    def get_indicators(self):
        return self.indicator_list


class ConvexGraphPartitioner(Partitioner):

    def __init__(self,
                 budget,
                 allocator,
                 params,
                 random_state,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.random_state = random_state
        self.allocator = allocator
        self.allocator.add_budget('convex budget', budget)
        if params is None:
            self.params = {}
        else:
            self.params = params


    def partition(self, level):
        budget = self.allocator.budget_dict[level.name]['convex budget']
        graph_size = int(((1 + np.sqrt(1 + 8 * budget)) / 2))
        if graph_size > len(level.unique_list):
            graph_size = len(level.unique_list)
        if graph_size < 2:
            return self.no_partition_info

        vectorised_performance_function = np.vectorize(level.indicator.performance_function,
                                                       signature='(n)->()')
        graph_samples = seeded_choice_no_replace(level.unique_list,
                                                 graph_size,
                                                 random_state=self.random_state)
        graph_arrays = np.array([samp.array for samp in graph_samples])
        graph_performances = np.array([samp.performance for samp in graph_samples])
        combs = np.array([(i, j) for i in range(len(graph_arrays))
                          for j in range(len(graph_arrays)) if i < j])
        points = np.linspace(graph_arrays[combs.T[0]],
                             graph_arrays[combs.T[1]],
                             2,
                             endpoint=False)[1:, :, :]
        minimum_convex = np.min(vectorised_performance_function(points), axis=0)
        endpoint_minimums = np.minimum(graph_performances[combs.T[0]],
                                       graph_performances[combs.T[1]])
        adj_info = minimum_convex >= endpoint_minimums
        adj_matrix = np.zeros(shape=(len(graph_arrays), len(graph_arrays)))
        for comb, entry in zip(combs, adj_info):
            adj_matrix[comb[0], comb[1]] = int(entry)
            adj_matrix[comb[1], comb[0]] = int(entry)
        G = nx.from_numpy_array(adj_matrix)
        partition = nx.algorithms.community.asyn_lpa_communities(G, seed=self.random_state)
        sample_partition = [[graph_samples[ind] for ind in part] for part in partition]

        if len(sample_partition) == 1:
            indicator_list = [lambda x: 1]
            classifier = lambda x: 0
            model = None
        else:
            data = np.array([samp.array for part in sample_partition for samp in part])
            labels = np.array([i for i, part_set in enumerate(sample_partition) for _ in range(len(part_set))])

            self.params['random_state'] = self.random_state.integers(100)
            model = BarebonesPipeline([('scaler', BarebonesStandardScaler()),
                                       ('svc', BarebonesLinearSVC(**self.params))])

            model.fit(data, labels)
            var = model.steps[0][1].var_
            var[var == 0] = 1

            classifier = model.predict_single


            indicator_list = [self.indicator_factory(classifier,
                                                     label)
                              for label in range(len(sample_partition))]
        return ConvexGraphPartitionInformation(indicator_list=indicator_list,
                                               graph=G,
                                               graph_samples=graph_samples,
                                               classifier=classifier,
                                               partition=partition,
                                               model=model,
                                               sample_partition=sample_partition)



    @staticmethod
    def indicator_factory(classifier, label):
        return lambda x: int(classifier(x) == label)

    @property
    def no_partition_info(self):
        return ConvexGraphPartitionInformation(indicator_list=[lambda x: 1],
                                               graph=None,
                                               graph_samples=None,
                                               classifier=None,
                                               model=None,
                                               partition=None,
                                               sample_partition=None)
