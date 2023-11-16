import numpy as np
from anytree import PreOrderIter, RenderTree
from matplotlib import pyplot as plt
import branching_sus.estimate as est
from .utils import plotting_range, indicators_to_classifier, contour_plot


def all_levels(bss):
    for level in PreOrderIter(bss.initial_level):
        plotter = np.array([sample.array for sample in level.sample_list]).T
        plt.scatter(plotter[0], plotter[1])

def ccdf(bss, thresh, num_points):
    smallest_performance = bss.initial_level.sorted_list[0].performance
    perfs = np.linspace(smallest_performance, thresh, num_points)
    log_probs = np.array([np.log(est.exceedance_probability(bss, perf)) for perf in perfs])
    plt.plot(perfs, log_probs)
    plt.xlabel('Performance')
    plt.ylabel('Log Probability')
    plt.title('Complementary Cumulative Distribution Function')


def branch_ccdf(leaf, thresh, num_points):
    smallest_performance = leaf.branch[0].sorted_list[0].performance
    perfs = np.linspace(smallest_performance, thresh, num_points)
    log_probs = np.array([np.log(est.estimate_leaf_probability(perf, leaf)) for perf in perfs])
    plt.plot(perfs, log_probs)
    plt.xlabel('Performance')
    plt.ylabel('Log Probability')
    plt.title('Complementary Cumulative Distribution Function')



def render_tree(bss):
    print(RenderTree(bss.initial_level).by_attr())


def partition(bss,step):
    leaves = bss.leaves
    all_samples = [samp.array for samp in bss.all_samples]
    x_range, y_range = plotting_range(all_samples)
    indis = [leaf.indicator.partition_indicator
             for leaf in leaves]
    classifier = indicators_to_classifier(indis)
    labels = [i for i in range(len(leaves))]
    levels = np.arange(1, len(labels) + 1) - 0.5
    contour_plot(x_range,
                 y_range,
                 step,
                 classifier,
                 levels=levels)






