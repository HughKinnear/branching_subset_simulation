from .allocation import Allocator
from .branching import BranchingFramework
from numpy.random import default_rng
from .choose import First
from .convex_graph_partition import ConvexGraphPartitioner
from .log import Log
from .markov_chain import ModifiedMetropolis
from .partition import NoPartitioner
from .stop import Failure, Constant, Stopper


class SubsetSimulation(BranchingFramework):

    def __init__(self,
                 performance_function,
                 dimension,
                 level_size,
                 threshold,
                 level_probability,
                 seed,
                 verbose
                 ):
        log = Log(verbose)
        random_state = default_rng(seed)
        condition_list = [Failure(threshold=threshold,
                                  probability_limit=level_probability,
                                  log=log),
                          Constant(log=log)]
        stopper = Stopper(condition_list, log=log)
        allocator = Allocator()
        std_list = [1 for _ in range(dimension)]
        markov_chain = ModifiedMetropolis(std_list=std_list,
                                          random_state=random_state,
                                          indicator=None)
        chooser = First()
        partitioner = NoPartitioner()

        super().__init__(performance_function=performance_function,
                         dimension=dimension,
                         level_size=level_size,
                         level_probability=level_probability,
                         stopper=stopper,
                         partitioner=partitioner,
                         markov_chain=markov_chain,
                         chooser=chooser,
                         allocator=allocator,
                         random_state=random_state,
                         log=log
                         )


class ConvexGraphBranching(BranchingFramework):

    def __init__(self,
                 performance_function,
                 dimension,
                 level_size,
                 threshold,
                 level_probability,
                 seed,
                 convex_budget,
                 params,
                 verbose
                 ):
        log = Log(verbose)
        random_state = default_rng(seed)
        condition_list = [Failure(threshold=threshold,
                                  probability_limit=level_probability,
                                  log=log),
                          Constant(log=log)]
        stopper = Stopper(condition_list, log)
        allocator = Allocator()
        std_list = [1 for _ in range(dimension)]
        markov_chain = ModifiedMetropolis(std_list=std_list,
                                          random_state=random_state,
                                          indicator=None)
        chooser = First()
        partitioner = ConvexGraphPartitioner(budget=convex_budget,
                                             allocator=allocator,
                                             params=params,
                                             random_state=random_state)

        super().__init__(performance_function=performance_function,
                         dimension=dimension,
                         level_size=level_size,
                         level_probability=level_probability,
                         stopper=stopper,
                         partitioner=partitioner,
                         markov_chain=markov_chain,
                         chooser=chooser,
                         allocator=allocator,
                         random_state=random_state,
                         log=log
                         )

