# Branching Subset Simulation

This is a companion repository for the paper [Branching Subset Simulation](https://arxiv.org/abs/2209.02468). It has two purposes.

The first is reproducibility. All the code for the numerical experiments and figures used in the paper are shared here. The numerical_examples folder contains a data folder where all the data is stored and the notebooks used to create the data. The data file names starting with "bre" and "him" correspond to experiments with Breitung's piecewise linear function and Himmelblaus's function respectively. File names with "sus"and "bss" refer to experiments using Subset Simulation and Branching Subset Simulation respectively. "dim_x" referes the to dimensionality of the problem where x is the number of dimensions. "_l1" or "_l2" refers to sensitivity analysis conducted with a specific penalty function.

The second is to provide an implementation of Branching Subset Simulation that others can use on their own problems. The python implementation is stored in the branching_sus folder. The file branching.py provides the BranchingFramework class which can be used to create a Branching Subset Simulation implementation. This allows the user to define any choose function, stopping conditions, partitioner and markov chain algorithm. Two examples of how to to this are stored in implementation.py. The first is subset simulation, which is just a specific instance of the more general branching framework. The second is a version using the convex graph partitioner.

The examples folder contains jupyter notebooks showing how to actually used the implementations.
