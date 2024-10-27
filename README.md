# CATCH
The code for "CATCH: Causal Attention Enhanced Robust Hyperbolic Heterogeneous Graph Representation Learning" (under review)
## Requirements
This repository has been tested with the following packages:
+ Python == 3.7.13
+ PyTorch == 1.12.1+cu113
+ DGL == 0.9.1
## Important Hyper-parameters
+ `curvature`: the curvature of the hyperbolic space.
+ `causality_lambda`: the balancing coefficient of the causality-related term.
+ `lr`:the learning rate.
+ `weight_decay`: the value of weight-decay.
+ `aggre_type`: the type of the information aggregation.
## How to run
For example:
run `multi.py`
