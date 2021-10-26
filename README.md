# Physarum-Differentiable-LP-Layer
This repository contains the reference code for our paper [Physarum Powered Differentiable Linear Programming Layers and Applications](https://ojs.aaai.org/index.php/AAAI/article/view/17081) (AAAI-2021)

## Requirements
* Python 3
* PyTorch 1.1+

## Use our physarum solver to solve a given LP differentiably:
```python
from physarum_solver import physarum_solve

# construct the LP problem in standard form with A, b, c, which can be from either constant or output of some deep network
# then solve the LP problem by calling the solver
x_sol = physarum_solve(A, b, c, step_size=0.5, max_iter=10)
# x_sol is the solution to this LP and can be used in desired further processing
```

## Use our physarum solver in video object segmentation ([DMM_Net](https://github.com/ZENGXH/DMM_Net)):
```python
cd ./DMM_Net
```
Follow the instructions in ./DMM_Net to prepare the data and do the training/testing.
 
We replaced the original solver from DMM_Net which is customized for matching problem with our physarum solver which works for general LPs, including the matching problem as a special case. 

The replacement happens in **DMM_Net/dmm/modules/submodules/relax_match.py**)


To test the performance of our solver on randomly constructed LP problems and compare with the solver from DMM_Net:
```python
cd DMM_Net/dmm/modules/submodules/
python relax_match.py
```

## Use our physarum solver in few-shot learning ([MetaOptNet](https://github.com/zihangm/MetaOptNet)):
```python
cd ./MetaOptNet
```
Follow the instructions in ./MetaOptNet to prepare the data and do the training/testing.

We replaced the original L-2 SVM solved using Optnet with our L-1 SVM solved using our physarum solver. 

The L-1 SVM and our physarum solver are used in **./MetaOptNet/models/classification_heads_pairwise_physarum.py**



