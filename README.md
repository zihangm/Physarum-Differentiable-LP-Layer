# Physarum-Differentiable-LP-Layer
Physarum Powered Differentiable Linear Programming Layers

Usage:
```python
from physarum_solver import physarum_solve

# construct the LP problem in standard form with A, b, c, which can be from either constant or output of some deep network
# then solve the LP problem by calling the solver
x_sol = physarum_solve(A, b, c, step_size=0.5, max_iter=10)
# x_sol is the solution to this LP and can be used in desired further processing
```

Use our solver in the experiments reported in our paper:


