# Tree-Parametric-Algorithm

This repository contains the code of the algorithm in the paper "A Parametric Approach for Solving Quadratic Optimization with Indicators Over Trees".


Use line below to import the function:
```
from Parametric import Para_Algo
```

The function *Para_Algo* requires 4 arguments.

*Q*: n x n dimention numpy array. This is the quadratic term in the objective function.

*c*: n dimention numpy array. This is the linear term in the objective function.

*lam*: n dimention numpy array. This is the regularizer $\lambda$ in the objective function.

*M*: *(Optional. The default value is 5000.)* A Float. This is the bounds on variable x. 
