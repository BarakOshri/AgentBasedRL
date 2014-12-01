#from scipy.optimize import *
import numpy as np
from scipy import stats
from scipy.optimize import fmin_l_bfgs_b


def minf(x):
    return ((x[0]-2)*(x[0]-2) + (x[1]-2)*(x[1]-2))

limits = [(0, 4), (0, 4)]
print fmin_l_bfgs_b(minf, x0=np.array([4, 4]), approx_grad=True, bounds=limits)[0][0]
