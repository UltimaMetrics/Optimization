# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 20:45:16 2022

@author: sigma
"""

import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
%matplotlib inline

#Investment problem
# Construct parameters
rate = 1.15

# Objective function parameters
c_ex2 = np.array([1.91*5, 0, 0, 1.8, 1.62])

# Inequality constraints
A_ex2 = np.array([[1,  1,  0,  0,  0],
                  [1, -rate, 1, 0, 1],
                  [1, 0, -rate, 1, 0]])
b_ex2 = np.array([100000, 0, 0])

# Bounds on decision variables
bounds_ex2 = [(  0,    None),
              (-25000, None),
              (-25000, None),
              (-25000, None),
              (  0,   80000)]

# Solve the problem
res_ex2 = linprog(-c_ex2, A_eq=A_ex2, b_eq=b_ex2,
                  bounds=bounds_ex2, method='revised simplex')

res_ex2