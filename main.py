from DUC_functions import *
import math
import numpy as np

# ===> Elements <===
NoE = 510  # Number of maximum elements

# ===> Beam properties <===
YoungModulus = 3 * 10 ** 9  # The Young's Modulus
Poisson = 0.058201058  # The Poisson ratio
ShearModulus = YoungModulus / 2 / (1 + Poisson)
SCpr = 5 / 8  # The shear correction parameter

# ===> Beam shape <==
L = 10  # The beam length
b = 0.1 * L  # The beam width
h = 0.1 * L  # The beam height

# ==> Loading parameter <==
# Point load:
# 0 - Concentrated load;
# 1 - Concentrated moment
PointLoad = np.array([[0], [0], [0], [0], [0], [0]])
# Distributed load:
# 0 - Distributed vertical load;
# 1 - Distributed moment
DistributedLoad = np.array([-5 * 10 ** 4, 0])

# Boundary conditions
# 0 - The First point
# 1 - The end point
FixedNodes = np.array([[0, 0, 0],
                       [0, 1, 0],
                       [-1, 0, 1],
                       [-1, 1, 1]])

Inertia = b * h ** 3 / 12  # Moment of inertia
Area = b * h

t1 = Beam_EulerBernoulli(YoungModulus, L, Inertia, PointLoad, DistributedLoad, FixedNodes, NoE)
t2 = Beam_Timoshenko(YoungModulus, L, Inertia, ShearModulus, SCpr, Area, PointLoad, DistributedLoad, FixedNodes, NoE)
print("Euler beam: = ", t1)
print("Timoshenko beam = ", t2)
