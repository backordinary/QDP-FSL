# https://github.com/TorsteinTenstad/scattering_correction_TFE4580/blob/0e96fdec58beec281dc157812353339203fd7235/testing_scripts/admm_test.py
import time
from typing import List, Optional, Any
import numpy as np
import matplotlib.pyplot as plt

from docplex.mp.model import Model

from qiskit import BasicAer
from qiskit.aqua.algorithms import QAOA, NumPyMinimumEigensolver
from qiskit.optimization.algorithms import CobylaOptimizer, MinimumEigenOptimizer
from qiskit.optimization.problems import QuadraticProgram
from qiskit.optimization.algorithms.admm_optimizer import ADMMParameters, ADMMOptimizer

# If CPLEX is installed, you can uncomment this line to import the CplexOptimizer.
# CPLEX can be used in this tutorial to solve the convex continuous problem,
# but also as a reference to solve the QUBO, or even the full problem.
#
# from qiskit.optimization.algorithms import CplexOptimizer

# define COBYLA optimizer to handle convex continuous problems.
cobyla = CobylaOptimizer()

# define QAOA via the minimum eigen optimizer
qaoa = MinimumEigenOptimizer(QAOA(quantum_instance=BasicAer.get_backend('statevector_simulator')))

# exact QUBO solver as classical benchmark
exact = MinimumEigenOptimizer(NumPyMinimumEigensolver()) # to solve QUBOs

# in case CPLEX is installed it can also be used for the convex problems, the QUBO,
# or as a benchmark for the full problem.
#
# cplex = CplexOptimizer()

# construct model using docplex
mdl = Model('ex6')

v = mdl.binary_var(name='v')
w = mdl.binary_var(name='w')
t = mdl.binary_var(name='t')
u = mdl.continuous_var(name='u')

mdl.minimize(v + w + t + 5 * (u-2)**2)
mdl.add_constraint(v + 2 * w + t + u <= 3, "cons1")
mdl.add_constraint(v + w + t >= 1, "cons2")
mdl.add_constraint(v + w == 1, "cons3")

# load quadratic program from docplex model
qp = QuadraticProgram()
qp.from_docplex(mdl)
print(qp.export_as_lp_string())