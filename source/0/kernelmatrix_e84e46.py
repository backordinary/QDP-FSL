# https://github.com/Qiskit-Partners/qiskit-dell-runtime/blob/a1e5df1086fa56ac80fb7d00e7a99625728a3c05/examples/programs/qkad/qtils/kernelmatrix.py
import itertools
import json
import numpy as np
from numpy.random import RandomState
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.compiler import transpile
from cvxopt import matrix, solvers  # pylint: disable=import-error

