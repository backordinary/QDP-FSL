# https://github.com/magn5452/QiskitQaoa/blob/183d21ab167c8ed58cf29f7f80eca78c0d822034/VehicleRouting/framework/interfaces/MinimumEigenSolver.py
from abc import abstractmethod, ABC

from qiskit import Aer
from qiskit.algorithms import NumPyMinimumEigensolver, QAOA
from qiskit.algorithms.optimizers import optimizer, COBYLA
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.problems import quadratic_program


class MinimumEigenSolver(ABC):
    @abstractmethod
    def solve(self, quadratic_program):
        pass


