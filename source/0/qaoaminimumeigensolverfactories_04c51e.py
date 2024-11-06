# https://github.com/magn5452/QiskitQaoa/blob/c1ee63b4fc3bad2e2e31814386db6e564ae5aaf4/VehicleRouting/standard/factories/QaoaMinimumEigenSolverFactories.py
from qiskit import Aer
from qiskit.algorithms import QAOA
from qiskit.algorithms.optimizers import COBYLA
from qiskit.providers.aer import QasmSimulator, StatevectorSimulator

from VehicleRouting.framework.factory.QaoaMinimumEigenSolverFactory import QAOAMinimumEigenSolverFactory


class StandardQaoaMinimumEigenSolverFactory(QAOAMinimumEigenSolverFactory):
    def create_qaoa(self):
        precision = 12
        classical_optimization_method = COBYLA()
        #backend = StatevectorSimulator(precision='single')
        backend = QasmSimulator()
        return QAOA(optimizer=classical_optimization_method, reps=precision, quantum_instance=backend)
