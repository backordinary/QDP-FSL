# https://github.com/qiskit-community/qiskit-qcgpu-provider/blob/5386d3c9110d7e71f4c97d756c21724f3f417498/tests/test_statevector_simulator.py
import unittest
import math

from qiskit_qcgpu_provider import QCGPUProvider
from qiskit import execute, QuantumRegister, QuantumCircuit, BasicAer
from qiskit.quantum_info import state_fidelity

from .case import MyTestCase


class TestStatevectorSimulator(MyTestCase):
    """Test the state vector simulator"""

    def test_computations(self):
        circ = self.random_circuit(2, 5)
        self._compare_outcomes(circ)
        circ = self.random_circuit(3, 5)
        self._compare_outcomes(circ)
        circ = self.random_circuit(4, 5)
        self._compare_outcomes(circ)
        circ = self.random_circuit(5, 5)
        self._compare_outcomes(circ)
        circ = self.random_circuit(6, 5)
        self._compare_outcomes(circ)
        circ = self.random_circuit(7, 5)
        self._compare_outcomes(circ)
        circ = self.random_circuit(8, 5)
        self._compare_outcomes(circ)
        circ = self.random_circuit(9, 5)
        self._compare_outcomes(circ)


    def _compare_outcomes(self, circ):
        Provider = QCGPUProvider()
        backend_qcgpu = Provider.get_backend('statevector_simulator')
        statevector_qcgpu = execute(circ, backend_qcgpu).result().get_statevector()

        backend_qiskit = BasicAer.get_backend('statevector_simulator')
        statevector_qiskit = execute(circ, backend_qiskit).result().get_statevector()

        self.assertAlmostEqual(state_fidelity(statevector_qcgpu, statevector_qiskit), 1, 5)


if __name__ == '__main__':
    unittest.main()
