# https://github.com/qudoor/qutrunk/blob/6d4b564c18d7a79ca6edcd2ad0ced1c1e385fe9a/qutrunk/test/gate/local/test_h_gate.py
import pytest
import numpy as np

from qiskit import QuantumCircuit, BasicAer, transpile

from qutrunk.circuit.gates import H
from qutrunk.circuit import QCircuit
from qutrunk.circuit.gates import Matrix
from qutrunk.test.gate.local.backend_fixture import backend, backend_type

class TestHGate:
    @pytest.fixture
    def result_gate(self, backend):
        circuit = QCircuit(backend=backend)
        qr = circuit.allocate(1)
        H * qr[0]
        result_gate = np.array(circuit.get_statevector()).reshape(-1, 1)
        return result_gate

    def test_result_matrix(self, result_gate):
        circuit = QCircuit()
        qr = circuit.allocate(1)
        Matrix(H.matrix.tolist()) * qr[0]
        result_matrix = np.array(circuit.get_statevector()).reshape(-1, 1)
        assert np.allclose(result_gate, result_matrix)

    def test_result_qiskit(self, result_gate):
        qc = QuantumCircuit(1, 1)
        backend = BasicAer.get_backend('statevector_simulator')
        qc.h(0)
        job = backend.run(transpile(qc, backend))
        result_qiskit = np.array(job.result().get_statevector(qc)).reshape(-1, 1)
        assert np.allclose(result_gate, result_qiskit)

    def test_gate_inverse(self, backend):
        circuit = QCircuit(backend=backend)
        qr = circuit.allocate(1)
        result_src = np.array(circuit.get_statevector()).reshape(-1, 1)
        H * qr[0]
        H.inv() * qr[0]
        result_des = np.array(circuit.get_statevector()).reshape(-1, 1)
        assert np.allclose(result_src, result_des)

    def test_matrix_inverse(self, backend):
        circuit = QCircuit(backend=backend)
        qr = circuit.allocate(1)
        result_src = np.array(circuit.get_statevector()).reshape(-1, 1)
        Matrix(H.matrix.tolist()) * qr[0]
        Matrix(H.matrix.tolist()).inv() * qr[0]
        result_des = np.array(circuit.get_statevector()).reshape(-1, 1)
        assert np.allclose(result_src, result_des)