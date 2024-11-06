# https://github.com/bjader/quantum-neural-network/blob/74a6ec99a3f0df3b89833ad6cd3502c6b0476c24/qnn/ansatz/null_ansatz.py
from qiskit import QuantumCircuit, QuantumRegister

from ansatz.variational_ansatz import VariationalAnsatz


class NullAnsatz(VariationalAnsatz):
    def add_rotations(self, n_data_qubits):
        pass

    def add_entangling_gates(self, n_data_qubits):
        pass

    def get_quantum_circuit(self, n_data_qubits):
        self.qr = QuantumRegister(n_data_qubits, name='qr')
        self.qc = QuantumCircuit(self.qr, name='Shifted circ')
        return self.qc
