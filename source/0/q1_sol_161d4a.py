# https://github.com/EXPmaster/COMP3366/blob/d1e072e1651d768d707a60bde4deb3b5a2a164ea/assignment4/q1_sol.py
import os
import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit import Parameter
from qiskit.quantum_info.operators import Pauli
from qiskit.utils import QuantumInstance
from qiskit.providers.aer import QasmSimulator
from qiskit.opflow import PauliOp
from qiskit.opflow.gradients import Gradient
from qiskit.algorithms import VQE, NumPyMinimumEigensolver
from qiskit.algorithms.optimizers import SPSA, SLSQP


class VQETrainer:

    def __init__(self, num_qubits, num_layers):
        self.num_qubits = num_qubits
        self.num_layers = num_layers

    def get_circuit(self, barriers=True):
        qc = QuantumCircuit(self.num_qubits)
        params = []
        # initial Euler Rotation Layer
        for i in range(self.num_qubits):
            for _ in range(2):  # two new parameters
                params.append(Parameter(f'p{len(params):02}'))
            # rotation with the two new parameters. Don't need the first
            # z rotation
            qc.u(params[-2], params[-1], 0, i)
        if barriers:
            qc.barrier()
        for l in range(self.num_layers):
            # entangling layer
            for i in range(self.num_qubits - 1):
                qc.cnot(i, i + 1)
            if barriers:
                qc.barrier()
            for i in range(self.num_qubits):
                for _ in range(3):
                    params.append(Parameter(f'p{len(params):02}'))
                qc.u(params[-3], params[-2], params[-1], i)
            if barriers:
                qc.barrier()
        return qc

    def get_hamitonian_ising(self, g=None):
        operators = []
        op_str = 'I' * self.num_qubits
        for i in range(self.num_qubits - 1):
            tmp_op = op_str[:i] + 'ZZ' + op_str[i + 2:]
            operators.append(PauliOp(Pauli(tmp_op), -1.0))
        g = 2
        for i in range(self.num_qubits):
            tmp_op = op_str[:i] + 'X' + op_str[i + 1:]
            operators.append(PauliOp(Pauli(tmp_op), -g))
        hamitonian = sum(operators)
        return hamitonian

    def train(self, circuit, hamitonian, num_iters=1000, num_shots=1, save_path=None):
        # backend = Aer.get_backend('qasm_simulator')
        # qinstance = QuantumInstance(backend=backend,
        #                             shots=num_shots)
        qinstance = QuantumInstance(QasmSimulator(method='matrix_product_state'), shots=num_shots)
        optimizer = SPSA(maxiter=num_iters)
        # optimizer = SLSQP(maxiter=num_iters)
        vqe = VQE(ansatz=circuit,
                  # gradient=Gradient(grad_method='lin_comb'),
                  optimizer=optimizer,
                  quantum_instance=qinstance,
                  include_custom=True
                  )
        result_vqe = vqe.compute_minimum_eigenvalue(operator=hamitonian)
        npme = NumPyMinimumEigensolver()
        result_np = npme.compute_minimum_eigenvalue(operator=hamitonian)
        ref_value = result_np.eigenvalue.real
        print('VQE result: {},\t Calculation result: {}'.format(result_vqe.optimal_value, ref_value))


if __name__ == '__main__':
    trainer = VQETrainer(5, 2)
    circuit = trainer.get_circuit()
    H = trainer.get_hamitonian_ising(1.0)
    trainer.train(circuit, H)
