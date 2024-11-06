# https://github.com/dreaming-panda/quantum-ml/blob/109436d1aea5e15c856b7400cefa4e4d43723098/qoptimizer.py
from qiskit import QuantumCircuit
import torch
import numpy as np
from qiskit.quantum_info.operators import Operator
from qiskit import IBMQ, Aer, assemble, transpile
from qram import QRAMDataSet
from qmodel import QModel
from qiskit.circuit.library import Diagonal
class QOptimizer:
    def __init__(self, qc : QuantumCircuit, dataset_qubits:list, output, data : QRAMDataSet, model : QModel, allqubits : list) -> None:
        self.qc = qc
        self.data = data
        self.model = model
        self.mark = dataset_qubits[-1]
        self.label = dataset_qubits[-2]
        self.output = output
        self.oracle_matrix = torch.zeros((8, 8))
        self.diffusion_diagonal = []
        self.allqubits = allqubits
        self.oracle_op : Operator = None
        self.diffusion_op : Diagonal = None
        self.build_op()
    def build_op(self):
        self.oracle_matrix[0][0] = 1
        self.oracle_matrix[1][1] = -1
        self.oracle_matrix[2][2] = 1
        self.oracle_matrix[3][3] = 1
        self.oracle_matrix[4][4] = 1
        self.oracle_matrix[5][5] = 1
        self.oracle_matrix[6][6] = 1
        self.oracle_matrix[7][7] = -1
        self.oracle_op = Operator(self.oracle_matrix.numpy())
        # self.diffusion_diagonal = [-1 for _ in range(int(2**len(self.allqubits)))]
        # self.diffusion_diagonal[0] = 1
        # self.diffusion_op = Diagonal(self.diffusion_diagonal)
    def oracle(self):
        self.qc.barrier()
        self.qc.unitary(self.oracle_op, [self.mark, self.label, self.output],label='oracle')
        self.qc.barrier()
    def diffusion(self):
        self.qc.barrier()
        self.model.de_forward()
        self.data.decode()
        for q in self.allqubits:
            self.qc.x(q)
        self.qc.h(self.allqubits[0])
        self.qc.mcx(control_qubits=self.allqubits[1:], target_qubit=self.allqubits[0])
        self.qc.h(self.allqubits[0])
        for q in self.allqubits:
            self.qc.x(q)
        self.data.encode()
        self.model.forward()
        self.qc.barrier()
    def optimize(self, iter):
        for _ in range(iter):
            self.oracle()
            self.diffusion()
        
    
if __name__ == '__main__':
        pass
    
        