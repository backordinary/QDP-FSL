# https://github.com/wxzsan/ITE_proj/blob/010f16d6083ab14d0db801b3189b025b296f6a60/vqa/transform.py
import numpy as np
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit

def circuit2unitary(qc: QuantumCircuit) -> list:
    for gate in qc.data:
        if hasattr(gate[0], 'to_matrix'):
            print(gate[0].to_matrix())