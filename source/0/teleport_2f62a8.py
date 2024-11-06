# https://github.com/one-marker/qiskitIBM/blob/90a3ad246aae4c296b8aad26ac0c99ca83b660f6/Teleport.py
import numpy as np
import math
import BellState
from qiskit import(
  QuantumCircuit,
  QuantumRegister,
  ClassicalRegister,
  Aer,
  execute)
from qiskit.visualization import plot_histogram

def build():
    c1 = ClassicalRegister(1)
    c2 = ClassicalRegister(1)
    q = QuantumRegister(3)
    qc = QuantumCircuit(q, c1, c2)
    qc.initialize([1/math.sqrt(2), 1/math.sqrt(2)], 0)
    # qc.h(1)
    # qc.cx(1,2)
    qc = BellState.getBellsState(qc, 1, 2)
    
    
    qc.barrier()
    qc.cx(0,1)
    qc.h(0)
    qc.barrier()
    qc.measure(q[0], c1)
    qc.measure(q[1], c2)
    
    
    qc.x(2).c_if(c1, 1)
    qc.z(2).c_if(c2, 1)
    sim = Aer.get_backend('qasm_simulator')
    job = execute(qc, sim)
    res = job.result().get_counts()
    # print(res)
    
    return qc;

