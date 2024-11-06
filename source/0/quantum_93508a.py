# https://github.com/Albisourous/quantumgameoflife/blob/3f320a774eae714e8e17dbdc2601239d3de8a5f8/quantum.py
import numpy as np
from math import pi
from noise import snoise2
from qiskit import QuantumCircuit, execute, Aer, IBMQ
import matplotlib.pyplot as plt


def quantum_grid(grid_size):
    provider = IBMQ.load_account()
    #backend = provider.backend.ibmq_qasm_simulator
    backend = Aer.get_backend('qasm_simulator')

    live_cells = np.zeros((grid_size, grid_size))

    for row in range(0, grid_size):
        qc = QuantumCircuit(2, 2)
        qc2 = QuantumCircuit(2, 2)
        # 00 or 11
        qc.h(0)
        qc.cx(0, 1)
        qc.ry((np.pi/3), 0)
        qc.measure([0, 1], [0, 1])

        qc2.h(0)
        qc2.cx(0, 1)
        qc2.ry((np.pi)/1.1, 1)
        qc2.measure([0, 1], [0, 1])

        job = execute(qc, backend, shots=grid_size, memory=True)
        result = job.result()
        memory = result.get_memory()

        job2 = execute(qc2, backend, shots=grid_size, memory=True)
        result2 = job2.result()
        memory2 = result2.get_memory()

        for col in range(0, grid_size):
            if memory2[col] == '11' or memory2[col] == '00':
                live_cells[row, col] = -1
            if memory[col] == '11':
                live_cells[row, col] = 1

    return live_cells
