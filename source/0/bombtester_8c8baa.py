# https://github.com/Kylet1699/Quantum-Minesweeper/blob/707e54e42249e7d0ca402f6536a149566cafc952/BombTester.py
from qiskit import *
from qiskit.circuit.library import CCXGate
from qiskit.circuit.library import RXGate
import math
# print(qiskit.__qiskit_version__)

from qiskit.providers.aer import QasmSimulator

def quantum_sweeper(cycles) -> QuantumCircuit:
    qr = QuantumRegister(3, 'q')
    cr = ClassicalRegister(cycles + 1, 'c')
    qc = QuantumCircuit(qr, cr)

    qc.h(qr[0])

    theta = math.pi/cycles
    
    for cycle in range(cycles - 1):
        qc.append(RXGate(theta), [qr[1]])
        qc.ccx(qr[0], qr[1], qr[2])
        qc.measure(qr[2], cr[cycle])
        if cycle < cycles - 1:
            qc.reset(qr[2])

    qc.append(RXGate(theta), [qr[1]])
    qc.measure(qr[1], cr[cycles - 1])
    qc.measure(qr[0], cr[cycles])
    return qc

def get_count(cycle):
    simulator = QasmSimulator()
    qsweeper_circuit = quantum_sweeper(cycle)
    qsweeper_job = simulator.run(qsweeper_circuit, shots = 1)
    qsweeper_result = qsweeper_job.result()
    qsweeper_count = qsweeper_result.get_counts(qsweeper_circuit)
    return sorted(qsweeper_count.keys())

def get_probability(cycle):
    simulator = QasmSimulator()
    qsweeper_circuit = quantum_sweeper(cycle)
    qsweeper_job = simulator.run(qsweeper_circuit, shots = 1000)
    qsweeper_result = qsweeper_job.result()
    qsweeper_count = qsweeper_result.get_counts(qsweeper_circuit)
    return qsweeper_count

print(get_probability(3))