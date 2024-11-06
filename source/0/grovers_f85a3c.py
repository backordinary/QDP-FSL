# https://github.com/Sammyalhashe/Thesis/blob/c22cff964f1c635eb28be1130c02fe2d95e536c8/Grover/Yousef/grovers.py
import sys
if sys.version_info < (3, 5):
    raise Exception('Run with python 3')
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import available_backends, execute
from qiskit.tools.visualization import plot_histogram, plot_state
import Qconfig
from qiskit.tools.visualization import circuit_drawer
from matplotlib import pyplot as plt
import numpy as np
from scipy import linalg as la

# re-creation of Grover's algorithm from Yusuf's paper
# qp = QuantumProgram()
# qp.enable_logs()

# creating quantum register with 5 qubits
# qr = qp.create_quantum_register('qr', 9)

qr = QuantumRegister(5)
cr = ClassicalRegister(5)

# create classical registers to store measured outputs of the quantum registers
# cr = qp.create_classical_register('cr', 9)

# create the quantum qc
# qc = qp.create_qc('qc', [qr], [cr])

qc = QuantumCircuit(qr, cr)

# get the qc by name
# qc = qp.get_qc('qc')

# get the quantum and classical registers by name
# q = qp.get_quantum_register('qr')
# c = qp.get_classical_register('cr')

# specific Grover's search algorithm that was
# implemented un Yusuf's thesis
qc.x(qr[4])
qc.h(qr[0])
qc.h(qr[1])
qc.h(qr[3])
qc.h(qr[4])
qc.x(qr[1])
qc.ccx(qr[0], qr[1], qr[2])
qc.ccx(qr[2], qr[3], qr[4])
qc.ccx(qr[0], qr[1], qr[2])
qc.ccx(qr[2], qr[3], qr[4])
qc.x(qr[1])
qc.h(qr[0])
qc.h(qr[1])
qc.h(qr[3])
qc.x(qr[0])
qc.x(qr[1])
qc.x(qr[3])
qc.h(qr[3])
qc.ccx(qr[0], qr[1], qr[3])
qc.h(qr[3])
qc.x(qr[0])
qc.x(qr[1])
qc.x(qr[3])
qc.h(qr[0])
qc.h(qr[1])
qc.h(qr[3])
qc.h(qr[4])
qc.measure(qr[0], cr[0])
qc.measure(qr[1], cr[1])
qc.measure(qr[3], cr[3])
qc.measure(qr[4], cr[4])

job_sim = execute(qc, "local_qasm_simulator")
sim_result = job_sim.result()

circuit_drawer(qc, filename="Yousef.png")

# Show the results
print("simulation: ", sim_result)
print(sim_result.get_counts(qc))
