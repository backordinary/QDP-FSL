# https://github.com/jean-philippe-arias-zapata/Qiskit-Toolbox/blob/541e096baccbc5435d4e85f345e68a8a6f94829c/Encoding/Test/benchmark_v1.py
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit, execute, Aer
import numpy as np
import os
os.chdir('../QRAM-Encoding')
from QRAM_encoding import qRAM_encoding
os.chdir('../Test')
from qiskit.visualization import plot_histogram
from matplotlib.pyplot import savefig

n_qubits = 5
sigma = 1
mu = 2**(n_qubits - 1)

def gaussienne(sigma, mu, nb_points):
    gaussienne = []
    for i in range(nb_points):
        gaussienne.append(np.exp(-float((i - mu)**2) / (2 * sigma)))
    gaussienne = np.array(gaussienne)/sum(gaussienne)
    return gaussienne
    

circuit = qRAM_encoding(gaussienne(sigma, mu, 2**(n_qubits)), n_qubits)

q = circuit.qubits
c = circuit.clbits
circuit.measure(q, c)

shots = 10000

backend = Aer.get_backend('qasm_simulator')

job_sim = execute(circuit, backend, shots=shots)

sim_result = job_sim.result()
plot_histogram(sim_result.get_counts(circuit)).savefig('gaussian_5_qubits.png')
