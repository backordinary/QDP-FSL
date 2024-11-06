# https://github.com/benc2/BachelorProjectVQA/blob/d359b58a5df86085990cd2171eb142df368caf62/measuring_phase/stddev_datagen.py
import csv
import numpy as np
from qiskit import *
from qiskit.providers.aer import AerSimulator
from qiskit.test.mock import FakeParis


def counts_to_array(counts):
    n = counts.memory_slots
    counts = counts.int_outcomes()
    return np.array([counts.get(ell, 0) for ell in range(2**n)])


def gen_state_circuit(theta):  # generates state (|0⟩ + e^iθ |1⟩) / √2
    circ = QuantumCircuit(1)
    circ.h(0)
    circ.p(theta, 0)  # p gate (aka u1) adds phase factor to |1⟩ state
    return circ


def measuring_circuit_ghz(axis, N):
    q = QuantumRegister(N)
    c = ClassicalRegister(1)
    circ = QuantumCircuit(q, c)
    if axis == 'y':
        circ.sdg(0)
    elif axis != 'x':
        raise ValueError("Invalid axis")
    if N == 1:
        circ.h(0)
        circ.measure(0, c)
    else:
        for i in range(N):
            circ.h(i)
            circ.measure(i, c)
            circ.z(N-1).c_if(c, 1)
    return circ


def measuring_circuit_1qubit(axis, N=1):  # meas
    circ = QuantumCircuit(N)
    if axis == 'y':
        circ.sdg(0)
    elif axis != 'x':
        raise ValueError("Invalid axis")
    circ.h(0)
    circ.measure_all()
    return circ


def estimate_theta(state_circuit, shots=1024, backend_sim=Aer.get_backend('qasm_simulator')):
    # estimates theta in range [-π, π]
    N = state_circuit.num_qubits
    shots //= 2   # divide by 2 to spread across x and y
    # simulate results for cos(θ)
    circ = state_circuit.compose(measuring_circuit_ghz('x', N))  # combine state-generating- & measurement circuit
    circ = transpile(circ, backend_sim)
    counts = counts_to_array(execute(circ, backend_sim, shots=shots).result().get_counts())
    x = 2 * counts['0'] / shots - 1

    # simulate results for sin(θ)
    circ = state_circuit.compose(measuring_circuit_ghz('y', N))
    circ = transpile(circ, backend_sim)
    counts = counts_to_array(execute(circ, backend_sim, shots=shots).result().get_counts())
    y = 2 * counts['0'] / shots - 1

    theta_estimate = np.arctan2(y, x)
    return theta_estimate


true_theta = np.pi/6
noisy_sim = AerSimulator.from_backend(FakeParis())
n_shots = 1024
n_rep = 100


circ1 = QuantumCircuit(1, 1)
circ1.h(0)
circ1.p(true_theta, 0)


csvfile = open(f"measurements_unentangled.csv", 'w')
writer = csv.writer(csvfile)
writer.writerow(["ideal", "noise", "ideal", "noise"])
for j in range(n_rep):
    est_ideal = estimate_theta(circ1, shots=n_shots)
    est_noise = estimate_theta(circ1, shots=n_shots, backend_sim=noisy_sim)
    writer.writerow([est_ideal, est_noise])
exit()
print(10)
csvfile = open(f"measurements_entangled_singa_N={10}.csv", 'w')
writer = csv.writer(csvfile)
writer.writerow(["ideal", "noise"])
circ = QuantumCircuit(10, 1)
circ.h(0)  # first qubit in equal superposition
circ.cx(0, range(1, 10))  # match other qubits -> GHZ
circ.p(true_theta, range(10))  # add angle to |1> state
for j in range(n_rep):
    print('    ', j)
    t = estimate_theta(circ, shots=n_shots//10)
    if t < 0:
        t += 2*np.pi
    theta_entangled_ideal = t/10
    t = estimate_theta(circ, shots=n_shots//10, backend_sim=noisy_sim)
    if t < 0:
        t += 2*np.pi
    theta_entangled_noise = t/10
    writer.writerow([theta_entangled_ideal, theta_entangled_noise])

csvfile.close()

print('Done!')

