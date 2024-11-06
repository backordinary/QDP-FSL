# https://github.com/sami-b95/Qiskit-Toolbox/blob/14a378f4bb3594a441ae0a2f7e0af59dd4c07adf/QuantumAddition/Test/Quantum_multiplier.py
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer
from math import pi
from QFT import quantum_fourier_transform, do_swaps


def quantum_multiplier(sigma, circuit, qubits): #Something wrong here, problem to handle
    circuit = quantum_fourier_transform(circuit, qubits)
    n_qubits = len(qubits)
    for i in range(n_qubits):
        i = n_qubits - 1 - i
        for distance in range(n_qubits - i - 1):
            distance = distance + 1
            circuit.cu1(- sigma * pi / 2**distance, qubits[distance + i], qubits[i])
        circuit.h(qubits[i])
    circuit = do_swaps(circuit, qubits)
    return circuit

q = QuantumRegister(1)
c = ClassicalRegister(1)
circuit = QuantumCircuit(q, c)

circuit.x(q)
circuit = quantum_multiplier(10, circuit, q)
circuit.measure(q, c)

backend = Aer.get_backend('qasm_simulator')

job_sim = execute(circuit, backend, shots=10000)

sim_result = job_sim.result()
print(sim_result.get_counts(circuit))