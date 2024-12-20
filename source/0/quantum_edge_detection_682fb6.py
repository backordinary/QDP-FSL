# https://github.com/ntanetani/QIskitCampAsia2019/blob/9041fc7af907c3848cfc93b46866ffed2bc9f35b/quantum_edge_detection.py
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit, execute
from qiskit.aqua.circuits.fourier_transform_circuits import FourierTransformCircuits
from math import pi
from qiskit import Aer

def quantum_adder(circuit, epsilon):
    qubits = circuit.qubits
    n_qubits = circuit.n_qubits
    FourierTransformCircuits.construct_circuit(circuit, qubits)
    for i in range(n_qubits):
        circuit.u1(float(2 * pi * epsilon)/2**(i + 1), qubits[n_qubits - i - 1])
    FourierTransformCircuits.construct_circuit(circuit, qubits, inverse=True)
    return circuit


def quantum_edge_detection(circuit):        #gives you the quantum state where you have to measure the ancilla and obtain 0
    qubits = circuit.qubits
    ancilla = qubits[0]
    circuit.h(ancilla) 
    circuit = quantum_adder(circuit, -1)
    circuit.h(ancilla)
    circuit.x(ancilla)
    return circuit

#/!\ Vertical and horizontal encoding
'''
circuit.measure(ancilla, clbit)

backend = Aer.get_backend('qasm_simulator')
job_sim = execute(circuit, backend)
sim_result = job_sim.result()

print(sim_result.get_counts(circuit))
'''