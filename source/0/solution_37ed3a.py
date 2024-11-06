# https://github.com/mustafamrahman/ThreeEntangledQubits2/blob/2565e2257437cea275957199c4db595a468732e4/steps/solution.py
from qiskit import execute, Aer, ClassicalRegister, QuantumCircuit, QuantumRegister
import json


def main():
    # Define registers and circuit
    q = QuantumRegister(3)
    c = ClassicalRegister(3)
    circuit = QuantumCircuit(q, c)

    # Quantum circuit starts here
    circuit.h(q[0])
    circuit.cnot(q[0], q[1])
    circuit.cnot(q[0], q[2])
    circuit.measure(q, c)
    # End quantum circuit

    # Execute with qiskit
    result = execute(circuit, Aer.get_backend("qasm_simulator"), shots=10000).result()
    counts = result.get_counts(circuit)

    with open("results.json", "w") as outfile:
        json.dump(counts, outfile)
