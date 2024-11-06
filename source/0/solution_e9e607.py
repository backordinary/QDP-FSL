# https://github.com/mustafamrahman/ThreeEntangledQubits3_QuantumComputing/blob/db7bdeb99b3cc94b1055ea769e6ca393304fb87c/steps/solution.py
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
    circuit.cnot(q[1], q[2])
    circuit.measure(q, c)
    # End quantum circuit

    # Execute with qiskit
    result = execute(circuit, Aer.get_backend("qasm_simulator"), shots=10000).result()
    counts = result.get_counts(circuit)

    with open("results.json", "w") as outfile:
        json.dump(counts, outfile)
