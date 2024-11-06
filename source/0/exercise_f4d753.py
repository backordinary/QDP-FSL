# https://github.com/mustafamrahman/workshop-bell-state/blob/040729febeeba5a39c148e150c52e165de2ef9e7/steps/exercise.py
from qiskit import execute, Aer, ClassicalRegister, QuantumCircuit, QuantumRegister
import json


def main():
    # Define registers and circuit
    q = QuantumRegister(2)
    c = ClassicalRegister(2)
    circuit = QuantumCircuit(q, c)

    # Quantum circuit starts here
    # MISSING
    # MISSING
    circuit.measure(q, c)
    # End quantum circuit

    # Execute with qiskit
    result = execute(circuit, Aer.get_backend("qasm_simulator"), shots=10000).result()
    counts = result.get_counts(circuit)

    with open("results.json", "w") as outfile:
        json.dump(counts, outfile)
