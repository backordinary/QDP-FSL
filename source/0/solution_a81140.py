# https://github.com/koderat/workshop-bell-state./blob/80b11310eb228cb615ecfc1af2e66169ca2b6ec5/steps/solution.py
from qiskit import execute, Aer, ClassicalRegister, QuantumCircuit, QuantumRegister
import json


def main():
    # Define registers and circuit
    q = QuantumRegister(2)
    c = ClassicalRegister(2)
    circuit = QuantumCircuit(q, c)

    # Quantum circuit starts here
    circuit.h(q[0])
    circuit.cnot(q[0], q[1])
    circuit.measure(q, c)
    # End quantum circuit

    # Execute with qiskit
    result = execute(circuit, Aer.get_backend("qasm_simulator"), shots=10000).result()
    counts = result.get_counts(circuit)

    with open("results.json", "w") as outfile:
        json.dump(counts, outfile)
