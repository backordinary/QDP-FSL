# https://github.com/richardbowmaker/qubits/blob/c21a992f40d394b8a4660a16569073138df1fedb/circuit2.py

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer import QasmSimulator
# from qiskit.visualization import plot_histogram

def circuit2():

    print("\nCircuit 2")

    # Use Aer's qasm_simulator
    simulator = QasmSimulator()

    # Create a Quantum Circuit acting on the q register
    circuit = QuantumCircuit(3, 3, name="Circuit2")

    # Add a H gate on qubit 0
    circuit.h(0)

    # Add a CX (CNOT) gate on control qubit 0 and target qubit 1
    circuit.cx(0, 1)

    # Add a CX (CNOT) gate on control qubit 0 and target qubit 1
    circuit.cx(0, 2)

    # Map the quantum measurement to the classical bits
    circuit.measure([0,1,2], [0,1,2])
    #
    # compile the circuit down to low-level QASM instructions
    # supported by the backend (not needed for simple circuits)
    compiled_circuit = transpile(circuit, simulator)

    # Execute the circuit on the qasm simulator
    job = simulator.run(compiled_circuit, shots=100)

    # Grab results from the job
    result = job.result()

    # Draw the circuit
    #circuit.draw()
    print(circuit)
    circuit.draw(output='mpl', filename='E:\_Ricks\Python\Qiskit1\circuit2.png')

    # Returns counts
    counts = result.get_counts(circuit)
    print("\nTotal counts are:",counts)


