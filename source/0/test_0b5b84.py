# https://github.com/rickapocalypse/final_paper_qiskit_sat/blob/bfd57cca11bdd3c70afb294bc74ed3e8ade27fa0/test.py
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, transpile, BasicAer, execute
from qiskit.providers.aer import QasmSimulator
from qiskit.visualization import plot_histogram


# Use Aer's qasm_simulator
simulator = QasmSimulator()

# Create a Quantum Circuit acting on the q register
circuit = QuantumCircuit(2, 2)                     # Inicializar variavel 
# Aqui eu estou iniciando com 2 qubits no estado zero; com 2 bit clássicos definicos como zero
# Circuit é o circuito quântico 
# Add a H gate on qubit 0
circuit.h(0)

# Add a CX (CNOT) gate on control qubit 0 and target qubit 1
circuit.cx(0, 1)

# Map the quantum measurement to the classical bits
circuit.measure([0,1], [0,1])

# compile the circuit down to low-level QASM instructions
# supported by the backend (not needed for simple circuits)
compiled_circuit = transpile(circuit, simulator)

# Execute the circuit on the qasm simulator
job = simulator.run(compiled_circuit, shots=1000)

# Grab results from the job
result = job.result()

# Returns counts
counts = result.get_counts(circuit)
print("\nTotal count for 00 and 11 are:",counts)

# Draw the circuit
circuit.draw(output='mpl')

plot_histogram(counts)

plt.ylabel(counts)
plt.show()



