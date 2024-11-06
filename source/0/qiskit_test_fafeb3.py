# https://github.com/asu-trans-ai-lab/Learning_Quantum_Computing-QUBO_for_Multimodal_Transportation/blob/d37aaaa6f008e56648007ffc8a6558cc7dd5dd5d/QC_simulators/Qiskit_test.py
'''
https://qiskit.org/  Real quantum computers → api to IBM Q Experience https://quantum-computing.ibm.com/#

install step:
conda create -n name_of_my_env python=3
conda activate name_of_my_env
pip install qiskit
pip install qiskit[visualization]

started:https://qiskit.org/documentation/getting_started.html
'''
import numpy as np
from qiskit import(QuantumCircuit,execute,Aer)
from qiskit.visualization import plot_histogram

# Use Aer's qasm_simulator
simulator = Aer.get_backend('qasm_simulator')

# Create a Quantum Circuit acting on the q register
circuit = QuantumCircuit(2, 2)

# Add a H gate on qubit 0
circuit.h(0)

# Add a CX (CNOT) gate on control qubit 0 and target qubit 1
circuit.cx(0, 1)

# Map the quantum measurement to the classical bits
circuit.measure([0,1], [0,1])

# Execute the circuit on the qasm simulator
job = execute(circuit, simulator, shots=1000)

# Grab results from the job
result = job.result()

# Returns counts
counts = result.get_counts(circuit)
print("\nTotal count for 00 and 11 are:",counts)

# Draw the circuit
circuit.draw()

#      ┌───┐     ┌─┐
# q_0: ┤ H ├──■──┤M├───
#      └───┘┌─┴─┐└╥┘┌─┐
# q_1: ─────┤ X ├─╫─┤M├
#           └───┘ ║ └╥┘
# c: 2/═══════════╩══╩═
#                 0  1
