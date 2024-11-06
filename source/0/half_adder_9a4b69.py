# https://github.com/AdamNaoui/LOG6953C_TPs/blob/343e81dd6e3f546f25ba88e67f61d62502ae68a5/TP1/half_adder.py
import numpy as np
from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
from qiskit_aer import QasmSimulator
from qiskit.visualization import plot_histogram

# Use Aer's qasm_simulator
simulator = QasmSimulator()
inputs = QuantumRegister(2, name='inputs')
const_0 = QuantumRegister(1, name='const_0')
sum_carry_res = ClassicalRegister(2, name='sum_carry')

# Create a Quantum Circuit acting on the q register
circuit = QuantumCircuit(inputs, const_0, sum_carry_res)

circuit.h(inputs)  # Apply Hadamard gate to qubit 0 (A)


circuit.ccx(inputs[0], inputs[1],
            const_0)  # Apply CCX gate to qubit A, B, 0 in order to get Carry output (classical AND)
circuit.cx(inputs[0], inputs[1])  # Apply CX gate to qubit A, B in order to get Sum output (classical XOR)

# Map the quantum measurement to the classical bits
circuit.measure([inputs[1], const_0[0]],
                [sum_carry_res[0], sum_carry_res[1]])  # Sum will be at index 0, Carry will be at index 1

compiled_circuit = transpile(circuit, simulator)

# Execute the circuit on the qasm simulator
job = simulator.run(compiled_circuit, shots=10000)

# Grab results from the job
result = job.result()

# Returns counts
counts = result.get_counts(compiled_circuit)

# Draw the circuit
circuit.draw('mpl', filename='half_adder_circuit.png')

plot_histogram(counts, filename='half_adder_histogram.png', title='Half Adder Histogram', bar_labels=True)
