# https://github.com/BenLirio/quantum_computation/blob/5f32e561449c7ee8625d68ac70c9db68ee03377f/qiskit/main.py
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer import QasmSimulator
from qiskit.visualization import plot_histogram

simulator = QasmSimulator()

circuit = QuantumCircuit(2,2)

circuit.h(0)
circuit.h(1)
circuit.cx(0, 1)
circuit.h(0)

circuit.measure([0,1], [0,1])

compiled_circuit = transpile(circuit, simulator)

job = simulator.run(compiled_circuit, shots=1000)

result = job.result()

counts = result.get_counts(compiled_circuit)
print(f"Total count for 00 and 11 are {counts}")
print(circuit.draw(output="text"))
