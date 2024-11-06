# https://github.com/Gallium314/LearningWithQiskit/blob/64c0a8e2dd91a617080aa4e614d67a3948a0dc64/QiskitCircuitIntro.py
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer import QasmSimulator
from qiskit.visualization import plot_histogram

simulator = QasmSimulator()

circuit = QuantumCircuit(2, 2)

#H gate on the first qubit (0)
circuit.h(0)

#CX/CNOT gate controlled by qb 0 and targeting qb 1
circuit.cx(0,1)

#Make the measurements from the circuits into classical bits (0,1)
circuit.measure([0,1],[0,1])

#compile the circuit
compiled_circuit=transpile(circuit, simulator)

#exicute the circuit 10 times
sum00 = 0
sum11 = 0
job = simulator.run(compiled_circuit, shots=1000)
result = job.result()
sum00+=result.get_counts(compiled_circuit)['00']
sum11+=result.get_counts(compiled_circuit)['11']
job = simulator.run(compiled_circuit, shots=1000)
result = job.result()
sum00+=result.get_counts(compiled_circuit)['00']
sum11+=result.get_counts(compiled_circuit)['11']
job = simulator.run(compiled_circuit, shots=1000)
result = job.result()
sum00+=result.get_counts(compiled_circuit)['00']
sum11+=result.get_counts(compiled_circuit)['11']
job = simulator.run(compiled_circuit, shots=1000)
result = job.result()
sum00+=result.get_counts(compiled_circuit)['00']
sum11+=result.get_counts(compiled_circuit)['11']
job = simulator.run(compiled_circuit, shots=1000)
result = job.result()
sum00+=result.get_counts(compiled_circuit)['00']
sum11+=result.get_counts(compiled_circuit)['11']
job = simulator.run(compiled_circuit, shots=1000)
result = job.result()
sum00+=result.get_counts(compiled_circuit)['00']
sum11+=result.get_counts(compiled_circuit)['11']
job = simulator.run(compiled_circuit, shots=1000)
result = job.result()
sum00+=result.get_counts(compiled_circuit)['00']
sum11+=result.get_counts(compiled_circuit)['11']
job = simulator.run(compiled_circuit, shots=1000)
result = job.result()
sum00+=result.get_counts(compiled_circuit)['00']
sum11+=result.get_counts(compiled_circuit)['11']
job = simulator.run(compiled_circuit, shots=1000)
result = job.result()
sum00+=result.get_counts(compiled_circuit)['00']
sum11+=result.get_counts(compiled_circuit)['11']
job = simulator.run(compiled_circuit, shots=1000)
result = job.result()
sum00+=result.get_counts(compiled_circuit)['00']
sum11+=result.get_counts(compiled_circuit)['11']


#counts from the circuit
print("\Average count for 00: " + str(sum00/10) + ", average count for 11: " + str(sum11/10))

#art!
circuit.draw()
