# https://github.com/ninadgawad/quantum-computing/blob/43b7bc57567624da5453e210ef6fa27d0134accd/hello_qiskit.py
## Learn about Quantum Circuit Gates

from qiskit import *
circuit = QuantumCircuit(1,1)
circuit.x(0)
simulator = Aer.get_backend('statevector_simulator')
result = execute(circuit, backend=simulator).result()
statevector = result.get_statevector()
print(statevector)
