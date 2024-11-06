# https://github.com/kushvp/quantum-circuit-simulation/blob/be2d04f4d0ed6c2350d48961d7b02879bc8c76b9/QuantumCircuit.py
import qiskit as q
from math import pi

qr = q.QuantumRegister(1, 'q')
cr = q.ClassicalRegister(1, 'c')
circuit = q.QuantumCircuit(qr, cr)

circuit.measure(qr, cr)
print(circuit)

# print(q.Aer.get_backends())
qasm_simulator = q.Aer.get_backend('qasm_simulator')
qasm_result = q.execute(circuit, backend=qasm_simulator, shots=1024).result()
statevector_simulator = q.Aer.get_backend('statevector_simulator')
statevector_result = q.execute(circuit, backend=statevector_simulator).result()        
print('statevector_result: ', statevector_result.get_statevector())

# unitary_simulator = q.Aer.get_backend('unitary_simulator')
# unitary_result = q.execute(circuit, backend=unitary_simulator).result()
# print('unitary_result: ', *unitary_result.get_unitary())

