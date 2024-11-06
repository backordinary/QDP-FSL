# https://github.com/ahkatlio/Quantum-tunnelling/blob/87da14098cad56a7be87a798611cbe63644b8b04/quantum%20tunneling.py
from qiskit import QuantumCircuit, QuantumRegister, execute, Aer

#  quantum register with 2 qubits
qr = QuantumRegister(2)

# quantum circuit
qc = QuantumCircuit(qr)

# quantum tunneling operation
qc.x(qr[0])
qc.h(qr[1])
qc.cx(qr[0], qr[1])
qc.h(qr[1])

backend = Aer.get_backend('statevector_simulator')
result = execute(qc, backend).result()
print(result.get_statevector())
