# https://github.com/f-fathurrahman/ffr-quantum-computing/blob/2ee318932a9c8b395993e10dfe4d9f2cb1d52c28/qiskit/test_working_01.py
from qiskit import QuantumCircuit

qc = QuantumCircuit(2) # Create circuit with 2 qubits
qc.h(0)    # Do H-gate on q0
qc.cx(0,1) # Do CNOT on q1 controlled by q0
qc.measure_all()
qc.draw()

print("This code works!")