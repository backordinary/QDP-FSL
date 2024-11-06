# https://github.com/f-fathurrahman/ffr-quantum-computing/blob/2ee318932a9c8b395993e10dfe4d9f2cb1d52c28/qiskit/01_state.py
from qiskit import QuantumRegister, QuantumCircuit, Aer, execute

q = QuantumRegister(1)
hello_qubit = QuantumCircuit(q)

hello_qubit.id(q[0])


S_simulator = Aer.backends(name="statevector_simulator")[0]

job = execute(hello_qubit, S_simulator)

