# https://github.com/vicvaleeva/qiskit-learn/blob/e8d179369e6c3f4ba815604f8183cec5044a8c7a/MultipleQubits/gammaplus.py
from qiskit import Aer, execute, QuantumCircuit

qc = QuantumCircuit(2)

qc.h(0)
qc.x(1)
qc.cx(0, 1)

print(qc)

backend = Aer.get_backend('statevector_simulator')
vector = execute(qc, backend).result().get_statevector()
print(vector)
print()
backend = Aer.get_backend('unitary_simulator')
unitary = execute(qc, backend).result().get_unitary()
print(unitary)