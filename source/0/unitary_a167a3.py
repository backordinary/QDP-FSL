# https://github.com/vicvaleeva/qiskit-learn/blob/e8d179369e6c3f4ba815604f8183cec5044a8c7a/MultipleQubits/unitary.py
from qiskit import Aer, execute, QuantumCircuit

qc = QuantumCircuit(3)
qc.h(0)
qc.z(1)
qc.x(2)

backend = Aer.get_backend('unitary_simulator')
out = execute(qc, backend).result().get_unitary()
print(out)