# https://github.com/vicvaleeva/qiskit-learn/blob/e8d179369e6c3f4ba815604f8183cec5044a8c7a/MultipleQubits/state.py
from qiskit import Aer, execute, QuantumCircuit

qc = QuantumCircuit(3)
qc.h(0)
qc.h(1)
qc.h(2)

print(qc)

backend = Aer.get_backend('statevector_simulator')
out = execute(qc, backend).result().get_statevector()
print(out)