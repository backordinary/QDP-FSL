# https://github.com/arshpreetsingh/Qiskit-cert/blob/38de9feb7eac0b0fa2fbfab3ea04cbbe5fc0442c/x-gate3.py
from qiskit import qiskit, QuantumCircuit

# Create a quantum circuit with 1 qubit and 1 classical bit.
qc = QuantumCircuit(1, 1)

# Apply a NOT gate on qubit 0.
qc.h(0)
qc.y(0)
#qc.x(0)
# Measure qubit 0.
qc.measure(0,0)

job = qiskit.execute(qc, qiskit.BasicAer.get_backend('statevector_simulator'))
#print(job.result().get_counts())
print(job.result().get_statevector())
