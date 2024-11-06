# https://github.com/arshpreetsingh/Qiskit-cert/blob/b2a93d296ee45646bd428570ffa668ea49534398/x-gate.py
from qiskit import qiskit, QuantumCircuit

# Create a quantum circuit with 1 qubit and 1 classical bit.
qc = QuantumCircuit(3, 3)
qc.x(0)
# 100,
# 
#qc.h(0)
#qc.h(1)
# 11, 00
# 01, 10
#qc.x(1) # 1
# If
# AND gate for two QUbits!
#qc.x(1)
# 1
# Apply a NOT gate on qubit 0.
qc.ccx(0,1,2)  # Apply H-Gate! to get Super-Positions!
#qc.y(1)  # now state 01, 11, 10, 00
#qc.h(2)
# 010,011,
# {'011': 257, '110': 242, '111': 267, '010': 258}
# 010, 111,110,011
#
## Measurement Happens from Right to LEft.
# Measure qubit 0.
qc.measure(range(2), range(2))
#qc.measure(1, 1)
# {'00': 260, '11': 263, '01': 240, '10': 261}
#qc.measure_all()
# {'00 00': 281, '11 00': 249, '10 00': 242, '01 00': 252}
job = qiskit.execute(qc, qiskit.BasicAer.get_backend('qasm_simulator'))
print(job.result().get_counts())