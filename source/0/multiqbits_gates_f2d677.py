# https://github.com/jcylim/QiskitProject/blob/92131a069db06122ee5196bd0fd3d06f8d54bfff/qiskit/gates/multiQBits_gates.py
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, QISKitError
from qiskit import available_backends, execute, register, get_backend
from qiskit.tools.visualization import circuit_drawer
from qiskit.tools.qi.qi import state_fidelity

# Useful additional packages
import matplotlib.pyplot as plt
#matplotlib inline
import numpy as np
from math import pi

q = QuantumRegister(3)
qc = QuantumCircuit(q)
'''
#Controlled Pauli Gates
# 1. Controlled-X (or, controlled-NOT) gate
qc.cx(q[0], q[1])
job = execute(qc, backend='local_unitary_simulator')
np.round(job.result().get_data(qc)['unitary'], 3)
print(np.round(job.result().get_data(qc)['unitary'], 3))

# 2. Controlled Y gate
qc.cy(q[0], q[1])
job = execute(qc, backend='local_unitary_simulator')
np.round(job.result().get_data(qc)['unitary'], 3)
print(np.round(job.result().get_data(qc)['unitary'], 3))

# 3. Controlled Z (or, controlled phase) gate
qc.cz(q[0], q[1])
job = execute(qc, backend='local_unitary_simulator')
np.round(job.result().get_data(qc)['unitary'], 3)
print(np.round(job.result().get_data(qc)['unitary'], 3))

# 4. Controlled Hadamard gate
qc.ch(q[0], q[1])
job = execute(qc, backend='local_unitary_simulator')
np.round(job.result().get_data(qc)['unitary'], 3)
print(np.round(job.result().get_data(qc)['unitary'], 3))

#Controlled rotation gates
# 5. Controlled rotation around Z-axis
qc.crz(pi/2,q[0],q[1])
job = execute(qc, backend='local_unitary_simulator')
np.round(job.result().get_data(qc)['unitary'], 3)
print(np.round(job.result().get_data(qc)['unitary'], 3))

# 6. Controlled phase rotation
qc.cu1(pi/2,q[0], q[1])
job = execute(qc, backend='local_unitary_simulator')
np.round(job.result().get_data(qc)['unitary'], 3)
print(np.round(job.result().get_data(qc)['unitary'], 3))

# 7. Controlled u3 rotation
qc.cu3(pi/2, pi/2, pi/2, q[0], q[1])
job = execute(qc, backend='local_unitary_simulator')
np.round(job.result().get_data(qc)['unitary'], 3)
print(np.round(job.result().get_data(qc)['unitary'], 3))

# 8. Swap rotation
qc.swap(q[0], q[1])
job = execute(qc, backend='local_unitary_simulator')
np.round(job.result().get_data(qc)['unitary'], 3)
print(np.round(job.result().get_data(qc)['unitary'], 3))

# 9. Toffoli gate
qc.ccx(q[0], q[1], q[2])
job = execute(qc, backend='local_unitary_simulator')
np.round(job.result().get_data(qc)['unitary'], 3)
print(np.round(job.result().get_data(qc)['unitary'], 3))
'''
# 10. Controlled swap gate (Fredkin Gate)
qc.cswap(q[0], q[1], q[2])
job = execute(qc, backend='local_unitary_simulator')
np.round(job.result().get_data(qc)['unitary'], 3)
print(np.round(job.result().get_data(qc)['unitary'], 3))