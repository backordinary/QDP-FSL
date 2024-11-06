# https://github.com/jcylim/QiskitProject/blob/92131a069db06122ee5196bd0fd3d06f8d54bfff/qiskit/gates/singleQBit_gates.py
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, QISKitError
from qiskit import available_backends, execute, register, get_backend
from qiskit.tools.visualization import circuit_drawer
from qiskit.tools.qi.qi import state_fidelity

# Useful additional packages
import matplotlib.pyplot as plt
#matplotlib inline
import numpy as np
from math import pi

q = QuantumRegister(1)
qc = QuantumCircuit(q)

# 1. u/unitary gate
'''qc.u1(pi/2,q) #u0, u1, u2, u3
#circuit_drawer(qc)
job = execute(qc, backend='local_unitary_simulator')

# 2. identity gate
qc.iden(q) #or u0(1)
job = execute(qc, backend='local_unitary_simulator')
print(np.round(job.result().get_data(qc)['unitary'], 3))

#Pauli Gates
# 3. X (bit-flip) gate
qc.x(q)
job = execute(qc, backend='local_unitary_simulator')
np.round(job.result().get_data(qc)['unitary'], 3)
print(np.round(job.result().get_data(qc)['unitary'], 3))

# 4. Y (bit- and phase-flip) gate
qc.y(q)
job = execute(qc, backend='local_unitary_simulator')
np.round(job.result().get_data(qc)['unitary'], 3)
print(np.round(job.result().get_data(qc)['unitary'], 3))

# 5. Z (phase-flip) gate
qc.z(q)
job = execute(qc, backend='local_unitary_simulator')
np.round(job.result().get_data(qc)['unitary'], 3)
print(np.round(job.result().get_data(qc)['unitary'], 3))

#Clifford Gates
# 6. hadamard gate
qc.h(q)
job = execute(qc, backend='local_unitary_simulator')
np.round(job.result().get_data(qc)['unitary'], 3)
print(np.round(job.result().get_data(qc)['unitary'], 3))

# 7. S (or sqrt(Z) phase) gate
qc.s(q)
job = execute(qc, backend='local_unitary_simulator')
np.round(job.result().get_data(qc)['unitary'], 3)
print(np.round(job.result().get_data(qc)['unitary'], 3))

# 8. Sdg (or conjugate of sqrt(Z) phase) gate
qc.sdg(q)
job = execute(qc, backend='local_unitary_simulator')
np.round(job.result().get_data(qc)['unitary'], 3)
print(np.round(job.result().get_data(qc)['unitary'], 3))

#C3 Gates
# 9. T (or sqrt(S) phase) gate
qc.t(q)
job = execute(qc, backend='local_unitary_simulator')
np.round(job.result().get_data(qc)['unitary'], 3)
print(np.round(job.result().get_data(qc)['unitary'], 3))

# 10. Tdg (or sqrt(S) phase) gate
qc.tdg(q)
job = execute(qc, backend='local_unitary_simulator')
np.round(job.result().get_data(qc)['unitary'], 3)
print(np.round(job.result().get_data(qc)['unitary'], 3))

#Standard Rotations
# 11. Rotation around X-axis
qc.rx(pi/2,q)
job = execute(qc, backend='local_unitary_simulator')
np.round(job.result().get_data(qc)['unitary'], 3)
print(np.round(job.result().get_data(qc)['unitary'], 3))

# 12. Rotation around Y-axis
qc.ry(pi/2,q)
job = execute(qc, backend='local_unitary_simulator')
np.round(job.result().get_data(qc)['unitary'], 3)
print(np.round(job.result().get_data(qc)['unitary'], 3))
'''
# 13. Rotation around Z-axis
qc.rz(pi/2,q)
job = execute(qc, backend='local_unitary_simulator')
np.round(job.result().get_data(qc)['unitary'], 3)
print(np.round(job.result().get_data(qc)['unitary'], 3))
