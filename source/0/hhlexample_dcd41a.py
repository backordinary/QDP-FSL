# https://github.com/Cmollan/QuantumAlgorithms/blob/d46372687b35480714c7393a0e1f9b6238817a9b/HHLExample.py
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
import numpy as np

nqubits = 4  # Total number of qubits
nb = 1  # Number of qubits representing the solution
nl = 2  # Number of qubits representing the eigenvalues
t = 2.3449156
theta = 0  # Angle defining |b>

a = 1  # Matrix diagonal
b = -1 / 3  # Matrix off-diagonal

# Initialise the quantum registers
qr = QuantumRegister(nqubits)

# Initialise the classical register
cr = ClassicalRegister(nqubits)

# Create a Quantum Circuit
qc = QuantumCircuit(qr, cr)

qrb = qr[0:nb]
qrl = qr[nb:nb + nl]
qra = qr[nb + nl:nb + nl + 1]

# State preparation.
qc.ry(2 * theta, qrb[0])

# QPE with e^{iAt}
for qu in qrl:
    qc.h(qu)

qc.u1(a * t, qrl[0])
qc.u1(a * t * 2, qrl[1])

qc.u3(b * t / m, -np.pi / 2, np.pi / 2, qrb[0])

# Controlled e^{iAt} on \lambda_{1}:
params = b * t / m

qc.u1(np.pi / 2, qrb[0])
qc.cx(qrl[0], qrb[0])
qc.ry(params, qrb[0])
qc.cx(qrl[0], qrb[0])
qc.ry(-params, qrb[0])
qc.u1(3 * np.pi / 2, qrb[0])

# Controlled e^{2iAt} on \lambda_{2}:
params = b * t * 2 / m

qc.u1(np.pi / 2, qrb[0])
qc.cx(qrl[1], qrb[0])
qc.ry(params, qrb[0])
qc.cx(qrl[1], qrb[0])
qc.ry(-params, qrb[0])
qc.u1(3 * np.pi / 2, qrb[0])

# Inverse QFT
qc.h(qrl[1])
qc.rz(-np.pi / 4, qrl[1])
qc.cx(qrl[0], qrl[1])
qc.rz(np.pi / 4, qrl[1])
qc.cx(qrl[0], qrl[1])
qc.rz(-np.pi / 4, qrl[0])
qc.h(qrl[0])

# Eigenvalue rotation
t1 = (-np.pi + np.pi / 3 - 2 * np.arcsin(1 / 3)) / 4
t2 = (-np.pi - np.pi / 3 + 2 * np.arcsin(1 / 3)) / 4
t3 = (np.pi - np.pi / 3 - 2 * np.arcsin(1 / 3)) / 4
t4 = (np.pi + np.pi / 3 + 2 * np.arcsin(1 / 3)) / 4

qc.cx(qrl[1], qra[0])
qc.ry(t1, qra[0])
qc.cx(qrl[0], qra[0])
qc.ry(t2, qra[0])
qc.cx(qrl[1], qra[0])
qc.ry(t3, qra[0])
qc.cx(qrl[0], qra[0])
qc.ry(t4, qra[0])

qc.draw('mpl')