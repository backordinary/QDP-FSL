# https://github.com/salvatorezam/quantum-stuff/blob/f76262982cdad35b3a2afd3dff993835508c9ca1/tests/entangledcomm.py
from qiskit import QuantumCircuit, Aer
from qiskit.quantum_info import Statevector
import matplotlib.pyplot as plt

#
# -------------- sender -----------------------
#

MESSAGE = "00"

qc_alice = QuantumCircuit(2, 2)

# the following procedure could have worked reversed,
# encoding first and the entagling, but we would not have
# gotten any significant advantage from it, which is that alice
# could even send just the entagled quibits and only after set their state

# entagling states
qc_alice.h(1)
qc_alice.cx(1, 0)

# encoding the message
if MESSAGE[-2] == "1":
    qc_alice.x(1)
if MESSAGE[-1] == "1":
    qc_alice.z(1)


qc_alice.draw(output="mpl")  # ealice1.png image
plt.show()

ket_alice = Statevector(qc_alice)
print(ket_alice)

#
# -------------- receiver -----------------------
#

qc_bob = QuantumCircuit(2, 2)

# Bob unentangles undoing cx and h
qc_bob.cx(0, 1)
qc_bob.h(0)

# then measures
qc_bob.measure([0, 1], [0, 1])

qc_bob.draw(output="mpl")  # ebob1.png image
plt.show()

#
# -------------- channel -----------------------
#

backend = Aer.get_backend("aer_simulator")

received_message = backend.run(qc_alice.compose(qc_bob)).result().get_counts()

print("Received message: ", received_message)
