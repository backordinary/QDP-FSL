# https://github.com/salvatorezam/quantum-stuff/blob/f76262982cdad35b3a2afd3dff993835508c9ca1/tests/qadder.py
from qiskit import QuantumCircuit
from qiskit.providers.aer import AerSimulator
import matplotlib.pyplot as plt

#
# -------------- building and visualizing the circuit ----------------------------
#

# creating quantum circuit with 4 qubits and 2 classical bits
qc = QuantumCircuit(4, 2)

# encoding input '11' in the circuit
qc.x([0, 1])

# implement adder logic
qc.cx(0, 2)
qc.cx(1, 2)  # CNOT controlled by qubit 0 and targeting qubit 1
qc.ccx(0, 1, 3)  # Toffoli gate that controls qubit 3

# measuring the bottom two quibits to extract the output
qc.measure(2, 0)
qc.measure(3, 1)

qc.draw(output="mpl")
plt.show()


#
# -------------- creating and running the simulation ------------------------------
#

sim = AerSimulator()

job = sim.run(qc)
result = job.result()
print("Result: ", result.get_counts())
