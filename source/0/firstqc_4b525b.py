# https://github.com/salvatorezam/quantum-stuff/blob/f76262982cdad35b3a2afd3dff993835508c9ca1/tests/firstqc.py
from qiskit import QuantumCircuit
from qiskit.providers.aer import AerSimulator
import matplotlib.pyplot as plt

#
# -------------- building and visualizing the circuit ----------------------------
#

# creating a quantum circuit with 3 quibits and 3 classical bits
qc = QuantumCircuit(3, 3)

# perform X-gates (NOT) on qubits 0 & 1
qc.x([0, 2])

# measure qubits 0, 1 & 2 to classical bits 0, 1 & 2 respectively
qc.measure([0, 1, 2], [0, 1, 2])

# showing the returned value of qc.draw() with plt.show()
qc.draw(output="mpl")
plt.show()


#
# -------------- creating and running the simulation ------------------------------
#

sim = AerSimulator()  # make new simulator object

job = sim.run(qc)  # run the experiment
result = job.result()  # get the results
result.get_counts()  # interpret the results as a "counts" dictionary

print(result.get_counts())
