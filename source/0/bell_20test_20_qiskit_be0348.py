# https://github.com/Marduk-42/Quantum-Algorithm-Tutorials/blob/85b3958ca71851c30b335f6950ae4d9dad28b322/src/01%20-%20Bell%20test/Bell%20test%20(qiskit).py
from qiskit import *
import matplotlib.pyplot as plt

qreg = QuantumRegister(2)
creg = ClassicalRegister(2)
circuit = QuantumCircuit(qreg,creg)

circuit.reset(qreg[0])
circuit.reset(qreg[1])
circuit.h(qreg[0])
circuit.cx(qreg[0],qreg[1])
circuit.measure(qreg[0],creg[0])
circuit.measure(qreg[1],creg[1])

backend = Aer.get_backend('qasm_simulator')
job = execute(circuit, backend, shots=2048)
result = job.result()

ax1 = circuit.draw("mpl")
ax1.suptitle("Bell test circuit")
ax2 = visualization.plot_histogram(result.get_counts(circuit))
ax2.suptitle("Results")
plt.show()
