# https://github.com/Marduk-42/Quantum-Algorithm-Tutorials/blob/85b3958ca71851c30b335f6950ae4d9dad28b322/src/01%20-%20Bell%20test/Bell%20test%20(qasm).py
from qiskit import Aer, execute, QuantumCircuit, visualization
import matplotlib.pyplot as plt


qc = QuantumCircuit.from_qasm_file('Bell test.qasm')
backend = Aer.get_backend('qasm_simulator')
job = execute(qc, backend, shots=2048)
result = job.result()

ax1 = qc.draw("mpl")
ax1.suptitle("Bell test circuit")
ax2 = visualization.plot_histogram(result.get_counts(qc))
ax2.suptitle("Results")
plt.show()
