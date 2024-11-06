# https://github.com/Quantum-Computing-Philippines/ibm-quantum-coin-flipper/blob/4b89151f5b1cdd6989b582e406e758d51ef116e4/sessionfiles/aer_simulation.py
from qiskit import QuantumRegister, ClassicalRegister
from qiskit import QuantumCircuit, Aer, execute
from qiskit.tools.visualization import plot_histogram
from IPython.display import display

import matplotlib.pyplot as plt
plt.show()

q = QuantumRegister(1)
c = ClassicalRegister(1)
qc = QuantumCircuit(q, c)

qc.h(q[0])
qc.measure(0, 0)
display(qc.draw('mpl'))
print(Aer.backends())
backend = Aer.get_backend('qasm_simulator')
job = execute(qc, backend, shots=1000)
result = job.result()
counts = result.get_counts(qc)
print(counts)
display(plot_histogram(counts))