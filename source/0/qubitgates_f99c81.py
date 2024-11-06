# https://github.com/vicvaleeva/qiskit-learn/blob/e8d179369e6c3f4ba815604f8183cec5044a8c7a/QubitGates/qubitGates.py
from qiskit import *
from math import pi, sqrt
from qiskit.visualization import plot_bloch_multivector
import matplotlib.pyplot as plt

qc = QuantumCircuit(1)
qc.x(0)
print(qc)
vec = [1/sqrt(2), -1/sqrt(2)]
backend = Aer.get_backend('statevector_simulator')
out = execute(qc, backend).result().get_statevector()
plot_bloch_multivector(vec)
plt.show()