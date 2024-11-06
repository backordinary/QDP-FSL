# https://github.com/vicvaleeva/qiskit-learn/blob/e8d179369e6c3f4ba815604f8183cec5044a8c7a/PhaseKickback/exer3.py
from qiskit import Aer, execute, QuantumCircuit
from qiskit.visualization import plot_bloch_multivector
from math import pi
import matplotlib.pyplot as plt

qc = QuantumCircuit(2)
qc.x(0)
qc.x(1)
qc.cu1(pi/4, 0, 1)

backend = Aer.get_backend('statevector_simulator')
out = execute(qc, backend).result().get_statevector()
print(out)
plot_bloch_multivector(out)
plt.show()