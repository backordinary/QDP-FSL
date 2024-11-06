# https://github.com/jamesjcai/quantum/blob/b18436492456e62db64a8b725554e07068ef2f3e/draw_bloch_sphere.py
Ω = 3
4 + Ω

from qiskit import QuantumCircuit, assemble, Aer
from math import pi, sqrt
from qiskit.visualization import plot_bloch_multivector, plot_histogram
sim = Aer.get_backend('aer_simulator')

qc = QuantumCircuit(1)

qc.h(0)
qc.x(0)
# qc.h(0)
qc.draw()

qc.save_statevector()
qobj = assemble(qc)
state = sim.run(qobj).result().get_statevector()
plot_bloch_multivector(state).show()



