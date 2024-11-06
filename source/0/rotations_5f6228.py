# https://github.com/Schwarf/qiskit_fundamentals/blob/c95f00e69f605408f8f0b2a535eaa09efae716c4/one_qubit_gates/rotations.py
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, assemble, Aer
from qiskit.visualization import plot_bloch_multivector
from math import pi

sim = Aer.get_backend('aer_simulator')

#  U(θ,ϕ,λ)=[[       cos(θ/2), −exp(iλ)sin(θ/2)   ],
#            [exp(iϕ)sin(θ/2), exp(i(ϕ+λ))cos(θ/2)]]

qubit = 0
theta = pi/2
phi = 0
lamb = pi
qc1 = QuantumCircuit(1)
qc1.r(theta, phi, qubit)
qc1.draw()
qc1.save_statevector()
qobj_h = assemble(qc1)
state = sim.run(qobj_h).result().get_statevector()
plot_bloch_multivector(state, title="Rotation?")

plt.show()
