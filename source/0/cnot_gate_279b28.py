# https://github.com/bvigerzi/learning-qiskit/blob/7fc59ec7c9f9c7e17eff390e8fdea494c346fe5d/atoms_of_computation/cnot_gate.py
from qiskit import QuantumCircuit
import os

root_dir = os.path.dirname(os.path.realpath(__file__))

qc_cnot = QuantumCircuit(2)
qc_cnot.cx(0,1)
qc_cnot.draw()

figure = qc_cnot.draw(output='mpl')

figure.savefig('{}/cnot_gate.png'.format(root_dir))
