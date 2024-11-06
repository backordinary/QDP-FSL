# https://github.com/bvigerzi/learning-qiskit/blob/7fc59ec7c9f9c7e17eff390e8fdea494c346fe5d/atoms_of_computation/full_adder.py
from qiskit import QuantumCircuit, execute, Aer
from qiskit.visualization import plot_histogram
import os

root_dir = os.path.dirname(os.path.realpath(__file__))

qc_ha = QuantumCircuit(4,2)
# encode inputs in qubits 0 and 1
qc_ha.x(0) # invert qubit 0 (now is value 1)
qc_ha.x(1) # invert qubit 1 (now is value 1)
qc_ha.barrier()
# use cnots to write the XOR of the inputs on qubit 2
qc_ha.cx(0,2) # qubit 0 is CNOT to qubit 2
qc_ha.cx(1,2) # qubit 1 is CNOT to qubit 2
# use ccx to write the AND of the inputs on qubit 3
qc_ha.ccx(0,1,3)
qc_ha.barrier()
# extract outputs
qc_ha.measure(2,0) # extract XOR value
qc_ha.measure(3,1)

figure = qc_ha.draw(output='mpl')

figure.savefig('{}/full_adder.png'.format(root_dir))

# 1 + 1 = 10
counts = execute(qc_ha,Aer.get_backend('qasm_simulator')).result().get_counts()
histogram = plot_histogram(counts)
histogram.savefig('{}/full_adder_result.png'.format(root_dir))
