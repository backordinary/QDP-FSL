# https://github.com/arshpreetsingh/Qiskit-cert/blob/b2a93d296ee45646bd428570ffa668ea49534398/bell_stateentanglement_1.py
from qiskit import *
import pylab
from qiskit.visualization import plot_histogram
qc = QuantumCircuit(2,2)
#state_vector = [0.+1.j/sqrt(2), 1/sqrt(2)+0.j]
#qc.initialize(state_vector, 0)
#qc.measure_all()
qc.h(0)
qc.x(1)
qc.cx(0,1)
qc.measure([0,1],[0,1])
qc.draw("mpl")
from qiskit.quantum_info import Statevector

sim = Aer.get_backend('aer_simulator')
qc.save_statevector()
qobj = assemble(qc)
state = sim.run(qobj).result().get_statevector()
print("State",str(state))
counts = sim.run(qobj).result().get_counts()
print("Counts", counts)
figure = plot_histogram(counts)
pylab.show()

