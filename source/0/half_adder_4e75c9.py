# https://github.com/andrei-saceleanu/Quantum_Computing/blob/768ed606975a6bd62c07242f4ba5151ff8e1196f/half_adder.py
from qiskit import QuantumCircuit,execute,Aer
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

qc_ha=QuantumCircuit(4,2)
qc_ha.x(0)
qc_ha.x(1)
qc_ha.barrier()
qc_ha.cx(0,2)
qc_ha.cx(1,2)
qc_ha.ccx(0,1,3)
qc_ha.barrier()
qc_ha.measure(2,0)
qc_ha.measure(3,1)
#qc_ha.draw(output='mpl')
counts = execute(qc_ha,Aer.get_backend('qasm_simulator')).result().get_counts()
plot_histogram(counts)
plt.show()


