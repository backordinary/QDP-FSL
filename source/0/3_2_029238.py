# https://github.com/3-count/ibm_quantam_training/blob/1b1bb2c5dc2eb54b004906414f189c6fb14fd135/1.2/3-2.py
from qiskit import QuantumCircuit, assemble, Aer
from qiskit.visualization import plot_histogram
qc_output=QuantumCircuit(8)
qc_output.measure_all()
sim=Aer.get_backend('aer_simulator')
qobj=assemble(qc_output)
result=sim.run(qobj).result()
counts=result.get_counts()
plot_histogram(counts)