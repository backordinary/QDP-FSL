# https://github.com/SuperClassical/QC-codespaces-template/blob/8b3d26b771a19095187e01588f491fe60b56b7b9/examples/qiskit/hello_world.py
from qiskit import QuantumCircuit, assemble, Aer
from qiskit.visualization import plot_histogram, plot_bloch_vector
from math import sqrt, pi

# Simple Qiskit Hello World program copied from the tutorial for demonstration purposes

sim = Aer.get_backend('aer_simulator')
qc = QuantumCircuit(1)  
initial_state = [0, 1]  
qc.initialize(initial_state, 0) 
qc.draw()  # will not work in codespaces environment
qc.save_statevector()  
qobj = assemble(qc)    
result = sim.run(qobj).result() 
out_state = result.get_statevector()
print(out_state)