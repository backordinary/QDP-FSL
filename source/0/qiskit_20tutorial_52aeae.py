# https://github.com/TimVroomans/Quantum-Mastermind/blob/32a6f6cf8daa40faaa378a5fee0584a8469fc4c2/Qiskit%20tutorials/Qiskit%20tutorial.py
from qiskit import *
from qiskit.visualization import plot_histogram,plot_state_city
import matplotlib.pyplot as plt

# Circuit that generates GHZ state
circ = QuantumCircuit(3)
circ.h(0)
circ.cx(0,1)
circ.cx(0,2)

circ.draw(output='mpl',filename='circuit.png')

# Circuit that measures final state
meas = QuantumCircuit(3,3)
meas.barrier(range(3))
meas.measure(range(3),range(3))

meas.draw(output='mpl',filename='measure.png')

# Full circuit
qc = circ + meas
qc.draw(output='mpl',filename='full circuit.png')

# Define job and execute
method = 'qasm_simulator'
shots = 1024
backend = Aer.get_backend(method)
if method == 'qasm_simulator':
    job = execute(qc, backend, shots=shots)
else:
    job = execute(circ, backend)

# Obtain results and visualise
result = job.result()
if method == 'statevector_simulator':
    outputstate = result.get_statevector(circ, decimals=3)
    print(outputstate)
    print(plot_state_city(outputstate))
elif method == 'unitary_simulator':
    fullunitary = result.get_unitary(circ, decimals=3)
    print(fullunitary)
elif method == 'qasm_simulator':
    counts = result.get_counts(qc)
    print(counts)
    plot_histogram(counts)
else:
    print('Please choose backend.')
    
plt.show()