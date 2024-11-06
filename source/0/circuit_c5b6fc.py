# https://github.com/jhson989/learn-qiskit-quantum-computing/blob/60f05600072fd8a355c49c9539476213c56ce533/ex-02/circuit.py
from turtle import back
from numpy import pi
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, execute, Aer
from qiskit.visualization import plot_histogram

################################################################
### Define Circuit 
################################################################

# Configuration
N_BIT = 3
N_QUBIT = 3
backend = Aer.get_backend('qasm_simulator')
circuit = QuantumCircuit(N_QUBIT, N_BIT)

### Generate circuit
circuit.h(0) # qubit[0] -> |0> or |1>
circuit.u(0, 0, 0, 1) # qubit[1] -> |0>
circuit.u(pi, 0, 0, 2) # qubit[2] -> |1>

circuit.cx(0, 1)
circuit.cx(0, 2)


circuit.measure(range(0, N_QUBIT), range(0, N_BIT))

### Calculate results
results = execute(circuit, backend).result()
counts = results.get_counts()

### Visualize results
circuit.draw(output='mpl', style={'backgroundcolor': '#EEEEEE'})
plot_histogram(counts)
plt.show()