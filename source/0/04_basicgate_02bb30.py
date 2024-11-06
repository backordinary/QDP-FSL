# https://github.com/tofu0924/qiskit_tutorial/blob/497c5df7f1cacb4c331956c91a0d700c685e36cb/04.basicgate.py
#%%

from qiskit import QuantumCircuit, execute, Aer
from qiskit.visualization import plot_histogram
from qiskit.visualization import plot_bloch_multivector
circuit = QuantumCircuit(2,2)
circuit.x(0)
circuit.h(1)
circuit.measure([0,1],[0,1])
circuit.draw()

#%%
backend = Aer.get_backend('statevector_simulator')
result = execute(circuit, backend).result()
states = result.get_statevector()
print(states)
plot_bloch_multivector(states)