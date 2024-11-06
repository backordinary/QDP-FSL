# https://github.com/bzkarimi/Quantum-Computing/blob/7d30a7c19df17107d2e88e6fd188b05ce4b39491/basics/q01.py
#!/usr/bin/env python 

'''
Quantum Entanglement
|0> ---> (1/2)**0.5 (|00> + |11>)
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from qiskit import QuantumCircuit, execute, Aer
from qiskit.visualization import plot_histogram

show = 'plot'

simulator = Aer.get_backend('qasm_simulator')
circuit = QuantumCircuit(2,2)
circuit.h(0)
circuit.cx(0, 1)
circuit.measure([0,1], [0,1])
job = execute(circuit, simulator, shots=1000)
result = job.result()
counts = result.get_counts(circuit)
print("\nTotal count for 00 and 11 are:", counts)
if (show == 'plot'):
    circuit.draw(output='mpl')
    plt.show()
elif (show == 'terminal'):
    print(circuit.draw())
