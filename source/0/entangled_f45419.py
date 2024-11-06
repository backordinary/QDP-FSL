# https://github.com/SergioViteri/q/blob/ce7450f58d5fa3228761852d1d4e00103dead687/entangled.py
#!/usr/bin/env python3
# coding: utf-8

import sys, os
from qiskit import QuantumCircuit, assemble, Aer
from qiskit.visualization import plot_histogram

#
# Creación del cricuito
#
circuit = QuantumCircuit(2, 2)
circuit.h(0) # Puerta lógica cuántica H 
circuit.barrier()
circuit.cx(0, 1) # Puerta CNOT entre los dos cúbits
circuit.barrier()
circuit.measure(0, 0)
circuit.measure(1, 1)
circuit.draw(output = 'mpl').savefig(os.path.join(os.path.dirname(__file__), 'img', 'entangled.jpg'))


#
# Ejecuta el circuito en el simulador y muestra los resultados
#
sim = Aer.get_backend('aer_simulator') 
result = sim.run(circuit).result()
counts = result.get_counts()
print(counts)


