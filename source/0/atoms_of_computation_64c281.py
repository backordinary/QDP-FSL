# https://github.com/Julio-Medina/Seminario/blob/8bc266e51433a94f2104adf0ea65292a26d2d341/Qiskit/atoms_of_computation.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 13:02:16 2022

@author: julio
"""

from qiskit import QuantumCircuit, assemble, Aer
from qiskit.visualization import plot_histogram

from qiskit_textbook.widgets import binary_widget
#binary_widget(nbits=5)
qc_output=QuantumCircuit(8)
qc_output.measure_all()
#qc_output.draw(initial_state=True, output='mpl')

sim=Aer.get_backend('aer_simulator')
result=sim.run(qc_output).result()
counts=result.get_counts()
#plot_histogram(counts)

qc_encode=QuantumCircuit(8)
qc_encode.x(7)
qc_encode.measure_all()

qc_encode.draw(output='mpl',initial_state=True)