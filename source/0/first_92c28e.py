# https://github.com/Standazwicky/qosftask2/blob/1f359e29cdeb64c34cb716378ad8166238302530/first.py
from qiskit import QuantumCircuit, Aer, execute
from math import pi
import numpy as np
from qiskit.tools.visualization import plot_histogram, plot_state_city
from qiskit import BasicAer

qc=QuantumCircuit(2,2)
qc.h(0)
qc.cx(0,1)
qc.measure([0,1], [0,1])

qc.draw()

backend = BasicAer.get_backend('qasm_simulator')
job=execute(qc,backend).result()
counts = job.get_counts(qc)
plot_histogram(counts,title='Bell-State counts')
