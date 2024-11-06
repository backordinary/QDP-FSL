# https://github.com/tt-nakamura/bell/blob/8cef96e0174030d8e42c742e78f86e7d7a9aef90/fig2.py
import matplotlib.pyplot as plt
from math import pi
from qiskit import QuantumCircuit

circuits = []
c = QuantumCircuit(2, name='circuit%d'%0)
c.h(0); c.cx(0,1)
circuits.append(c)
c = QuantumCircuit(2, name='circuit%d'%1)
c.h(0); c.cx(0,1)
circuits.append(c)
c = QuantumCircuit(2, name='circuit%d'%2)
c.h(0); c.cx(0,1)
circuits.append(c)

circuits[0].ry( pi/3, 0)
circuits[1].ry(-pi/3, 1)
circuits[2].ry( pi/3, 0)
circuits[2].ry(-pi/3, 1)
for c in circuits: c.measure_all()

for c in circuits:
    ax = plt.figure().add_subplot()
    c.draw('mpl', ax=ax)
