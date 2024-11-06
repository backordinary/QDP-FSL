# https://github.com/NiskuT/INSIDE/blob/1e98de84594da1940e6fc20a5555d807a9d8ae8b/exemple.py
# -*- coding: utf-8 -*-
"""
Created on Sun May 30 09:11:54 2021

@author: Yacine
"""

import inside
import csv
import qiskit
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
import qiskit.pulse as pulse
from qiskit.providers.aer.pulse import PulseSystemModel
from qiskit.providers.aer import PulseSimulator
from qiskit.visualization import plot_histogram
from qiskit import Aer,execute
from qiskit.compiler import assemble,transpile,schedule as scheduler
from qiskit.test.mock.backends.valencia.fake_valencia import FakeValencia

backend_mock = FakeValencia()
q1 = QuantumCircuit(3,1)
q1.x(0)
q1.h(1)
q1.cx(1,2)
q1.cx(0,1)
q1.h(0)
q1.cz(0,2)
q1.cx(1,2)
q1.h(0)
q1.cx(0,1)
q1.cx(1,2)
q1.h(1)
q1.measure(2,0)
t1 = transpile(q1,backend=backend_mock,optimization_level=0)
schedule1 = scheduler(t1,backend=backend_mock)
schedule1.draw()
plt.show()

inside.Visualization(schedule1,backend_mock,0.22)
