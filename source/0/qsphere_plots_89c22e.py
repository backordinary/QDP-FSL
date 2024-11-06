# https://github.com/Julio-Medina/Seminario/blob/14aaf1f5337af7ea80977c0592fac83b3cc20edc/Qiskit/Qsphere-plots.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 16 12:42:03 2022

@author: julio
"""

import numpy as np 
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_state_qsphere

n=2
qc=QuantumCircuit(n)
#qc.h(0)
qc.cx(0,1)
#qc.x(1)
statevec=Statevector.from_instruction(qc).data
print(statevec)
qc.draw(output='mpl')
print(qc.draw(output='text'))
plot_state_qsphere(statevec)

