# https://github.com/asgunzi/Qiskit_transpile_passManager/blob/a2aba5084d9165a391ca20e536ff4eb870a798cd/PassManager.py
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 02:01:26 2020

@author: asgun
"""

from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Unroller

circ = QuantumCircuit(3)
circ.h(0)
circ.cx(0,1)
circ.cx(0,2)


pass1 = Unroller(['u1','u2','cx'])
pm = PassManager(pass1)

new_circ = pm.run(circ)

# circ.draw(output ='mpl')
#new_circ.draw(output ='mpl')
print(circ)
print(new_circ)

new_circ.draw(output ='mpl')