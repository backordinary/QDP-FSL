# https://github.com/reet-pandey/reet-pandey/blob/8f90d055b05e92af14c8cf31bb57335ffdbc398a/corelation.py
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 00:40:04 2022

@author: HP
"""

import qiskit.quantum_info as qi
from qiskit.circuit.library import FourierChecking
from qiskit.visualization import plot_histogram

f=[1,-1,-1,-1]
g=[1,1,-1,-1]

circ=FourierChecking(f=f,g=g)
circ.draw()

zero = qi.Statevector.from_label('00')
sv = zero.evolve(circ)
probs = sv.probabilities_dict()
plot_histogram(probs)