# https://github.com/JouziP/MQITE/blob/1b9ae93f41fdbbe4d94cc02ecd394eb64b728d10/BasicFunctions/getUQUCirc.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 12:26:49 2022

@author: pejmanjouzdani
"""


from qiskit import QuantumCircuit

def getUQUCirc(circ_U, circ_Q):
    circ_UQU = QuantumCircuit.copy(circ_Q)  ## QU|0>
    circ_UQU = circ_UQU.compose(QuantumCircuit.copy(circ_U.inverse()) )## U^QU|0>
    return circ_UQU ## U^QU|0>