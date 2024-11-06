# https://github.com/asgunzi/Qiskit_transpile_passManager/blob/a2aba5084d9165a391ca20e536ff4eb870a798cd/couplingMap.py
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 02:07:04 2020

@author: asgun
"""


from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.transpiler import PassManager
from qiskit.transpiler import CouplingMap, Layout
from qiskit.transpiler.passes import BasicSwap, LookaheadSwap, StochasticSwap


circ = QuantumCircuit(4)
circ.h(0)
circ.cz(2,3)
circ.cx(0,1)
circ.cx(1,3)

coupling = [[0,1],[1,2],[0,3]]


coupling_map = CouplingMap(couplinglist = coupling)

#BAsic Swap
bs = BasicSwap(coupling_map = coupling_map)
pass_manager = PassManager(bs)
basic_circ = pass_manager.run(circ)

print(circ)
print(basic_circ)


#Lookahead
ls = LookaheadSwap(coupling_map = coupling_map)
pass_manager2 = PassManager(ls)
look_circ = pass_manager2.run(circ)

print(look_circ)


#StochasticSwap
ss = StochasticSwap(coupling_map = coupling_map)
pass_manager3 = PassManager(ss)
sto_circ = pass_manager3.run(circ)

print(sto_circ )