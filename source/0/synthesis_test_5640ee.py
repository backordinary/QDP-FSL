# https://github.com/dclarino/ngc-dd/blob/146e49517165681ebf22e0a078e4b9062137a5b0/test/synthesis_test.py
import logging
import os
import sys
from dd.autoref import BDD
import graphviz

import pytest
from mqt.qcec import Configuration, Strategy, verify

from qiskit import *
from qiskit.circuit import classical_function, Int1

sys.path.append(os.path.join('../src'))
print(sys.path)


from circuit_synthesis import synthesize,calculate

class SynthesisTests(object):
    def test_calc_simp(self):
        bdd = BDD()
        bdd.declare('x0','x1','x2','x3','x4')

        qr = QuantumRegister(5, 'x')
        qc = QuantumCircuit(qr)
        #Sousai paper Fig 4
        qc.cx(qr[0],qr[1])

        qc.ccx(qr[0],qr[1],qr[2])
        qc.ccx(qr[0],qr[2],qr[1])
        qc.ccx(qr[1],qr[2],qr[3])
        qc.ccx(qr[2],qr[1],qr[0])
        qc.ccx(qr[0],qr[2],qr[3])
        qc.ccx(qr[0],qr[1],qr[3])

        qc.cx(qr[1],qr[2])
        qc.cx(qr[2],qr[0])

        qc.ccx(qr[0],qr[2],qr[1])
        qc.ccx(qr[1],qr[2],qr[3])
        qc.ccx(qr[0],qr[2],qr[3])
        qc.ccx(qr[0],qr[1],qr[3])

        #reverse input
        qc.ccx(qr[0],qr[2],qr[1])        
        qc.cx(qr[2],qr[0])        
        qc.cx(qr[1],qr[2])
        qc.ccx(qr[2],qr[1],qr[0])
        qc.ccx(qr[0],qr[2],qr[1])
        qc.ccx(qr[0],qr[1],qr[2])
        qc.cx(qr[0],qr[1])

        exp_phase_function = []
        exp_phase_function.append(bdd.add_expr(("(~x0 & ~x1 & ~x2) | "
                                                "(x0 & x1 & ~x2) | "
                                                "(x0 & ~x1 & x2) ")))
        exp_phase_function.append(bdd.add_expr('False'))
        exp_phase_function.append(bdd.add_expr(r'~x0 & x1 & x2'))
        exp_phase_function.append(bdd.add_expr(("(x0 & ~x1 & ~x2) | "
                                                "(~x0 & x1 & ~x2) | "
                                                "(~x0 & ~x1 & x2) | "
                                                "(x0 & x1 & x2)")))

        act_phase_function = calculate(qc,bdd)

        for i in range(len(exp_phase_function)):
            assert exp_phase_function[i] == act_phase_function[i], (exp_phase_function[i], act_phase_function[i])


    def dont_test_paper_example(self):
        qr = QuantumRegister(5, 'x')

        #Declare BDDs
        bdd = BDD()
        bdd.declare('x1','x2','x3','f','aux1')
        phase_function = []
        phase_function.append(bdd.add_expr('False'))
        phase_function.append(bdd.add_expr('False'))
        phase_function.append(bdd.add_expr(r'x1 & x2 & !x3'))
        phase_function.append(bdd.add_expr('False')) #DCDEBUG Come back to this later

        
        qc_exp   = QuantumCircuit(qr)
        qc_exp.x(qr[1])
        qc_exp.ccx(qr[0],qr[1],qr[3])
        qc_exp.ccx(qr[2],qr[3],qr[4])
        qc_inv   = qc_exp.inverse()
        qc_exp.s(qr[4])
        qc_exp.s(qr[4])
        qc_exp   = qc_exp.compose(qc_inv)

        qc_act_arr      = synthesize(phase_function,bdd,qr)

        qc_act          = qc_act_arr[2] #DCDEBUG for now only debug the pi case

        config          = Configuration()
        config.strategy = Strategy.compilationflow

        result          = verify(qc_exp,qc_act,config)

        print(result)
        print(str(result.equivalence))
        
        assert str(result.equivalence) == "Equivalence.equivalent", (result)

