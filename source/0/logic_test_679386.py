# https://github.com/dclarino/ngc-dd/blob/f10cd0090fccb1c76cf9ca6b41de4eb9a171acf8/test/logic_test.py
import logging
import os
import sys
from dd.autoref import BDD
import graphviz

import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import classical_function,  Int1

sys.path.append(os.path.join('../src'))
print(sys.path)


from circuit_to_logic import *

class CircuitToLogicTests(object):

    def test_and(self):
        assert and_string('True', 'True') == 'True'
        assert and_string('True', 'False') == 'False'
        assert and_string('False', 'True') == 'False'
        assert and_string('False', 'False') == 'False'
        assert and_string('False', 'q0') == 'False'
        assert and_string('q0', 'False') == 'False'
        assert and_string('q0', 'q1') == '( q0 /\\ q1 )'

    def test_or(self):
        assert or_string('True', 'True') == 'True'
        assert or_string('True', 'False') == 'True'
        assert or_string('False', 'True') == 'True'
        assert or_string('False', 'False') == 'False'
        assert or_string('False', 'q0') == 'q0'
        assert or_string('q0', 'False') == 'q0'
        assert or_string('q0', 'q1') == '( q0 \\/ q1 )'


    def test_not(self):
        bdd = BDD()
        bdd.declare('q0')

        qc = QuantumCircuit(1)
        qc.x(0)

        bdd_exp_list = []
        bdd_exp_list.append(bdd.add_expr(r'~q0'))

        bdd_act_list = get_qc_bdd(qc,bdd)

        for i in range(len(bdd_exp_list)):
            assert bdd_exp_list[i] == bdd_act_list[i], (bdd_exp_list[i],bdd_act_list[i])
        


    def test_cnot(self):
        bdd = BDD()
        bdd.declare('q0','q1')

        qc = QuantumCircuit(2)
        qc.cx(0,1)

        bdd_exp_list = []
        bdd_exp_list.append(bdd.add_expr(r'q0'))
        bdd_exp_list.append(bdd.add_expr(r"( ( ~q0 ) /\ q1 ) \/ ( q0 /\ ~q1 )"))

        bdd_act_list = get_qc_bdd(qc,bdd)
        for i in range(len(bdd_exp_list)):
            assert bdd_exp_list[i] == bdd_act_list[i], (bdd_exp_list[i], bdd_act_list[i])
        

    def test_mct(self):
        bdd = BDD()
        bdd.declare('q0','q1','q2')

        qc = QuantumCircuit(3)
        qc.ccx(0,1,2)

        bdd_exp_list = []
        bdd_exp_list.append(bdd.add_expr(r'q0'))
        bdd_exp_list.append(bdd.add_expr(r'q1'))
        bdd_exp_list.append(bdd.add_expr(r"(~q2 /\ (q0 /\ q1) ) \/ ( ( q2 /\ ~(q0 /\ q1 ) ) )"))

        bdd_act_list = get_qc_bdd(qc,bdd)
        for i in range(len(bdd_exp_list)):
            assert bdd_exp_list[i] == bdd_act_list[i], (bdd_exp_list[i], bdd_act_list[i])

    def not_test_mpmct(self):
        bdd = BDD()
        bdd.declare('q0','q1','q2','q3')

        @classical_function
        def mpmct(q0: Int1, q1: Int1, q2: Int1) -> Int1:
            return (not q0 and q1 and q2)
        
        qc = mpmct.synth()

        bdd_exp_list = []
        bdd_exp_list.append(bdd.add_expr(r'q0'))
        bdd_exp_list.append(bdd.add_expr(r'q1'))
        bdd_exp_list.append(bdd.add_expr(r'q2'))
        bdd_exp_list.append(bdd.add_expr(r"~q0 & q1 & q2"))

        bdd_act_list = get_qc_bdd(qc,bdd)

        for i in range(len(bdd_exp_list)):
            assert bdd_exp_list[i] == bdd_act_list[i], (bdd_exp_list[i], bdd_act_list[i])

            
    def test_circ(self):
        # example Qiskit Circuit
        bdd = BDD()
        bdd.declare('q0','q1','q2')

        qc = QuantumCircuit(3)
        qc.x(1)
        qc.cx(0, 1)

        bdd_exp_list = []
        bdd_exp_list.append(bdd.add_expr(r' q0'))
        bdd_exp_list.append(bdd.add_expr(r"( ( q0 ) /\ q1 ) \/ ( ~q0 /\ ~q1 )"))
        bdd_exp_list.append(bdd.add_expr(r'q2'))

        bdd_act_list = get_qc_bdd(qc,bdd)

        for i in range(len(bdd_exp_list)):
            assert bdd_exp_list[i] == bdd_act_list[i], (bdd_exp_list[i], bdd_act_list[i])

        


