# https://github.com/AndersHR/qem__master_thesis/blob/b032a90b683558404a6408fc9570850400c8d12b/uccsd_state_ansatze.py
from qiskit import *
from qiskit.circuit import Parameter, ParameterVector
from qiskit.quantum_info.operators import Operator

from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Unroller

import numpy as np
from typing import *

def cx_stair(qc, param):
    qc.cx(0,1)
    qc.cx(1,2)
    qc.cx(2,3)
    qc.rz(- param, 3)
    qc.cx(2,3)
    qc.cx(1,2)
    qc.cx(0,1)

def section(qc, param, h_q, rx_q):
    qc.h(h_q)
    qc.rx(np.pi / 2, rx_q)

    cx_stair(qc, param)

    qc.rx(- np.pi / 2, rx_q)
    qc.h(h_q)

def section_2(qc, param1, param2, q1, q2):
    qc.rx(np.pi / 2, q1)
    qc.h(q2)

    qc.cx(q1, q2)
    qc.rx(- param1, q2)
    qc.cx(q1, q2)

    qc.rx(- np.pi / 2, q1)
    qc.h(q2)

    qc.rx(np.pi / 2, q2)
    qc.h(q1)

    qc.cx(q1, q2)
    qc.rx(param2, q2)
    qc.cx(q1, q2)

    qc.rx(- np.pi / 2, q2)
    qc.h(q1)

def get_h2_uccsd_ansatz(param_name: str = "p", tot_qubits: int = 4, tot_clbits: int = 4):
    params = ParameterVector(param_name, 12)
    qc = QuantumCircuit(tot_qubits, tot_clbits)

    #qc.x(1)
    #qc.x(3)

    # FIRST HALF

    section(qc, - params[0], [0,2,3], [1])

    section(qc, - params[1], [0,1,2], [3])

    section(qc, - params[2], [0], [1,2,3])

    section(qc, - params[3], [2], [0,1,3])

    section(qc, params[4], [3], [0,1,2])

    section(qc, params[5], [1], [0,2,3])

    section(qc, params[6], [0,1,3], [2])

    section(qc, params[7], [1,2,3], [0])

    #

    section_2(qc, params[8], params[9], 2, 3)

    section_2(qc, params[10], params[11], 0, 1)

    return qc, params

def get_h2_uccsd_errordetectcircuit(tot_qubits: int = 8, tot_clbits: int = 6):
    qc = QuantumCircuit(tot_qubits, tot_clbits)

    # Parity check 1
    qc.cx(1,0)
    qc.cx(0,4)

    qc.cx(4,6)

    qc.cx(0,4)
    qc.cx(1,0)

    # Parity check 2
    qc.cx(2,3)
    qc.cx(3,5)

    qc.cx(3,5)
    qc.cx(2,3)

    qc.cx(4,7)

    qc.measure(6,4)
    qc.measure(7,5)

def get_h2_uccsd_errordetectcircuit_alt(tot_qubits: int = 6, tot_clbits: int = 6):
    qc = QuantumCircuit(tot_qubits, tot_clbits)

    qc.cx(0,4)
    qc.cx(1,4)

    qc.cx(2,5)
    qc.cx(3,5)

    qc.measure([4,5],[4,5])

    return qc, [4,5]

def decision_rule_uccsd_error_detect(mmt_str: str):
    error_detected = False
    if mmt_str == "01":
        error_detected = True
    elif mmt_str == "10":
        error_detected = True
    elif mmt_str == "00":
        error_detected = True
    return error_detected