# https://github.com/jarndejong/FTSWAP_experiment/blob/b9d9d171ea25d712d3f77285119c490b018e46e0/filesfortijn/Analysis/Compilation_analysis.py
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 15:57:14 2018

@author: Jarnd
"""

import sys
import getpass
try:
    # sys.path.append("../../") # go to parent dir
    import Qconfig
    qx_config = {
        "APItoken": Qconfig.APItoken,
        "url": Qconfig.config['url']}
    print('Qconfig loaded from %s.' % Qconfig.__file__)
except:
    APItoken = getpass.getpass('Please input your token and hit enter: ')
    qx_config = {
        "APItoken": APItoken,
        "url": "https://quantumexperience.ng.bluemix.net/api"}
    print('Qconfig.py not found in qiskit-tutorial directory; Qconfig loaded using user input.')

import numpy as np

# importing the QISKit
from qiskit import QuantumCircuit, QuantumProgram, register


# import tomography libary
import qiskit.tools.qcvv.tomography as tomo

# useful additional packages
from qiskit.tools.visualization import plot_state
from qiskit.tools.qi.qi import outer, vectorize, state_fidelity

# Register at IBM QExperience using token
# register(qx_config['APItoken'])


# Creating registers
Q_program = QuantumProgram()

# Creating registers
q = Q_program.create_quantum_register("qr", 3)
c = Q_program.create_classical_register("cr", 3)
q2 = Q_program.create_quantum_register("qr2", 3)
c2 = Q_program.create_classical_register("cr2", 3)

qc = Q_program.create_circuit("FTSWAP", [q], [c])
qc2 = Q_program.create_circuit("FTSWAPnoncomp", [q2], [c2])


#################################################################################
# Specify FT SWAP circuit

# Swap gate between qubit 0 and 2 as 3 CX's
# CX from qubit 2 to qubit 0
qc.cx(q[2], q[0])
# CX from qubit 0 to qubit 2 is not possible, flip using hadamards
qc.h(q[0])
qc.h(q[2])
qc.cx(q[2], q[0])
qc.h(q[0])
qc.h(q[2])
# CX from qubit 2 to qubit 0
qc.cx(q[2], q[0])


# Swap gate between qubit 0 and 1 as 3 CX's
# CX from qubit 1 to qubit 0
qc.cx(q[1], q[0])
# CX from qubit 0 to qubit 1 is not possible, flip using hadamards
qc.h(q[0])
qc.h(q[1])
qc.cx(q[1], q[0])
qc.h(q[0])
qc.h(q[1])
# CX from qubit 1 to qubit 0
qc.cx(q[1], q[0])

# Swap gate between qubit 1 and 2 as 3 CX's
# CX from qubit 2 to qubit 1
qc.cx(q[2], q[1])
# CX from qubit 1 to qubit 2 is not possible, flip using hadamards
qc.h(q[1])
qc.h(q[2])
qc.cx(q[2], q[1])
qc.h(q[1])
qc.h(q[2])
# CX from qubit 2 to qubit 1
qc.cx(q[2], q[1])

################################################################################
# Specify non compiled version of FT SWAP circuit
# Swap gate between qubit 0 and 2 as 3 CX's
# CX from qubit 2 to qubit 0
qc2.cx(q2[0], q2[2])

# CX from qubit 0 to qubit 2
qc2.cx(q2[0], q2[2])

# CX from qubit 2 to qubit 0
qc2.cx(q2[2], q2[0])

# Swap gate between qubit 0 and 1 as 3 CX's
# CX from qubit 1 to qubit 0
qc2.cx(q2[1], q2[0])

# CX from qubit 0 to qubit 1
qc2.cx(q2[0], q2[1])

# CX from qubit 1 to qubit 0
qc2.cx(q2[1], q2[0])

# Swap gate between qubit 1 and 2 as 3 CX's
# CX from qubit 2 to qubit 1
qc2.cx(q2[2], q2[1])

# CX from qubit 1 to qubit 2
qc2.cx(q2[1], q2[2])

# CX from qubit 2 to qubit 1
qc2.cx(q2[2], q2[1])
################################################################################
# Set number of shots and backend
shots = 20
backendsim = 'local_qasm_simulator'
backendreal = 'ibmqx4'
###############################################################################
initial_layout_1 = {("qr", 0): ("q", 0), ("qr", 1)                    : ("q", 1), ("qr", 2): ("q", 2)}
initial_layout_2 = {("qr2", 0): ("q", 0), ("qr2", 1)                    : ("q", 1), ("qr2", 2): ("q", 2)}
coupling_map = [[1, 0], [2, 0],  [2, 1], [3, 2], [3, 4], [4, 2]]

# Compile the circuit to ibmqx4 to test
#FTSWAP_comp = Q_program.compile(['FTSWAP'],coupling_map=coupling_map ,initial_layout=initial_layout_1)
FTSWAP_comp = Q_program.compile(['FTSWAP'], backend=backendreal)
print(Q_program.get_compiled_qasm(FTSWAP_comp, 'FTSWAP'))
print(Q_program.get_compiled_configuration(FTSWAP_comp, 'FTSWAP'))

# Compile non compiled version of circuit to ibmqx4 to test
#FTSWAPnoncomp_comp = Q_program.compile(['FTSWAPnoncomp'],coupling_map=coupling_map ,initial_layout=initial_layout_2)
FTSWAPnoncomp_comp = Q_program.compile(['FTSWAPnoncomp'], backend=backendreal)
print(Q_program.get_compiled_qasm(FTSWAPnoncomp_comp, 'FTSWAPnoncomp'))
print(Q_program.get_compiled_configuration(
    FTSWAPnoncomp_comp, 'FTSWAPnoncomp'))
