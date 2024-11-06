# https://github.com/jarndejong/FTSWAP_experiment/blob/b9d9d171ea25d712d3f77285119c490b018e46e0/Circuits/circuit_Id.py
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 16:02:12 2018

@author: Jarnd
This file specifies the circuit for which process tomography is to be run using the circuitname_experiment file from the root folder.
Since the transpiler is skipped when running the experiments, special care needs to be taken when specifying a circuit,
 so that there are no gates specified that are not possible using the quantum chip specified in the main experiment file.
In other words: compilation is skipped and should be done manually, for better control of the experiment.

Furthermore, the circuit name should be specified in the create_circuit() statement, because all save functions will use this name.
"""
from qiskit import QuantumProgram
import numpy as np
# Creating program
Q_program = QuantumProgram()

# Creating registers
q = Q_program.create_quantum_register("qr", 3)
c = Q_program.create_classical_register("cr", 3)

qc = Q_program.create_circuit("Id",[q],[c])


###############################################################################
#Specify FT SWAP circuit
qc.iden(q[0])
qc.iden(q[1])
###############################################################################
# Define perfect Unitary
Unitary = np.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]])